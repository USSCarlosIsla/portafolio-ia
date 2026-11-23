"""API REST para que otras personas/sistemas consuman el extractor de facturas.

Arrancar:   uvicorn api:app --reload
Documentación interactiva (Swagger): http://127.0.0.1:8000/docs

Seguridad: cada consumidor envía su clave en la cabecera `X-API-Key`. Las claves válidas se
configuran en la variable de entorno API_KEYS (separadas por comas). La clave de Anthropic NUNCA
sale del servidor: los clientes solo conocen su propia X-API-Key.
"""
import os
import tempfile
from pathlib import Path
from typing import Literal, Optional

import anthropic
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Query, Security, UploadFile
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from esquema import IGV, ruc_valido

load_dotenv(Path(__file__).resolve().parent / ".env")

MAX_BYTES = 10 * 1024 * 1024
TIPOS = {"image/png": ".png", "image/jpeg": ".jpg"}

app = FastAPI(
    title="Extractor de Facturas",
    description="Sube la imagen de una factura electrónica peruana (SUNAT) y recibe sus datos estructurados y validados (RUC, IGV 18%, totales).",
    version="1.0.0",
)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verificar_cliente(key: Optional[str] = Security(api_key_header)) -> str:
    validas = {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()}
    if not validas:
        raise HTTPException(503, "El servidor no tiene API_KEYS configuradas.")
    if key not in validas:
        raise HTTPException(401, "X-API-Key inválida o ausente.")
    return key


class RespuestaExtraccion(BaseModel):
    motor: str
    datos: dict
    valida: bool
    errores: list[str]
    intentos: int = 1
    costo_usd: float = 0.0


@app.get("/salud")
def salud():
    return {"estado": "ok", "llm_configurado": bool(os.getenv("ANTHROPIC_API_KEY"))}


@app.post("/v1/facturas/extraer", response_model=RespuestaExtraccion)
async def extraer(
    archivo: UploadFile = File(..., description="Imagen PNG o JPEG de la factura"),
    motor: Literal["llm", "ocr"] = Query("llm", description="llm = Claude (preciso); ocr = reglas locales (gratis)"),
    _cliente: str = Depends(verificar_cliente),
):
    if archivo.content_type not in TIPOS:
        raise HTTPException(415, f"Formato no soportado ({archivo.content_type}). Usa PNG o JPEG.")
    contenido = await archivo.read()
    if len(contenido) > MAX_BYTES:
        raise HTTPException(413, "La imagen supera 10 MB.")

    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / ("factura" + TIPOS[archivo.content_type])
        ruta.write_bytes(contenido)
        if motor == "ocr":
            return _con_ocr(ruta)
        return _con_llm(ruta)


def _con_ocr(ruta):
    from extractor_ocr import extraer_factura_ocr

    d = extraer_factura_ocr(ruta)
    errores = [f"No se encontró el campo '{k}'." for k, v in d.items() if v is None]
    og, igv, tot = d.get("op_gravada"), d.get("igv"), d.get("importe_total")
    if None not in (og, igv, tot):
        if abs(og + igv - tot) > 0.05:
            errores.append("Op. gravada + IGV no coincide con el importe total.")
        if abs(og * IGV - igv) > max(0.05, 0.01 * igv):
            errores.append("El IGV no es el 18% de la op. gravada.")
    for rol in ("emisor_ruc", "cliente_ruc"):
        if d.get(rol) and not ruc_valido(d[rol]):
            errores.append(f"{rol}: RUC inválido (dígito verificador SUNAT).")
    datos = {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in d.items()}
    return RespuestaExtraccion(motor="ocr", datos=datos, valida=not errores, errores=errores)


def _con_llm(ruta):
    from extractor_llm import extraer_factura

    try:
        res = extraer_factura(str(ruta))
    except anthropic.AuthenticationError:
        raise HTTPException(503, "El servidor no tiene una clave de Anthropic válida. Prueba motor=ocr.")
    except anthropic.RateLimitError:
        raise HTTPException(429, "Demasiadas solicitudes al LLM; reintenta en unos segundos.")
    except anthropic.APIConnectionError:
        raise HTTPException(502, "No se pudo conectar con la API del LLM.")
    datos = res.factura.model_dump(mode="json") if res.factura else {}
    return RespuestaExtraccion(motor="llm", datos=datos, valida=not res.requiere_revision, errores=res.errores,
                               intentos=res.intentos, costo_usd=round(res.costo_usd, 5))
