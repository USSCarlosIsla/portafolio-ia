"""API del cotizador de obras, pensada para flujos de automatización (n8n, Make, Power Automate, ERP).

Arrancar:  uvicorn api:app --reload        Documentación interactiva: http://127.0.0.1:8000/docs
Seguridad: cabecera X-API-Key; las claves válidas van en la variable de entorno API_KEYS (separadas por comas).

Endpoints:
  GET  /v1/partidas                      catálogo de ejemplo
  GET  /v1/precios/proyeccion?fecha=...  precios proyectados por el modelo entrenado
  POST /v1/cotizaciones                  JSON con ítems y metrados → presupuesto JSON
  POST /v1/cotizaciones/excel            planilla de metrados (+ tu catálogo de APU opcional) → Excel
  POST /v1/cotizaciones/desde-texto      pedido en lenguaje natural → partidas (Claude) → presupuesto
"""
import io
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import List, Optional

import anthropic
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Security, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

import catalogo as C
from catalogo import Catalogo
from cotizador import EJEMPLO, cotizar, cotizar_con_proyeccion, exportar_excel, leer_metrados_excel

load_dotenv(Path(__file__).resolve().parent / ".env")
MAX_BYTES = 20 * 1024 * 1024
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

app = FastAPI(title="Cotizador de Obras", version="1.0.0",
              description="Presupuestos de obra (APU, metrados, GG, utilidad, IGV) con proyección de precios.")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verificar_cliente(key: Optional[str] = Security(api_key_header)) -> str:
    validas = {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()}
    if not validas:
        raise HTTPException(503, "El servidor no tiene API_KEYS configuradas.")
    if key not in validas:
        raise HTTPException(401, "X-API-Key inválida o ausente.")
    return key


class Item(BaseModel):
    item: Optional[str] = Field(None, description="Numeración jerárquica, p. ej. 01.02.03")
    descripcion: Optional[str] = None
    codigo: Optional[str] = Field(None, description="Código de partida; vacío = título/agrupador")
    metrado: Optional[float] = None


class SolicitudCotizacion(BaseModel):
    items: List[Item]
    fecha_ejecucion: Optional[date] = Field(None, description="Si es futura, se usan precios proyectados")
    gastos_generales: float = Field(C.GASTOS_GENERALES, ge=0, le=0.5)
    utilidad: float = Field(C.UTILIDAD, ge=0, le=0.5)
    incluir_detalle: bool = Field(False, description="Incluir APU e insumos en la respuesta")


class SolicitudTexto(BaseModel):
    pedido: str = Field(..., min_length=10, max_length=5000)
    fecha_ejecucion: Optional[date] = None


def _ejecutar(items, fecha, gg, ut, cat=EJEMPLO):
    try:
        if fecha:
            return cotizar_con_proyeccion(items, fecha, gastos_generales=gg, utilidad=ut, cat=cat)
        return cotizar(items, gastos_generales=gg, utilidad=ut, cat=cat)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(422, str(e))


def _resumen(cot, detalle):
    claves = ["moneda", "costo_directo", "gastos_generales", "utilidad", "subtotal", "igv", "total",
              "parametros", "partidas", "proyeccion"]
    out = {k: cot[k] for k in claves if k in cot}
    if detalle:
        out.update(apu=cot["apu"], insumos=cot["insumos"])
    return out


@app.get("/salud")
def salud():
    return {"estado": "ok", "llm_configurado": bool(os.getenv("ANTHROPIC_API_KEY"))}


@app.get("/v1/partidas")
def partidas(_c: str = Depends(verificar_cliente)):
    return [{"codigo": p.codigo, "descripcion": p.descripcion, "unidad": p.unidad} for p in EJEMPLO.partidas.values()]


@app.get("/v1/precios/proyeccion")
def proyeccion(fecha: date = Query(..., description="Mes de ejecución, p. ej. 2027-06-01"),
               _c: str = Depends(verificar_cliente)):
    from pronostico import proyectar_precios
    try:
        p = proyectar_precios(fecha)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(422, str(e))
    return {"meses_adelante": p["h"], "insumos": p["insumos"]}


@app.post("/v1/cotizaciones")
def crear_cotizacion(sol: SolicitudCotizacion, _c: str = Depends(verificar_cliente)):
    items = [i.model_dump() for i in sol.items]
    return _resumen(_ejecutar(items, sol.fecha_ejecucion, sol.gastos_generales, sol.utilidad), sol.incluir_detalle)


async def _guardar(archivo: UploadFile, carpeta: Path, nombre: str) -> Path:
    contenido = await archivo.read()
    if len(contenido) > MAX_BYTES:
        raise HTTPException(413, f"{archivo.filename} supera 20 MB.")
    if not contenido.startswith(b"PK"):  # los .xlsx son archivos zip
        raise HTTPException(415, f"{archivo.filename} no es un Excel .xlsx válido.")
    ruta = carpeta / nombre
    ruta.write_bytes(contenido)
    return ruta


@app.post("/v1/cotizaciones/excel", response_class=StreamingResponse)
async def cotizacion_excel(
    metrados: UploadFile = File(..., description="Planilla .xlsx: item, descripcion, codigo, metrado"),
    catalogo: Optional[UploadFile] = File(None, description="Tu base de APU (.xlsx con hojas insumos y apu)"),
    fecha_ejecucion: Optional[date] = Form(None),
    gastos_generales: float = Form(C.GASTOS_GENERALES),
    utilidad: float = Form(C.UTILIDAD),
    _c: str = Depends(verificar_cliente),
):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        try:
            items = leer_metrados_excel(await _guardar(metrados, tmp, "metrados.xlsx"))
            cat = Catalogo.desde_excel(await _guardar(catalogo, tmp, "catalogo.xlsx")) if catalogo else EJEMPLO
        except (ValueError, KeyError) as e:
            raise HTTPException(422, f"Error leyendo los Excel: {e}")
        cot = _ejecutar(items, fecha_ejecucion, gastos_generales, utilidad, cat)
        salida = exportar_excel(cot, tmp / "presupuesto.xlsx").read_bytes()
    headers = {"Content-Disposition": 'attachment; filename="presupuesto.xlsx"',
               "X-Total": f"{cot['total']:.2f}", "X-Costo-Directo": f"{cot['costo_directo']:.2f}"}
    return StreamingResponse(io.BytesIO(salida), media_type=XLSX, headers=headers)


class MesAvance(BaseModel):
    mes: date = Field(..., description="Mes valorizado, p. ej. 2027-03-01")
    metrados: dict[str, float] = Field(..., description="{ítem: metrado ejecutado en el mes}")


class SolicitudValorizacion(BaseModel):
    presupuesto: List[Item] = Field(..., description="Presupuesto contractual (mismos ítems de la cotización)")
    gastos_generales: float = Field(C.GASTOS_GENERALES, ge=0, le=0.5)
    utilidad: float = Field(C.UTILIDAD, ge=0, le=0.5)
    fecha_base: date = Field(..., description="Mes del presupuesto (índices Io)")
    avance: List[MesAvance]
    adelanto_directo_pct: float = Field(0.0, ge=0, le=0.3)
    programado: Optional[dict[str, float]] = Field(None, description='{"AAAA-MM": monto acumulado programado}')
    permitir_mayores_metrados: bool = False
    incluir_detalle: bool = False


def _valorizar(cot, avance, fecha_base, adelanto, programado, permitir):
    from valorizaciones import valorizar_obra
    try:
        return valorizar_obra(cot, avance, fecha_base, adelanto, programado, permitir_mayores_metrados=permitir)
    except (ValueError, KeyError) as e:
        raise HTTPException(422, str(e))


@app.post("/v1/formula-polinomica")
def formula(items: List[Item], _c: str = Depends(verificar_cliente)):
    from valorizaciones import formula_polinomica
    return formula_polinomica(_ejecutar([i.model_dump() for i in items], None, C.GASTOS_GENERALES, C.UTILIDAD))


@app.post("/v1/valorizaciones")
def valorizaciones_json(sol: SolicitudValorizacion, _c: str = Depends(verificar_cliente)):
    cot = _ejecutar([i.model_dump() for i in sol.presupuesto], None, sol.gastos_generales, sol.utilidad)
    res = _valorizar(cot, [a.model_dump() for a in sol.avance], sol.fecha_base, sol.adelanto_directo_pct,
                     sol.programado, sol.permitir_mayores_metrados)
    if not sol.incluir_detalle:
        for v in res["valorizaciones"]:
            v.pop("detalle")
    return res


@app.post("/v1/valorizaciones/excel", response_class=StreamingResponse)
async def valorizaciones_excel(
    metrados: UploadFile = File(..., description="Presupuesto contractual: planilla .xlsx de metrados"),
    avance: UploadFile = File(..., description="Avance .xlsx: columna item + una columna por mes (AAAA-MM)"),
    fecha_base: date = Form(...),
    catalogo: Optional[UploadFile] = File(None, description="Tu base de APU (.xlsx)"),
    adelanto_directo_pct: float = Form(0.0),
    gastos_generales: float = Form(C.GASTOS_GENERALES),
    utilidad: float = Form(C.UTILIDAD),
    _c: str = Depends(verificar_cliente),
):
    from valorizaciones import exportar_excel as exportar_val
    from valorizaciones import leer_avance_excel
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        try:
            items = leer_metrados_excel(await _guardar(metrados, tmp, "metrados.xlsx"))
            meses = leer_avance_excel(await _guardar(avance, tmp, "avance.xlsx"))
            cat = Catalogo.desde_excel(await _guardar(catalogo, tmp, "catalogo.xlsx")) if catalogo else EJEMPLO
        except (ValueError, KeyError) as e:
            raise HTTPException(422, f"Error leyendo los Excel: {e}")
        cot = _ejecutar(items, None, gastos_generales, utilidad, cat)
        res = _valorizar(cot, meses, fecha_base, adelanto_directo_pct, None, False)
        salida = exportar_val(res, tmp / "valorizaciones.xlsx").read_bytes()
    ultima = res["valorizaciones"][-1]
    headers = {"Content-Disposition": 'attachment; filename="valorizaciones.xlsx"',
               "X-Neto-Ultima-Valorizacion": f"{ultima['neto_a_pagar']:.2f}",
               "X-Avance-Acumulado-Pct": f"{ultima['avance_acumulado_pct']:.2f}",
               "X-Alertas": str(sum(len(v["alertas"]) for v in res["valorizaciones"]))}
    return StreamingResponse(io.BytesIO(salida), media_type=XLSX, headers=headers)


@app.post("/v1/cotizaciones/desde-texto")
def cotizacion_desde_texto(sol: SolicitudTexto, _c: str = Depends(verificar_cliente)):
    from interprete_llm import interpretar, validar
    try:
        interp = interpretar(sol.pedido)
    except anthropic.AuthenticationError:
        raise HTTPException(503, "El servidor no tiene una clave de Anthropic válida.")
    except anthropic.RateLimitError:
        raise HTTPException(429, "Demasiadas solicitudes al LLM; reintenta en unos segundos.")
    except anthropic.APIConnectionError:
        raise HTTPException(502, "No se pudo conectar con la API del LLM.")
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    items, observaciones = validar(interp)
    if not items:
        raise HTTPException(422, {"mensaje": "No se reconoció ninguna partida.", "observaciones": observaciones,
                                  "no_cubierto": interp.no_cubierto})
    cot = _ejecutar(items, sol.fecha_ejecucion, C.GASTOS_GENERALES, C.UTILIDAD)
    return {"interpretacion": interp.model_dump(), "observaciones": observaciones,
            "presupuesto": _resumen(cot, False)}
