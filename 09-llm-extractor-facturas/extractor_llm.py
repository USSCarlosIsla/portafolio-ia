"""Extractor de facturas con Claude (visión + salida estructurada + autocorrección).

Flujo:
  1. Se envía la IMAGEN de la factura a Claude, pidiendo una salida que cumpla el esquema `Factura`
     (Pydantic). El SDK valida la respuesta: si no cumple el esquema, falla en vez de devolver basura.
  2. Se aplican reglas de negocio (sumas, IVA, dígito de verificación del NIT).
  3. Si alguna regla falla, se le devuelven los errores a Claude para que revise la imagen y corrija
     (hasta MAX_CORRECCIONES veces). Si sigue fallando, la factura se marca para revisión humana.

Credenciales: variable de entorno ANTHROPIC_API_KEY (o un archivo .env en esta carpeta; ver .env.example).

Uso:  python extractor_llm.py data/facturas/factura_000.png
"""
import base64
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import anthropic
from dotenv import load_dotenv

from esquema import Factura, validar_factura

load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL = "claude-opus-5-5"
PRECIO_INPUT, PRECIO_OUTPUT = 4.00, 20.00  # USD por millón de tokens (Claude Opus 5.5)
MAX_CORRECCIONES = 2

SYSTEM = """Eres un sistema de extracción de datos de facturas de proveedores colombianos.
Lee la imagen y devuelve los datos EXACTAMENTE como aparecen, normalizados así:
- Fechas en formato ISO AAAA-MM-DD, sin importar cómo estén escritas.
- Montos como números simples: "$ 1.234.567" y "$1,234,567.00" son 1234567. En Colombia el punto suele
  separar miles y la coma los decimales; deduce el formato por el contexto de toda la factura.
- NIT: solo los dígitos antes del guion en `proveedor_nit`; el dígito tras el guion en `proveedor_dv`.
- `iva` es el VALOR del impuesto en dinero, no el porcentaje.
- Incluye TODAS las líneas de productos o servicios, en el orden en que aparecen.
Si un dato no se lee con claridad, da tu mejor lectura: luego se validará con reglas aritméticas."""


@dataclass
class Resultado:
    factura: Optional[Factura]
    intentos: int
    errores: List[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def costo_usd(self):
        return (self.tokens_in * PRECIO_INPUT + self.tokens_out * PRECIO_OUTPUT) / 1e6

    @property
    def requiere_revision(self):
        return self.factura is None or bool(self.errores)


def _imagen_b64(path: Path):
    media = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return media, base64.standard_b64encode(path.read_bytes()).decode("utf-8")


def extraer_factura(path_imagen: str, client: Optional[anthropic.Anthropic] = None) -> Resultado:
    client = client or anthropic.Anthropic()
    media, data = _imagen_b64(Path(path_imagen))
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}},
            {"type": "text", "text": "Extrae los datos de esta factura."},
        ],
    }]
    res = Resultado(factura=None, intentos=0)

    for intento in range(1 + MAX_CORRECCIONES):
        res.intentos = intento + 1
        response = client.beta.messages.parse(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM,
            messages=messages,
            output_format=Factura,
            # Extracción = tarea acotada: esfuerzo bajo es más rápido y barato; súbelo si ves errores.
            output_config={"effort": "low"},
            # Si el filtro de seguridad rechazara la petición, la API la reintenta en otro modelo.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        res.tokens_in += response.usage.input_tokens
        res.tokens_out += response.usage.output_tokens

        if response.stop_reason == "refusal":
            res.errores = ["El modelo rechazó la solicitud."]
            return res
        if response.stop_reason == "max_tokens":
            res.errores = ["Respuesta truncada por max_tokens."]
            return res

        res.factura = response.parsed_output
        res.errores = validar_factura(res.factura)
        if not res.errores:
            return res

        # Autocorrección: el modelo ve su respuesta anterior + los errores concretos y vuelve a leer la imagen.
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content":
                         "Tu extracción no pasa estas validaciones:\n- " + "\n- ".join(res.errores) +
                         "\nVuelve a mirar la imagen con cuidado y devuelve la extracción corregida."})
    return res


def main():
    if len(sys.argv) < 2:
        print("Uso: python extractor_llm.py <imagen_factura>")
        sys.exit(1)
    try:
        res = extraer_factura(sys.argv[1])
    except anthropic.AuthenticationError:
        sys.exit("API key inválida o ausente. Crea un archivo .env con ANTHROPIC_API_KEY=... (ver .env.example)")
    except anthropic.RateLimitError:
        sys.exit("Límite de uso alcanzado; espera un momento y reintenta.")
    except anthropic.APIConnectionError:
        sys.exit("No hay conexión con la API.")
    if res.factura:
        print(json.dumps(json.loads(res.factura.model_dump_json()), indent=2, ensure_ascii=False))
    print(f"\nIntentos: {res.intentos} | tokens: {res.tokens_in} in / {res.tokens_out} out | "
          f"costo ≈ ${res.costo_usd:.4f} USD")
    print("⚠ Requiere revisión humana:\n- " + "\n- ".join(res.errores) if res.requiere_revision else "✓ Factura válida")


if __name__ == "__main__":
    main()
