# Extractor de facturas con LLM (+ API REST)

**Problema real:** las empresas reciben cientos de facturas de proveedores como foto o PDF, y alguien
las digita a mano en el sistema contable. Es un trabajo lento y con errores. Este proyecto lee la imagen y devuelve los datos
estructurados y **validados**, listos para el ERP o para Excel, y los expone como una **API** que
otras personas o sistemas pueden consumir.

## Arquitectura

```
imagen ──► Claude (visión + salida estructurada Pydantic) ──► reglas de negocio ──┬─► OK ──► JSON / Excel / API
                     ▲                                                             │
                     └──────── errores concretos ("subtotal + IVA ≠ total") ◄──────┘  (máx. 2 correcciones)
                                                                                   └─► sigue mal ──► revisión humana
```

| Archivo | Qué hace |
|---|---|
| `esquema.py` | Esquema `Factura` (Pydantic) + validaciones: sumas de líneas, subtotal + IVA = total, dígito de verificación del NIT (algoritmo DIAN). |
| `generar_facturas.py` | Genera facturas sintéticas con 3 diseños, formatos de fecha, número y NIT variados, ruido y rotación. Guarda la verdad de referencia en JSON. |
| `extractor_llm.py` | Extracción con **Claude Opus 5.5**: imagen → `Factura` validada, con bucle de **autocorrección**. |
| `extractor_ocr.py` | **Baseline sin LLM**: OCR local (RapidOCR) + regex. |
| `evaluar.py` | Exactitud campo por campo, costo por factura y exportación a Excel (`facturas` + `lineas`). |
| `api.py` | API REST (FastAPI) con autenticación por `X-API-Key`, límites de tamaño y errores HTTP claros. |
| `test_api.py` | Tests automáticos de la API (no gastan dinero). |
| `Dockerfile` | Imagen para desplegar la API en cualquier nube. |

## Puesta en marcha

```powershell
pip install -r requirements.txt
python generar_facturas.py --n 30
python evaluar.py --extractor ocr           # baseline gratis
copy .env.example .env                      # pon tu ANTHROPIC_API_KEY en .env
python evaluar.py --extractor llm --limit 5 # prueba pequeña con Claude
```

### Servir la API

```powershell
$env:API_KEYS = "mi-clave-cliente"
uvicorn api:app --reload
```

Abre http://127.0.0.1:8000/docs para probarla desde el navegador. Desde otro programa:

```python
import requests
r = requests.post("http://127.0.0.1:8000/v1/facturas/extraer?motor=llm",
                  headers={"X-API-Key": "mi-clave-cliente"},
                  files={"archivo": open("factura.png", "rb")})
print(r.json())  # {"motor": "llm", "datos": {...}, "valida": true, "errores": [], "costo_usd": ...}
```

## Resultados

| Campo | Baseline OCR + regex | Claude |
|---|---|---|
| Número, fecha, cliente | 100% | *ejecuta `evaluar.py --extractor llm`* |
| NIT y dígito de verificación | 83% | |
| Subtotal / IVA / total | 97–100% | |
| **Líneas de producto** | **0%** (no las extrae) | |
| **Factura 100% correcta** | **0%** | |

**Ojo con el baseline.** Las regex se escribieron *conociendo* los 3 diseños del generador, así que su
resultado es optimista. Con un diseño nuevo o con otra etiqueta ("Neto a pagar", "Vlr. IVA") se rompen
y toca escribir más reglas. El LLM generaliza sin reglas nuevas. Esa es la comparación que
justifica (o no) pagar por un LLM.

## Decisiones de diseño (para explicar en una entrevista)
- **Salida estructurada**: el SDK valida la respuesta contra el esquema Pydantic, así que nunca llega JSON roto al sistema.
- **Validación con reglas de negocio + autocorrección**: el LLM puede leer mal un dígito, pero la aritmética no miente. Si no cuadra, se le dan los errores exactos para que corrija; si aún falla, va a **revisión humana** en vez de entrar datos malos al ERP.
- **`effort: "low"`**: la extracción es una tarea acotada, así que se usa poco razonamiento para que sea rápida y barata. Súbelo si la evaluación muestra errores.
- **Fallbacks del servidor** (`fallbacks="default"`): si un filtro de seguridad rechazara una petición, la API la reintenta con otro modelo.
- **Caché de respuestas** en la evaluación: no se paga dos veces por la misma factura.
- **Seguridad de la API**: los clientes usan su propia `X-API-Key`; la clave de Anthropic nunca sale del servidor ni se sube a git (`.env` está en `.gitignore`).

## Retos
1. Soporta **PDF** (Claude acepta documentos PDF directamente como bloque `document`).
2. Añade un **4º diseño** al generador sin tocar el extractor y compara cuánto cae cada método.
3. Usa la **Batch API** (50% más barata) para procesar cientos de facturas de noche.
4. Guarda las facturas en una base de datos (SQLite/Postgres) y añade `GET /v1/facturas/{id}`.
5. Detecta **facturas duplicadas** (mismo NIT + número) antes de registrarlas.
