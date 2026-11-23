# Extractor de facturas electrónicas peruanas con LLM (+ API REST)

**Problema real:** una constructora o empresa recibe cientos de facturas de proveedores como PDF o foto
y alguien las digita a mano en el sistema contable para el Registro de Compras. Es lento y con errores
(un RUC mal digitado, un IGV que no cuadra). Este proyecto lee la representación impresa, devuelve los
datos **validados con las reglas de SUNAT** y los expone como una **API** para flujos de automatización.

## Arquitectura

```
imagen ──► Claude (visión + salida estructurada Pydantic) ──► validación SUNAT ──┬─► OK ──► JSON / Excel / API
                     ▲                                                            │
                     └──── errores concretos ("RUC con dígito verificador ◄──────┘  (máx. 2 correcciones)
                           inválido", "IGV ≠ 18% de la op. gravada")              └─► sigue mal ──► revisión humana
```

**Validaciones:**
- **RUC** del emisor y del cliente: 11 dígitos, prefijo 10/15/16/17/20 y dígito verificador con el algoritmo módulo 11 de SUNAT (verificado con el RUC de la propia SUNAT).
- Serie de factura (`F001`, `E001`…) y correlativo de hasta 8 dígitos.
- **IGV = 18%** de la op. gravada, op. gravada + IGV = importe total, y suma de las líneas = op. gravada.

| Archivo | Qué hace |
|---|---|
| `esquema.py` | Esquema `Factura` (Pydantic) y validaciones SUNAT. |
| `generar_facturas.py` | Facturas electrónicas sintéticas con 3 diseños, en soles y dólares, con formatos variados (S/, S/., US$, fechas como 17-set-2027) y ruido. Los RUC son válidos pero inventados. Los productos salen del catálogo del cotizador (módulo 10). |
| `extractor_llm.py` | Extracción con **Claude Opus 5.5**: imagen → `Factura` validada, con bucle de **autocorrección**. |
| `extractor_ocr.py` | **Baseline sin LLM**: OCR local (RapidOCR) + regex. |
| `evaluar.py` | Exactitud campo por campo, costo por factura y exportación a Excel (`facturas` + `lineas`). |
| `api.py` | API REST (FastAPI) con `X-API-Key`, límites de tamaño y errores HTTP claros. |
| `test_api.py` | Tests de la API (no gastan dinero). |
| `Dockerfile` | Imagen para desplegar la API. |

## Puesta en marcha

```powershell
pip install -r requirements.txt
python generar_facturas.py --n 30
python evaluar.py --extractor ocr           # baseline gratis
copy .env.example .env                      # pon tu ANTHROPIC_API_KEY en .env
python evaluar.py --extractor llm --limit 5 # prueba pequeña con Claude
$env:API_KEYS = "mi-clave-cliente"; uvicorn api:app --reload   # http://127.0.0.1:8000/docs
```

```python
import requests
r = requests.post("http://127.0.0.1:8000/v1/facturas/extraer?motor=llm",
                  headers={"X-API-Key": "mi-clave-cliente"}, files={"archivo": open("factura.png", "rb")})
print(r.json())  # {"motor": "llm", "datos": {"emisor_ruc": "20...", "igv": ..., ...}, "valida": true, ...}
```

## Resultados

| Campo | Baseline OCR + regex | Claude |
|---|---|---|
| Serie, correlativo, fecha, moneda | 100% | *pendiente: ejecuta `evaluar.py --extractor llm`* |
| RUC emisor / cliente | 100% / 100% | |
| Razón social emisor / cliente | 93% / 87% | |
| Op. gravada / IGV / importe total | 93% / 73% / 97% | |
| **Líneas de producto** | **0%** (no las extrae) | |
| **Factura 100% correcta** | **0%** | |

**Ojo con el baseline.** Las regex se escribieron *conociendo* los 3 diseños del generador, así que su
resultado es optimista. En facturas reales de cientos de proveedores distintos se rompen. Esa es la
comparación que justifica (o no) pagar por un LLM.

## ¿Qué más faltaría para usarlo en producción en Perú?

En orden de impacto:

1. **Leer el XML antes que la imagen.** Toda factura electrónica peruana tiene un **XML UBL 2.1**
   firmado que el proveedor envía junto al PDF. Si llega el XML, se lee con código exacto: costo cero y
   100% de precisión. El LLM solo hace falta para PDFs sin XML, fotos y escaneos. Es el cambio que más
   ahorra en un flujo real.
2. **Soporte de PDF.** La mayoría llega como PDF por correo. Claude acepta PDF directamente como bloque `document`.
3. **Validar contra SUNAT.** Comprobar que el RUC esté *activo y habido* y que el comprobante *exista y
   sea válido* con la consulta de validez de comprobantes de SUNAT (requiere credenciales de la empresa).
   Hoy solo se valida el formato.
4. **Detracciones (SPOT), retenciones y percepciones.** Muchas facturas de construcción y servicios
   están sujetas a detracción. Hay que extraer la leyenda, el porcentaje y la cuenta del Banco de la
   Nación, y alertar si falta. Los porcentajes y montos mínimos deben tomarse del anexo vigente de SUNAT.
5. **Otros comprobantes y operaciones**: boletas (`B001`, DNI), notas de crédito y débito, operaciones
   exoneradas, inafectas y gratuitas, anticipos, descuentos y facturas en dólares con tipo de cambio.
6. **Duplicados**: la misma combinación de RUC emisor + serie + correlativo no debe registrarse dos veces.
7. **Salida contable**: exportar al formato del **Registro de Compras (PLE / SIRE)** o al sistema contable (Concar, Siigo, ERP).
8. **Evaluación con facturas reales**: 50 a 100 facturas reales anonimizadas de distintos proveedores,
   porque las sintéticas no reemplazan esa prueba.

## Decisiones de diseño (para explicar en una entrevista)
- **Salida estructurada**: el SDK valida la respuesta contra el esquema Pydantic, así que nunca llega JSON roto al sistema.
- **Validación + autocorrección**: el LLM puede confundir un dígito del RUC, pero el dígito verificador lo detecta. Se le dan los errores exactos para que corrija; si persiste, la factura va a **revisión humana** en lugar de registrarse con datos malos.
- **`effort: "low"`**: la extracción es una tarea acotada y conviene que sea rápida y barata.
- **Fallbacks del servidor**: si un filtro rechazara una petición, la API la reintenta en otro modelo.
- **Seguridad**: los clientes usan su propia `X-API-Key`; la clave de Anthropic nunca sale del servidor ni se sube a git.
