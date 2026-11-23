# Cotizador de obras con IA (Perú) + API para automatización

**Problema:** armar el presupuesto de una obra grande (APU, metrados, gastos generales, utilidad,
IGV, resumen de insumos) toma días. Además, una obra que se cotiza hoy se ejecuta en meses, y si el
acero o la mano de obra suben en ese tiempo, se pierde margen. Este proyecto genera presupuestos
completos en segundos, con precios proyectados al mes de ejecución, y se integra en flujos de
automatización por API.

## Resultados

| Prueba | Resultado |
|---|---|
| Conjunto habitacional, 10 bloques × 8 pisos (**920 partidas**) | **S/ 24.5 millones**, Excel completo en **~1.4 s** |
| 20 bloques × 15 pisos (**3,400 partidas**) | **S/ 89.7 millones** en **~2.2 s** |
| Pronóstico de jornales (error medio a 1-12 meses) | **0.85%** frente a 2.16% si se asume que no cambian |
| Pronóstico de materiales | 1.80% frente a 1.59% de "sin cambio" (ver lecciones) |
| Cobertura del intervalo 80% de precios | 82% |
| **Valorizaciones**: 12 meses de la obra de 920 partidas | Calculadas en **0.55 s**, con alertas de atraso y de mayor metrado |
| Tests automáticos | 22/22 (12 del cotizador + 10 de valorizaciones) |

## Arquitectura

```
                       ┌──────────────── Tu base de APU (Excel / exportada de S10) ───────────────┐
Planilla de metrados ──┤                                                                         ├──► Presupuesto Excel
 (Excel jerárquico)    └──► Motor de presupuesto (código exacto) ◄── Precios proyectados (ML) ───┘    (Presupuesto, APU,
                                     ▲                                                               Insumos, Proyección)
Pedido en texto ──► Claude interpreta → partidas + EXPRESIÓN del metrado ("25*2.4") → el código la evalúa
```

**Regla de oro:** el LLM interpreta y el código calcula. Ningún precio ni metrado sale de la
"cabeza" del LLM: Claude escribe la expresión, el sistema la evalúa con un evaluador seguro y los
precios salen de los APU.

| Archivo | Qué hace |
|---|---|
| `catalogo.py` | Configuración Perú (IGV 18%, GG, utilidad), catálogo de ejemplo con 19 insumos y 15 partidas, e importación y exportación de **tu propia base de APU** en Excel. |
| `cotizador.py` | Motor: APU (cuadrilla × 8 h ÷ rendimiento), presupuesto jerárquico con subtotales por título, resumen de insumos, proyección y Excel. |
| `pronostico.py` | **Modelo entrenado** de precios de insumos a 1-12 meses, con selección de método por segmento e intervalos conformales. |
| `interprete_llm.py` | Claude convierte un pedido en texto libre en partidas, metrados, supuestos y preguntas para el cliente. |
| `generar_obra_grande.py` | Plantillas de ejemplo y una obra grande para la prueba de escala. |
| `api.py` | API REST para automatización. |
| `test_cotizador.py` | 12 tests del motor y de la API. |

## Puesta en marcha

```powershell
pip install -r requirements.txt
python pronostico.py              # entrena el modelo de precios
python generar_obra_grande.py     # plantillas + obra de ~S/ 24 M + medición de tiempos
python -m pytest test_cotizador.py
copy .env.example .env            # API_KEYS y, si usarás el intérprete, ANTHROPIC_API_KEY
uvicorn api:app --reload          # http://127.0.0.1:8000/docs
```

## Uso en un flujo de automatización

Desde n8n, Make, Power Automate o cualquier sistema que haga peticiones HTTP:

```python
import requests

# Obra grande: planilla de metrados + tu base de APU → Excel del presupuesto
r = requests.post("http://TU-SERVIDOR:8000/v1/cotizaciones/excel",
                  headers={"X-API-Key": "tu-clave"},
                  files={"metrados": open("metrados.xlsx", "rb"), "catalogo": open("mis_apu.xlsx", "rb")},
                  data={"fecha_ejecucion": "2027-06-01", "gastos_generales": 0.12, "utilidad": 0.08})
open("presupuesto.xlsx", "wb").write(r.content)
print("Total S/", r.headers["X-Total"])

# Cotización rápida a un cliente desde texto (usa Claude)
r = requests.post("http://TU-SERVIDOR:8000/v1/cotizaciones/desde-texto", headers={"X-API-Key": "tu-clave"},
                  json={"pedido": "Cerco de ladrillo de cabeza de 25 m x 2.4 m con cimiento corrido, tarrajeado y pintado"})
```

Ejemplo de flujo: llega un correo con la planilla de metrados → n8n la envía a `/v1/cotizaciones/excel` →
el Excel resultante se guarda en Drive y se avisa por WhatsApp o correo al ingeniero para su revisión.

### Formato de tus archivos

- **Base de APU** (`data/plantilla_catalogo.xlsx` es el ejemplo):
  - hoja `insumos`: `codigo, descripcion, unidad, tipo (MO/MAT/EQ), precio`
  - hoja `apu`: `partida, descripcion, unidad, rendimiento, insumo, cuadrilla, cantidad`, una fila por insumo
- **Metrados** (`data/obra_grande_metrados.xlsx`): `item, descripcion, codigo, metrado`. Las filas sin código son títulos.

## Valorizaciones mensuales (`valorizaciones.py`)

Después de ganar la obra, cada mes se valoriza lo ejecutado. El módulo toma el presupuesto contractual
(el mismo de la cotización) y la planilla de avance (`item` + una columna por mes con el metrado ejecutado):

1. **Fórmula polinómica automática**: los coeficientes salen de la incidencia de cada Índice Unificado
   (INEI) en el presupuesto: mano de obra 47, cemento 21, acero 03, ladrillo 17… GG y utilidad van al
   índice 39. Los monomios menores a 5% se agrupan, hay como máximo 8 y la suma da 1.000. Ejemplo:
   `K = 0.335·(47r/47o) + 0.264·(39r/39o) + 0.179·(03r/03o) + 0.133·(21r/21o) + 0.089·(17r/17o)`
2. **Por partida**: metrado anterior, actual, acumulado y saldo, y % de avance.
3. **Por mes**: costo directo, GG, utilidad (valorización bruta), **reajuste = V × (K − 1)**,
   **amortización del adelanto directo** proporcional (el saldo se cierra en la valorización final), IGV y neto a pagar.
4. **Alertas**:
   - **Mayor metrado**: lo ejecutado supera lo contratado. No se paga sin aprobación, salvo que se indique `permitir_mayores_metrados`.
   - **Atraso**: lo valorizado acumulado es menor al 80% de lo programado.
   - **K provisional**: el mes usa índices proyectados por el modelo y no publicados por el INEI.

```powershell
python simular_valorizaciones.py   # obra de S/ 24.5 M en 12 meses: atraso, mayor metrado, adelanto 10%
```

API: `POST /v1/valorizaciones` (JSON), `POST /v1/valorizaciones/excel` (presupuesto + avance en Excel →
Excel de valorizaciones; las cabeceras `X-Neto-Ultima-Valorizacion` y `X-Alertas` sirven para disparar
avisos en tu flujo) y `POST /v1/formula-polinomica`.

**⚠ Pendiente y por verificar antes de usarlo en obra real:**
- No se calcula la **deducción del reajuste que no corresponde por adelantos**. Su fórmula depende de la norma vigente y no la inventé.
- Falta el adelanto para **materiales**.
- Hay que confirmar los códigos de los índices y el mes de índice aplicable con el reglamento vigente.
- Los índices en `data/indices_iupc.csv` son **sintéticos**: reemplázalos por los publicados por el INEI.

## Lecciones (para la entrevista)
- **El modelo no siempre gana, y hay que medirlo.** El GBM pronostica muy bien los jornales porque aprende el aumento de cada junio. En materiales perdía contra "el precio no cambia", porque aprendió la inflación del shock 2021-22. La solución fue **elegir el método por segmento** en un periodo de calibración y reportar en un test que no se tocó. En test, la elección para materiales ("tendencia") quedó algo peor que "sin cambio", y se reporta tal cual: así pasa en producción.
- **El rango del total es conservador.** Supone que todos los insumos se mueven a la vez al extremo de su intervalo.
- **Los datos de precios son sintéticos.** Para uso real hay que reemplazar `data/historial_precios.csv` por precios reales (INEI, revista Costos o tus órdenes de compra) y reentrenar.

## Siguientes pasos
1. Cargar tu base real de APU (exportación de S10) y una planilla de una obra real ya presupuestada, y comparar el total.
2. Reentrenar el pronóstico con precios reales y añadir los **Índices Unificados del INEI** como variables.
3. Valorizaciones: deducción de reajuste por adelantos, adelanto de materiales y cargar índices INEI reales.
4. Detección de APU anómalos: rendimientos o cantidades fuera de rango frente a tu histórico.
5. Leer metrados directamente de modelos BIM (IFC) o de planos en PDF.
