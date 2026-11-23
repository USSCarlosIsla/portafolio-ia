# Portafolio de AI Engineering

Ruta práctica para aprender y demostrar habilidades de **AI Engineer**: desde las matemáticas
de un modelo lineal hasta redes neuronales, aprendizaje semi-supervisado, aprendizaje continuo y OCR.

Cada proyecto se construye **dos veces cuando tiene sentido**: primero *desde cero* (NumPy, para entender
qué pasa por dentro) y luego con la librería profesional (scikit-learn / PyTorch), comparando resultados.

## Cómo usarlo

```powershell
pip install -r requirements.txt
python 01-fundamentos/01-regresion-lineal/regresion_lineal.py
```

Todos los scripts funcionan **sin internet** (usan datasets sintéticos o incluidos en scikit-learn)
y guardan sus gráficas en una carpeta `outputs/` junto al script.

## Mapa del portafolio

| # | Módulo | Proyecto | Qué aprendes | Estado |
|---|--------|----------|--------------|--------|
| 01 | Fundamentos | [Regresión lineal](01-fundamentos/01-regresion-lineal/) | Ecuación normal, descenso de gradiente, MSE/R² | ✅ |
| 01 | Fundamentos | [Regresión logística](01-fundamentos/02-regresion-logistica/) | Sigmoide, entropía cruzada, frontera de decisión | ✅ |
| 02 | Aprendizaje supervisado | [Comparativa de clasificadores](02-supervisado/) | Pipelines, validación cruzada, matriz de confusión | ✅ |
| 03 | Redes neuronales | [MLP desde cero](03-redes-neuronales/01-mlp-desde-cero/) | Backpropagation a mano | ✅ |
| 03 | Redes neuronales | [CNN con PyTorch](03-redes-neuronales/02-cnn-pytorch/) | Convoluciones, entrenamiento en PyTorch | ✅ |
| 04 | No supervisado | [K-Means y PCA](04-no-supervisado/) | Clustering, reducción de dimensionalidad | ✅ |
| 05 | Semi-supervisado | [Pocas etiquetas](05-semi-supervisado/) | Self-training, Label Spreading | ✅ |
| 06 | Aprendizaje continuo | [Olvido catastrófico](06-aprendizaje-continuo/) | Tareas secuenciales, replay buffer, aprendizaje online | ✅ |
| 07 | Visión / OCR | [OCR de dígitos](07-ocr/) | Segmentación con OpenCV + CNN reconocedora | ✅ |
| 08 | Modelos predictivos | [Pronóstico de ventas a 28 días](08-series-temporales/) | Lags sin fuga, backtesting walk-forward, LSTM, intervalos | ✅ |
| 09 | LLMs + API | [Extractor de facturas](09-llm-extractor-facturas/) | Visión + salida estructurada, validación, autocorrección, FastAPI, Docker | ✅ |
| 10 | Proyecto integrador | [Cotizador y valorizaciones de obras (Perú) + API](10-cotizador-obras/) | APU, modelo de precios entrenado, LLM que interpreta, fórmula polinómica, valorizaciones con alertas, API para automatización | ✅ |

## Orden de estudio recomendado

1. **01 → 02**: entender el ciclo datos → modelo → pérdida → optimización → métricas.
2. **03**: el mismo ciclo, pero con capas y backpropagation.
3. **04 → 05**: qué hacer cuando faltan etiquetas.
4. **06**: qué pasa cuando los datos cambian con el tiempo.
5. **07**: un sistema completo (preprocesado + modelo + postprocesado).
6. **08 → 10**: lo que pide el mercado laboral (forecasting, LLMs, despliegue).

## Para cada proyecto, practica así

- Lee el `README.md` del proyecto y ejecuta el script.
- Cambia un hiperparámetro (learning rate, capas, % de etiquetas) y anota qué pasa.
- Responde los **retos** del final de cada README; ahí es donde de verdad se aprende.
