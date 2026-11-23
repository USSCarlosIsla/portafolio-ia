# Regresión lineal

**Objetivo:** predecir un valor continuo (precio) como combinación lineal de variables: `ŷ = Xw + b`.

## Qué implementa
- **Descenso de gradiente desde cero** minimizando el MSE: `w ← w − lr · (2/n) Xᵀ(ŷ − y)`.
- **Ecuación normal** (solución cerrada) con `np.linalg.lstsq`.
- **scikit-learn** como referencia. Los tres deben recuperar los coeficientes reales (1.5, 10, 30).

## Conceptos clave
- Por qué **estandarizar** las variables acelera la convergencia.
- MSE como función de pérdida y R² como métrica interpretable.
- División train/test para medir generalización.

## Retos
1. Sube `lr` a 1.5: ¿qué pasa con la curva de pérdida? ¿Por qué?
2. Quita la estandarización: ¿cuántas épocas necesitas ahora?
3. Añade regularización L2 (Ridge) al gradiente: `+ 2λw`.
4. Agrega una variable inútil (ruido puro) y observa su peso.
