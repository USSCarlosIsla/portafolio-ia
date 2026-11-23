# Regresión logística

**Objetivo:** clasificación binaria. El modelo calcula `p = sigmoid(Xw + b)` y se entrena minimizando
la **entropía cruzada binaria**.

## Qué implementa
- Regresión logística desde cero con regularización L2.
- Comparación con scikit-learn.
- Experimento de **ingeniería de features**: con grado 1 la frontera es una recta (subajuste);
  con features polinómicas de grado 3 el mismo modelo lineal dibuja una curva.

## Conceptos clave
- Por qué el gradiente es `Xᵀ(p − y)/n` (deriva la sigmoide + log-loss a mano, ¡hazlo en papel!).
- Umbral de decisión 0.5 y cómo moverlo cambia precisión vs recall.
- Subajuste vs sobreajuste.

## Retos
1. Prueba grado 10 sin regularización y luego con `l2=0.1`. Explica la diferencia.
2. Calcula precisión, recall y F1 a mano para distintos umbrales.
3. Extiende a multiclase con **softmax** (pista: one-hot + gradiente `Xᵀ(P − Y)`).
