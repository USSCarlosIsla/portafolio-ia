# Aprendizaje semi-supervisado

**Objetivo:** etiquetar datos es caro. ¿Cuánto rendimiento recuperamos usando los datos **sin etiquetar**?

## Qué implementa
- Escenario con 1%, 3%, 5%, 10% y 20% de etiquetas.
- **Self-training / pseudo-labeling desde cero**: el modelo se auto-etiqueta lo que predice con ≥80% de confianza.
- `SelfTrainingClassifier` de scikit-learn.
- **Label Spreading**: propaga etiquetas por un grafo de vecinos cercanos (asume que puntos cercanos comparten clase).
- Gráfica de accuracy vs % etiquetado comparada con el techo de 100% etiquetado.

## Conceptos clave
- Supuestos del semi-supervisado: *cluster assumption* y *manifold assumption*.
- Riesgo del pseudo-labeling: **confirmation bias** (el modelo refuerza sus propios errores).
- Relación con **active learning**: elegir qué ejemplos vale la pena que etiquete un humano.

## Retos
1. Mueve el `threshold` a 0.5 y 0.99. ¿Qué ocurre con pocas etiquetas?
2. Implementa **active learning** por incertidumbre: pide etiqueta a los 10 ejemplos con menor confianza por ronda.
3. Implementa **FixMatch** simplificado en PyTorch (consistencia entre aumentos débiles y fuertes).
