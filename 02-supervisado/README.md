# Aprendizaje supervisado: comparativa de clasificadores

**Objetivo:** el flujo profesional de un problema de clasificación con datos reales (diagnóstico de cáncer).

## Qué implementa
- 6 modelos: logística, KNN, SVM, árbol, Random Forest, Gradient Boosting.
- **Pipelines** (`StandardScaler` + modelo) para evitar *data leakage*.
- **Validación cruzada estratificada** de 5 folds y tabla comparativa (`outputs/resultados_cv.csv`).
- **GridSearchCV** para optimizar hiperparámetros.
- Evaluación final una sola vez sobre test + matriz de confusión + importancia de features.

## Conceptos clave
- Por qué el test set se toca **una sola vez**.
- Accuracy vs F1 vs recall: en medicina un falso negativo (maligno → benigno) es lo peor.
- Modelos lineales vs basados en árboles (los árboles no necesitan escalado).

## Retos
1. Ajusta el umbral para que el recall de "malignant" sea ≥ 0.99. ¿Cuánto pierdes en precisión?
2. Añade `XGBoost` o `LightGBM` y compáralos.
3. Usa SHAP para explicar una predicción individual.
