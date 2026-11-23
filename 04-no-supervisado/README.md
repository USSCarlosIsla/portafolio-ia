# Aprendizaje no supervisado: K-Means y PCA

**Objetivo:** encontrar estructura en datos **sin etiquetas**.

## Qué implementa
- **PCA desde cero** con SVD: proyección 2D y cuántas componentes explican el 95% de la varianza.
- **K-Means desde cero** con inicialización k-means++ y múltiples reinicios.
- Validación contra scikit-learn y evaluación con **ARI** (Adjusted Rand Index).
- Método del codo para elegir `k`.

## Retos
1. Usa **t-SNE** o **UMAP** en lugar de PCA para la visualización. ¿Se separan mejor los grupos?
2. Prueba **DBSCAN** y **clustering jerárquico**.
3. Usa PCA como preprocesado (reducir a 20 dims) antes de un clasificador: ¿cambia el accuracy? ¿y el tiempo?
4. Implementa un **autoencoder** en PyTorch y compáralo con PCA.
