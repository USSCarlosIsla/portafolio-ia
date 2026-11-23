# CNN con PyTorch

**Objetivo:** dominar el ciclo de entrenamiento de PyTorch y entender por qué las convoluciones
funcionan mejor que capas densas con imágenes.

## Qué implementa
- `Dataset` / `DataLoader`, `nn.Module`, loop de train/eval separado (`model.train()` / `model.eval()`).
- Conv2d + BatchNorm + MaxPool + Dropout.
- Adam con weight decay y scheduler coseno.
- Comparación CNN vs MLP y guardado de pesos (`outputs/cnn_digits.pt`).

## Retos
1. Pasa a **MNIST** o **Fashion-MNIST** con `torchvision.datasets` (28x28).
2. Añade *data augmentation* (rotaciones pequeñas, desplazamientos).
3. Usa **transfer learning** con un ResNet18 preentrenado en un dataset de imágenes propio.
4. Visualiza los filtros de la primera capa y los mapas de activación.
