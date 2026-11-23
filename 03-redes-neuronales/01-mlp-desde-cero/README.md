# MLP desde cero (NumPy)

**Objetivo:** entender exactamente qué hace una red neuronal implementando **backpropagation a mano**.

## Qué implementa
- Capas densas con ReLU, salida softmax, entropía cruzada.
- Inicialización He, SGD con mini-batches.
- **Gradient check** numérico: la forma profesional de verificar que tu backprop es correcto.
- XOR (no lineal) y clasificación de dígitos 8x8 (~97% accuracy).

## Conceptos clave
- Regla de la cadena: `delta_{l-1} = (delta_l · W_lᵀ) ⊙ ReLU'(z_{l-1})`.
- Por qué softmax + entropía cruzada da un gradiente tan simple: `P − Y`.
- Efecto de la inicialización y del learning rate.

## Retos
1. Implementa **momentum** y luego **Adam**. Compara las curvas de pérdida.
2. Añade **dropout** en las capas ocultas.
3. Cambia ReLU por sigmoide en una red de 6 capas y observa el *vanishing gradient*.
