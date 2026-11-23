"""Red neuronal multicapa (MLP) desde cero con NumPy: forward, backpropagation y SGD con mini-batches.

Se prueba en dos problemas:
  1. XOR: el ejemplo clásico que un modelo lineal NO puede resolver.
  2. Dígitos escritos a mano (8x8, 10 clases) incluidos en scikit-learn.
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from common.utils import save_fig


def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # estabilidad numérica
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    """MLP con capas ocultas ReLU y salida softmax, pérdida de entropía cruzada."""

    def __init__(self, sizes, lr=0.1, seed=0):
        rng = np.random.default_rng(seed)
        # Inicialización He: varianza 2/fan_in, adecuada para ReLU
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.lr = lr

    def forward(self, X):
        self.activations = [X]
        self.pre = []
        a = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            self.pre.append(z)
            a = softmax(z) if i == len(self.W) - 1 else relu(z)
            self.activations.append(a)
        return a

    def backward(self, Y):
        """Y en one-hot. Con softmax + entropía cruzada, dL/dz de la salida = (P - Y)/n."""
        n = Y.shape[0]
        delta = (self.activations[-1] - Y) / n
        for i in reversed(range(len(self.W))):
            grad_W = self.activations[i].T @ delta
            grad_b = delta.sum(axis=0)
            if i > 0:  # propagar el error a la capa anterior (regla de la cadena)
                delta = (delta @ self.W[i].T) * (self.pre[i - 1] > 0)
            self.W[i] -= self.lr * grad_W
            self.b[i] -= self.lr * grad_b

    def fit(self, X, y, epochs=100, batch_size=32, X_val=None, y_val=None):
        n_classes = self.W[-1].shape[1]
        Y = np.eye(n_classes)[y]
        rng = np.random.default_rng(0)
        history = {"loss": [], "val_acc": []}
        for _ in range(epochs):
            idx = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                batch = idx[start:start + batch_size]
                self.forward(X[batch])
                self.backward(Y[batch])
            P = self.forward(X)
            history["loss"].append(-np.mean(np.log(P[np.arange(len(y)), y] + 1e-12)))
            if X_val is not None:
                history["val_acc"].append(np.mean(self.predict(X_val) == y_val))
        return history

    def predict(self, X):
        return self.forward(X).argmax(axis=1)


def gradient_check():
    """Comprueba que backprop coincide con el gradiente numérico. Imprescindible al hacerlo a mano."""
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(5, 3)), rng.integers(0, 2, 5)
    Y = np.eye(2)[y]
    net = MLP([3, 4, 2], lr=0.0)

    def loss():
        P = net.forward(X)
        return -np.mean(np.log(P[np.arange(5), y]))

    net.forward(X)
    delta = (net.activations[-1] - Y) / 5
    analytic = net.activations[-2].T @ delta  # gradiente de la última capa
    numeric = np.zeros_like(net.W[-1])
    eps = 1e-5
    for i in np.ndindex(net.W[-1].shape):
        net.W[-1][i] += eps
        up = loss()
        net.W[-1][i] -= 2 * eps
        down = loss()
        net.W[-1][i] += eps
        numeric[i] = (up - down) / (2 * eps)
    diff = np.linalg.norm(analytic - numeric) / np.linalg.norm(analytic + numeric)
    print(f"Gradient check (diferencia relativa): {diff:.2e}  {'OK' if diff < 1e-6 else 'REVISAR'}")


def main():
    gradient_check()

    # --- XOR ---
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_xor = np.array([0, 1, 1, 0])
    net = MLP([2, 8, 2], lr=0.5, seed=3)
    net.fit(X_xor, y_xor, epochs=500, batch_size=4)
    print(f"XOR predicciones: {net.predict(X_xor)} (esperado {y_xor})")

    # --- Dígitos ---
    digits = load_digits()
    X = digits.data / 16.0
    X_tr, X_te, y_tr, y_te = train_test_split(X, digits.target, test_size=0.2, random_state=0)
    net = MLP([64, 128, 64, 10], lr=0.05)
    hist = net.fit(X_tr, y_tr, epochs=60, X_val=X_te, y_val=y_te)
    print(f"Dígitos — accuracy test: {hist['val_acc'][-1]:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["loss"])
    ax[0].set(title="Pérdida de entrenamiento", xlabel="época")
    ax[1].plot(hist["val_acc"])
    ax[1].set(title="Accuracy en test", xlabel="época")
    save_fig(fig, __file__, "mlp_digits.png")


if __name__ == "__main__":
    main()
