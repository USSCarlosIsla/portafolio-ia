"""Regresión lineal: desde cero (ecuación normal y descenso de gradiente) vs scikit-learn.

Problema: predecir el precio de una casa a partir de sus metros cuadrados y número de habitaciones
(datos sintéticos con una relación lineal conocida + ruido).
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from common.utils import save_fig

rng = np.random.default_rng(42)


# ---------- 1. Datos ----------
def make_data(n=500):
    m2 = rng.uniform(40, 200, n)
    rooms = rng.integers(1, 6, n)
    # precio real (miles): 1.5 por m2 + 10 por habitación + 30 base + ruido
    price = 1.5 * m2 + 10 * rooms + 30 + rng.normal(0, 15, n)
    X = np.column_stack([m2, rooms]).astype(float)
    return X, price


# ---------- 2. Modelo desde cero ----------
class LinearRegressionScratch:
    """y = X @ w + b, entrenado con descenso de gradiente sobre el MSE."""

    def __init__(self, lr=0.1, epochs=2000):
        self.lr, self.epochs = lr, epochs
        self.loss_history = []

    def fit(self, X, y):
        # Estandarizar es clave para que el gradiente converja rápido
        self.mu, self.sigma = X.mean(0), X.std(0)
        Xs = (X - self.mu) / self.sigma
        n, d = Xs.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.epochs):
            y_hat = Xs @ self.w + self.b
            error = y_hat - y
            self.loss_history.append(np.mean(error**2))
            # derivadas del MSE respecto a w y b
            self.w -= self.lr * (2 / n) * Xs.T @ error
            self.b -= self.lr * (2 / n) * error.sum()
        return self

    def predict(self, X):
        return ((X - self.mu) / self.sigma) @ self.w + self.b

    def coef_original_scale(self):
        """Convierte los pesos al espacio original para poder interpretarlos."""
        w = self.w / self.sigma
        b = self.b - np.sum(self.w * self.mu / self.sigma)
        return w, b


def normal_equation(X, y):
    """Solución cerrada: theta = (XᵀX)⁻¹ Xᵀy (usamos lstsq por estabilidad numérica)."""
    Xb = np.column_stack([np.ones(len(X)), X])
    theta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    return theta[1:], theta[0]


def r2(y, y_hat):
    return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)


# ---------- 3. Experimento ----------
def main():
    X, y = make_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

    gd = LinearRegressionScratch().fit(X_tr, y_tr)
    w_gd, b_gd = gd.coef_original_scale()
    w_ne, b_ne = normal_equation(X_tr, y_tr)
    sk = LinearRegression().fit(X_tr, y_tr)

    print("Valores reales:            w=[1.50, 10.00]  b=30.00")
    print(f"Descenso de gradiente:     w={np.round(w_gd, 2)}  b={b_gd:.2f}")
    print(f"Ecuación normal:           w={np.round(w_ne, 2)}  b={b_ne:.2f}")
    print(f"scikit-learn:              w={np.round(sk.coef_, 2)}  b={sk.intercept_:.2f}")
    print(f"R² test (desde cero): {r2(y_te, gd.predict(X_te)):.4f}")
    print(f"R² test (sklearn):    {sk.score(X_te, y_te):.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(gd.loss_history)
    ax[0].set(yscale="log", title="Curva de pérdida (MSE)", xlabel="época", ylabel="MSE")
    y_pred = gd.predict(X_te)
    ax[1].scatter(y_te, y_pred, alpha=0.6)
    lims = [y_te.min(), y_te.max()]
    ax[1].plot(lims, lims, "r--", label="predicción perfecta")
    ax[1].set(title="Real vs predicho (test)", xlabel="precio real", ylabel="precio predicho")
    ax[1].legend()
    save_fig(fig, __file__, "regresion_lineal.png")


if __name__ == "__main__":
    main()
