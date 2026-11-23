"""Regresión logística desde cero (clasificación binaria) vs scikit-learn.

Problema: dos clases que no se separan con una recta (make_moons). Comparamos el modelo lineal
con uno que usa features polinómicas para ver cómo cambia la frontera de decisión.
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from common.utils import save_fig


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegressionScratch:
    def __init__(self, lr=0.5, epochs=3000, l2=0.0):
        self.lr, self.epochs, self.l2 = lr, epochs, l2
        self.loss_history = []

    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        eps = 1e-12
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)
            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
            self.loss_history.append(loss + self.l2 * np.sum(self.w**2))
            # El gradiente de la entropía cruzada con sigmoide es sorprendentemente simple: (p - y)
            grad_w = X.T @ (p - y) / n + 2 * self.l2 * self.w
            grad_b = np.mean(p - y)
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def make_features(degree):
    poly = PolynomialFeatures(degree, include_bias=False)
    scaler = StandardScaler()

    def fit(X):
        return scaler.fit_transform(poly.fit_transform(X))

    def transform(X):
        return scaler.transform(poly.transform(X))

    return fit, transform


def main():
    X, y = make_moons(n_samples=600, noise=0.2, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    xx, yy = np.meshgrid(np.linspace(-1.6, 2.6, 300), np.linspace(-1.2, 1.7, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    for ax, degree in zip(axes[:2], [1, 3]):
        fit_t, transform = make_features(degree)
        F_tr = fit_t(X_tr)
        model = LogisticRegressionScratch(l2=1e-3).fit(F_tr, y_tr)
        acc = np.mean(model.predict(transform(X_te)) == y_te)
        sk_acc = LogisticRegression(max_iter=2000).fit(F_tr, y_tr).score(transform(X_te), y_te)
        print(f"grado {degree}: accuracy desde cero={acc:.3f} | sklearn={sk_acc:.3f}")

        zz = model.predict_proba(transform(grid)).reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.6)
        ax.contour(xx, yy, zz, levels=[0.5], colors="k")
        ax.scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap="RdBu", edgecolor="k", s=20)
        ax.set_title(f"Features polinómicas grado {degree} — acc {acc:.2f}")
        axes[2].plot(model.loss_history, label=f"grado {degree}")

    axes[2].set(title="Entropía cruzada", xlabel="época")
    axes[2].legend()
    save_fig(fig, __file__, "regresion_logistica.png")


if __name__ == "__main__":
    main()
