"""Aprendizaje no supervisado: K-Means y PCA desde cero, validados contra scikit-learn.

Usamos los dígitos 8x8 SIN sus etiquetas para descubrir grupos, y luego miramos las etiquetas
reales solo para evaluar qué tan buenos son esos grupos.
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

from common.utils import save_fig


def pca_scratch(X, k):
    """PCA vía descomposición en valores singulares de los datos centrados."""
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    explained = S**2 / np.sum(S**2)
    return Xc @ Vt[:k].T, explained


def kmeans_scratch(X, k, n_init=10, max_iter=300, seed=0):
    """K-Means con inicialización k-means++ y varios reinicios; se queda con la menor inercia."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centers = [X[rng.integers(len(X))]]
        for _ in range(k - 1):  # k-means++: elige centros lejanos con prob. ∝ distancia²
            d2 = np.min(((X[:, None] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
            centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        centers = np.array(centers)
        for _ in range(max_iter):
            labels = ((X[:, None] - centers[None]) ** 2).sum(-1).argmin(1)
            new = np.array([X[labels == j].mean(0) if np.any(labels == j) else centers[j] for j in range(k)])
            if np.allclose(new, centers):
                break
            centers = new
        inertia = ((X - centers[labels]) ** 2).sum()
        if best is None or inertia < best[2]:
            best = (labels, centers, inertia)
    return best


def main():
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target

    Z, explained = pca_scratch(X, 2)
    sk_ratio = PCA(2).fit(X).explained_variance_ratio_
    print(f"Varianza explicada 2 componentes — desde cero: {explained[:2].round(4)} | sklearn: {sk_ratio.round(4)}")
    n95 = np.searchsorted(np.cumsum(explained), 0.95) + 1
    print(f"Componentes necesarias para 95% de varianza: {n95} de 64")

    labels, _, inertia = kmeans_scratch(X, 10)
    sk = KMeans(10, n_init=10, random_state=0).fit(X)
    print(f"K-Means inercia — desde cero: {inertia:.1f} | sklearn: {sk.inertia_:.1f}")
    print(f"ARI vs etiquetas reales — desde cero: {adjusted_rand_score(y, labels):.3f} | "
          f"sklearn: {adjusted_rand_score(y, sk.labels_):.3f}")

    # Método del codo para elegir k
    ks = range(2, 16)
    inertias = [KMeans(k, n_init=5, random_state=0).fit(X).inertia_ for k in ks]

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    ax[0].scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=8)
    ax[0].set_title("PCA 2D coloreado por etiqueta REAL")
    ax[1].scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=8)
    ax[1].set_title("PCA 2D coloreado por cluster K-Means")
    ax[2].plot(list(ks), inertias, "o-")
    ax[2].set(title="Método del codo", xlabel="k", ylabel="inercia")
    save_fig(fig, __file__, "no_supervisado.png")


if __name__ == "__main__":
    main()
