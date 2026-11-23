"""Aprendizaje semi-supervisado: entrenar con muy pocas etiquetas aprovechando datos sin etiquetar.

Simulamos el caso real: tenemos 1.400 imágenes de dígitos pero solo alguien etiquetó el 1-20%.
Comparamos:
  - Baseline supervisado: solo usa las pocas imágenes etiquetadas.
  - Self-training (pseudo-labeling) desde cero.
  - SelfTrainingClassifier de scikit-learn.
  - Label Spreading (propagación de etiquetas por grafo de similitud).
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from common.utils import save_fig

UNLABELED = -1  # convención de scikit-learn para "sin etiqueta"


def self_training_scratch(base, X, y_partial, threshold=0.8, max_iter=10):
    """Pseudo-labeling: entrena, etiqueta lo que el modelo predice con alta confianza, repite."""
    y = y_partial.copy()
    for it in range(max_iter):
        labeled = y != UNLABELED
        model = clone(base).fit(X[labeled], y[labeled])
        if labeled.all():
            break
        proba = model.predict_proba(X[~labeled])
        confident = proba.max(1) >= threshold
        if not confident.any():
            break
        idx = np.where(~labeled)[0][confident]
        y[idx] = model.classes_[proba[confident].argmax(1)]
    return model


def experiment(X_tr, y_tr, X_te, y_te, frac, seed=0):
    rng = np.random.default_rng(seed)
    # Garantizamos al menos 1 ejemplo por clase etiquetado
    labeled_idx = np.concatenate([rng.choice(np.where(y_tr == c)[0], 1) for c in range(10)])
    extra = rng.choice(np.setdiff1d(np.arange(len(y_tr)), labeled_idx),
                       max(0, int(frac * len(y_tr)) - 10), replace=False)
    labeled_idx = np.concatenate([labeled_idx, extra])
    y_partial = np.full_like(y_tr, UNLABELED)
    y_partial[labeled_idx] = y_tr[labeled_idx]

    # Regresión logística: da probabilidades bien calibradas, clave para decidir qué pseudo-etiquetar
    base = LogisticRegression(C=10, max_iter=3000)
    scores = {
        "Solo etiquetados": clone(base).fit(X_tr[labeled_idx], y_tr[labeled_idx]).score(X_te, y_te),
        "Self-training (desde cero)": self_training_scratch(base, X_tr, y_partial).score(X_te, y_te),
        "SelfTraining sklearn": SelfTrainingClassifier(clone(base), threshold=0.8)
        .fit(X_tr, y_partial).score(X_te, y_te),
        "Label Spreading": LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2, max_iter=100)
        .fit(X_tr, y_partial).score(X_te, y_te),
    }
    return len(labeled_idx), scores


def main():
    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    full = SVC(gamma="scale").fit(X_tr, y_tr).score(X_te, y_te)
    print(f"Referencia con el 100% etiquetado: {full:.4f}\n")

    fracs = [0.01, 0.03, 0.05, 0.10, 0.20]
    table = {}
    for frac in fracs:
        n, scores = experiment(X_tr, y_tr, X_te, y_te, frac)
        print(f"{frac:>4.0%} etiquetado ({n:>3} imágenes): " +
              " | ".join(f"{k}: {v:.3f}" for k, v in scores.items()))
        for k, v in scores.items():
            table.setdefault(k, []).append(v)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, vals in table.items():
        ax.plot([f * 100 for f in fracs], vals, "o-", label=name)
    ax.axhline(full, color="gray", ls="--", label="100% etiquetado")
    ax.set(xlabel="% de datos etiquetados", ylabel="accuracy test", title="Semi-supervisado vs supervisado")
    ax.legend()
    save_fig(fig, __file__, "semi_supervisado.png")


if __name__ == "__main__":
    main()
