"""Aprendizaje supervisado: comparativa de clasificadores con pipelines y validación cruzada.

Dataset: cáncer de mama de Wisconsin (incluido en scikit-learn). 569 pacientes, 30 features,
clase maligno/benigno. Un caso realista donde los falsos negativos importan más.
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from common.utils import output_dir, save_fig

MODELS = {
    "Regresión logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC()),
    "Árbol de decisión": DecisionTreeClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(random_state=0),
    "Gradient Boosting": GradientBoostingClassifier(random_state=0),
}


def main():
    data = load_breast_cancer()
    X, y = data.data, data.target  # 0 = maligno, 1 = benigno
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    # 1) Validación cruzada estratificada sobre el set de entrenamiento
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    rows = []
    for name, model in MODELS.items():
        scores = cross_validate(model, X_tr, y_tr, cv=cv, scoring=["accuracy", "f1", "recall"])
        rows.append({
            "modelo": name,
            "accuracy": scores["test_accuracy"].mean(),
            "f1": scores["test_f1"].mean(),
            "recall": scores["test_recall"].mean(),
            "± acc": scores["test_accuracy"].std(),
        })
    results = pd.DataFrame(rows).sort_values("f1", ascending=False)
    print(results.round(4).to_string(index=False))
    results.to_csv(output_dir(__file__) / "resultados_cv.csv", index=False)

    # 2) Búsqueda de hiperparámetros para el SVM
    grid = GridSearchCV(
        MODELS["SVM (RBF)"],
        {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.001]},
        cv=cv, scoring="f1",
    ).fit(X_tr, y_tr)
    print(f"\nMejores hiperparámetros SVM: {grid.best_params_} (F1 CV={grid.best_score_:.4f})")

    # 3) Evaluación final UNA sola vez en test
    y_pred = grid.predict(X_te)
    print("\nReporte en test:")
    print(classification_report(y_te, y_pred, target_names=data.target_names))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_te, y_pred, display_labels=data.target_names, ax=ax[0])
    ax[0].set_title("Matriz de confusión — SVM optimizado")
    rf = MODELS["Random Forest"].fit(X_tr, y_tr)
    top = np.argsort(rf.feature_importances_)[-10:]
    ax[1].barh(np.array(data.feature_names)[top], rf.feature_importances_[top])
    ax[1].set_title("Top 10 features (Random Forest)")
    save_fig(fig, __file__, "clasificadores.png")


if __name__ == "__main__":
    main()
