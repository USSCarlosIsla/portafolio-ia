"""Aprendizaje continuo: modelos que siguen aprendiendo cuando llegan datos nuevos.

Parte 1 — Olvido catastrófico (class-incremental):
  Tarea A = dígitos 0-4, Tarea B = dígitos 5-9. Entrenamos secuencialmente A → B y comparamos:
    * Fine-tuning ingenuo (olvida A casi por completo).
    * Experience Replay (guarda unos pocos ejemplos de A en un buffer de memoria).
    * EWC — Elastic Weight Consolidation (penaliza mover los pesos importantes para A).

Parte 2 — Aprendizaje online con concept drift:
  Un flujo de datos cuya regla cambia a mitad de camino. Un modelo estático vs uno que se
  actualiza con partial_fit en cada lote.
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from torch import nn

from common.utils import save_fig

torch.manual_seed(0)


# ======================= Parte 1: olvido catastrófico =======================
def make_model():
    return nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))


def accuracy(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(1) == y).float().mean().item()


def fisher_diagonal(model, X, y):
    """Importancia de cada peso para la tarea A ≈ E[(∂ log p / ∂θ)²]."""
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    for i in range(len(X)):
        model.zero_grad()
        nn.functional.cross_entropy(model(X[i:i + 1]), y[i:i + 1]).backward()
        for n, p in model.named_parameters():
            fisher[n] += p.grad.detach() ** 2 / len(X)
    return fisher


def train_task(model, X, y, epochs=40, replay=None, ewc=None, lam=5000.0):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 32):
            idx = perm[i:i + 32]
            xb, yb = X[idx], y[idx]
            if replay is not None:  # mezclar ejemplos viejos con los nuevos
                r = torch.randint(0, len(replay[0]), (16,))
                xb, yb = torch.cat([xb, replay[0][r]]), torch.cat([yb, replay[1][r]])
            loss = nn.functional.cross_entropy(model(xb), yb)
            if ewc is not None:
                fisher, old = ewc
                loss = loss + lam / 2 * sum(
                    (fisher[n] * (p - old[n]) ** 2).sum() for n, p in model.named_parameters())
            opt.zero_grad()
            loss.backward()
            opt.step()


def forgetting_experiment():
    digits = load_digits()
    X = torch.tensor(digits.data / 16.0, dtype=torch.float32)
    y = torch.tensor(digits.target)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    a_tr, b_tr = ytr < 5, ytr >= 5
    a_te, b_te = yte < 5, yte >= 5

    results = {}
    for strategy in ["Fine-tuning ingenuo", "Experience Replay", "EWC"]:
        torch.manual_seed(0)
        model = make_model()
        train_task(model, Xtr[a_tr], ytr[a_tr])
        acc_a_before = accuracy(model, Xte[a_te], yte[a_te])

        replay = ewc = None
        if strategy == "Experience Replay":
            keep = torch.randperm(int(a_tr.sum()))[:50]  # solo 50 ejemplos de A en memoria
            replay = (Xtr[a_tr][keep], ytr[a_tr][keep])
        elif strategy == "EWC":
            fisher = fisher_diagonal(model, Xtr[a_tr][:200], ytr[a_tr][:200])
            old = {n: p.detach().clone() for n, p in model.named_parameters()}
            ewc = (fisher, old)

        train_task(model, Xtr[b_tr], ytr[b_tr], replay=replay, ewc=ewc)
        results[strategy] = {
            "A antes": acc_a_before,
            "A después": accuracy(model, Xte[a_te], yte[a_te]),
            "B después": accuracy(model, Xte[b_te], yte[b_te]),
            "Total": accuracy(model, Xte, yte),
        }
        r = results[strategy]
        print(f"{strategy:<22} A antes={r['A antes']:.3f}  A después={r['A después']:.3f}  "
              f"B={r['B después']:.3f}  total={r['Total']:.3f}")
    return results


# ======================= Parte 2: aprendizaje online =======================
def drift_stream(n_batches=60, batch=100, drift_at=30, seed=0):
    """Clasificación 2D; a partir de drift_at la frontera de decisión rota 90°."""
    rng = np.random.default_rng(seed)
    for t in range(n_batches):
        X = rng.normal(size=(batch, 2))
        y = (X[:, 0] > 0).astype(int) if t < drift_at else (X[:, 1] > 0).astype(int)
        yield t, X, y


def online_experiment():
    static = SGDClassifier(loss="log_loss", random_state=0)
    online = SGDClassifier(loss="log_loss", random_state=0, learning_rate="constant", eta0=0.05)
    acc_static, acc_online = [], []
    for t, X, y in drift_stream():
        if t == 0:
            static.fit(X, y)
            online.partial_fit(X, y, classes=[0, 1])
            continue
        # Evaluación "prequential": primero predecimos el lote nuevo, luego aprendemos de él
        acc_static.append(static.score(X, y))
        acc_online.append(online.score(X, y))
        online.partial_fit(X, y)
    print(f"Online — accuracy media después del drift: estático={np.mean(acc_static[30:]):.3f} | "
          f"online={np.mean(acc_online[30:]):.3f}")
    return acc_static, acc_online


def main():
    print("=== Parte 1: olvido catastrófico ===")
    results = forgetting_experiment()
    print("\n=== Parte 2: aprendizaje online con concept drift ===")
    acc_static, acc_online = online_experiment()

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    names = list(results)
    x = np.arange(len(names))
    for i, key in enumerate(["A antes", "A después", "B después"]):
        ax[0].bar(x + (i - 1) * 0.25, [results[n][key] for n in names], 0.25, label=key)
    ax[0].set_xticks(x, names)
    ax[0].set(title="Olvido catastrófico (A=0-4, B=5-9)", ylabel="accuracy")
    ax[0].legend()
    ax[1].plot(acc_static, label="modelo estático")
    ax[1].plot(acc_online, label="modelo online (partial_fit)")
    ax[1].axvline(29, color="red", ls="--", label="concept drift")
    ax[1].set(title="Flujo de datos con drift", xlabel="lote", ylabel="accuracy")
    ax[1].legend()
    save_fig(fig, __file__, "aprendizaje_continuo.png")


if __name__ == "__main__":
    main()
