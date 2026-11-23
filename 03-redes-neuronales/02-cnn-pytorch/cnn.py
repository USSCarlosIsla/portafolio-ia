"""Red convolucional (CNN) con PyTorch sobre dígitos 8x8, comparada con un MLP del mismo tamaño.

Muestra el ciclo estándar de PyTorch: Dataset/DataLoader → modelo nn.Module → loop de entrenamiento
→ evaluación → guardado de pesos.
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common.utils import output_dir, save_fig

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 4 * 4, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


def run_epoch(model, loader, loss_fn, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(xb)
    return total_loss / n, correct / n


def train(model, train_loader, test_loader, epochs=30):
    model.to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []
    for _ in range(epochs):
        run_epoch(model, train_loader, loss_fn, optimizer)
        scheduler.step()
        history.append(run_epoch(model, test_loader, loss_fn)[1])
    return history


def main():
    digits = load_digits()
    X = (digits.images / 16.0).astype(np.float32)[:, None]  # (N, 1, 8, 8)
    y = digits.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_te), torch.tensor(y_te)), batch_size=256)

    results = {}
    for name, model in [("CNN", SmallCNN()), ("MLP", MLP())]:
        n_params = sum(p.numel() for p in model.parameters())
        results[name] = train(model, train_loader, test_loader)
        print(f"{name}: {n_params:,} parámetros — accuracy test final {results[name][-1]:.4f}")
        if name == "CNN":
            torch.save(model.state_dict(), output_dir(__file__) / "cnn_digits.pt")

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, hist in results.items():
        ax.plot(hist, label=name)
    ax.set(title="Accuracy en test por época", xlabel="época", ylim=(0.8, 1.0))
    ax.legend()
    save_fig(fig, __file__, "cnn_vs_mlp.png")


if __name__ == "__main__":
    main()
