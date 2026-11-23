"""OCR de dígitos de punta a punta: OpenCV para detectar caracteres + CNN (PyTorch) para reconocerlos.

Pipeline (el mismo que usan los OCR clásicos):
  imagen → escala de grises → binarización (Otsu) → contornos → recorte de cada carácter
  → normalización a 28x28 → CNN → agrupación en palabras por distancia horizontal → texto.

Todo es offline: los datos de entrenamiento se GENERAN sintéticamente con las fuentes de OpenCV
y aumentos (rotación, grosor, ruido, desenfoque). Generar datos sintéticos es una técnica real
muy usada cuando no hay datos etiquetados suficientes.
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "common").is_dir())
sys.path.insert(0, str(ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from common.utils import output_dir, save_fig

FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
         cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX]
SIZE = 28
rng = np.random.default_rng(0)
torch.manual_seed(0)


# ---------------- Preprocesado compartido (entrenamiento e inferencia) ----------------
def normalize_char(binary_crop):
    """Centra el carácter en un cuadrado con margen y lo lleva a 28x28, valores en [0, 1]."""
    h, w = binary_crop.shape
    side = int(max(h, w) * 1.3) + 2
    canvas = np.zeros((side, side), np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = binary_crop
    return cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def add_noise(img):
    img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.1, 1.2))
    noise = rng.normal(0, rng.uniform(2, 18), img.shape)
    return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)


def binarize(gray):
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))  # quita puntitos de ruido


# ---------------- Generación de datos sintéticos ----------------
def render_digit(d):
    img = np.full((80, 80), 255, np.uint8)
    font = FONTS[rng.integers(len(FONTS))]
    scale, thick = rng.uniform(1.4, 2.4), int(rng.integers(2, 6))
    cv2.putText(img, str(d), (15, 62), font, scale, 0, thick, cv2.LINE_AA)
    M = cv2.getRotationMatrix2D((40, 40), rng.uniform(-10, 10), rng.uniform(0.9, 1.1))
    img = cv2.warpAffine(img, M, (80, 80), borderValue=255)
    bw = binarize(add_noise(img))
    ys, xs = np.nonzero(bw)
    if len(xs) == 0:
        return None
    return normalize_char(bw[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def make_dataset(n_per_class):
    X, y = [], []
    for d in range(10):
        while sum(1 for v in y if v == d) < n_per_class:
            img = render_digit(d)
            if img is not None:
                X.append(img)
                y.append(d)
    return torch.tensor(np.array(X))[:, None], torch.tensor(y)


# ---------------- Modelo ----------------
class OCRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7x7
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def train_model(X, y, epochs=8):
    model = OCRNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]
            loss = nn.functional.cross_entropy(model(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


# ---------------- Documento de prueba e inferencia ----------------
def render_document(text):
    """Dibuja una línea de números carácter a carácter (con fuente y tamaño aleatorios)."""
    font = FONTS[rng.integers(len(FONTS))]
    scale, thick = rng.uniform(1.6, 2.2), int(rng.integers(2, 5))
    img = np.full((110, 80 + 80 * len(text)), 255, np.uint8)  # lienzo amplio; se recorta al final
    x = 25
    for ch in text:
        if ch == " ":
            x += 55
            continue
        cv2.putText(img, ch, (x, 75), font, scale, 0, thick, cv2.LINE_AA)
        (w, _), _ = cv2.getTextSize(ch, font, scale, thick)
        x += w + int(rng.integers(8, 16))
    return add_noise(img[:, :x + 25])


def read_text(gray, model, return_boxes=False):
    bw = binarize(gray)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = sorted([b for b in boxes if b[3] > 15], key=lambda b: b[0])  # descarta ruido, ordena izq→der
    if not boxes:
        return ("", []) if return_boxes else ""
    crops = torch.tensor(np.array([normalize_char(bw[y:y + h, x:x + w]) for x, y, w, h in boxes]))[:, None]
    with torch.no_grad():
        digits = model(crops).argmax(1).tolist()
    # Espacios: medimos distancia entre CENTROS (no entre bordes, porque un "1" es estrecho y deja
    # huecos grandes). Una distancia muy superior a la típica = separación entre palabras.
    centers = np.array([x + w / 2 for x, y, w, h in boxes])
    steps = np.diff(centers)
    median_h = np.median([b[3] for b in boxes])
    typical = np.median(steps[steps < 1.2 * median_h]) if np.any(steps < 1.2 * median_h) else median_h
    text = str(digits[0])
    for step, d in zip(steps, digits[1:]):
        text += (" " if step > 1.5 * typical else "") + str(d)
    return (text, boxes) if return_boxes else text


def main():
    print("Generando dataset sintético...")
    X_tr, y_tr = make_dataset(600)
    X_te, y_te = make_dataset(100)
    model = train_model(X_tr, y_tr)
    with torch.no_grad():
        acc = (model(X_te).argmax(1) == y_te).float().mean().item()
    print(f"Accuracy por carácter (recortes aislados): {acc:.4f}")
    torch.save(model.state_dict(), output_dir(__file__) / "ocr_net.pt")

    # Evaluación de punta a punta sobre "documentos"
    n_docs, exact, char_ok, char_total, examples = 50, 0, 0, 0, []
    for _ in range(n_docs):
        truth = " ".join("".join(str(d) for d in rng.integers(0, 10, rng.integers(2, 6)))
                         for _ in range(rng.integers(1, 4)))
        img = render_document(truth)
        pred, boxes = read_text(img, model, return_boxes=True)
        exact += pred == truth
        t, p = truth.replace(" ", ""), pred.replace(" ", "")
        char_total += len(t)
        char_ok += sum(a == b for a, b in zip(t, p)) if len(t) == len(p) else 0
        if len(examples) < 4:
            examples.append((img, boxes, truth, pred))
    print(f"Documentos leídos exactamente: {exact}/{n_docs}")
    print(f"Accuracy por carácter de punta a punta: {char_ok / char_total:.4f}")

    fig, axes = plt.subplots(len(examples), 1, figsize=(10, 2.2 * len(examples)))
    for ax, (img, boxes, truth, pred) in zip(axes, examples):
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in boxes:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 1)
        ax.imshow(vis)
        ax.set_title(f"real: '{truth}'   →   OCR: '{pred}'", color="green" if truth == pred else "red")
        ax.axis("off")
    save_fig(fig, __file__, "ocr_resultados.png")


if __name__ == "__main__":
    main()
