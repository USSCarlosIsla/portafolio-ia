"""Utilidades compartidas por todos los proyectos del portafolio."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # guarda gráficas a archivo sin abrir ventanas
import matplotlib.pyplot as plt  # noqa: E402


def output_dir(script_file: str) -> Path:
    """Devuelve (y crea) la carpeta outputs/ junto al script que la llama."""
    path = Path(script_file).resolve().parent / "outputs"
    path.mkdir(exist_ok=True)
    return path


def save_fig(fig, script_file: str, name: str) -> Path:
    path = output_dir(script_file) / name
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  gráfica guardada en {path}")
    return path
