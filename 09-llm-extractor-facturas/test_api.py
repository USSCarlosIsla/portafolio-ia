"""Tests de la API (no gastan dinero: usan el motor OCR o simulan la falta de clave de Anthropic).

Ejecutar:  python -m pytest test_api.py -v
"""
import os
from pathlib import Path

os.environ["API_KEYS"] = "clave-de-prueba"
os.environ.pop("ANTHROPIC_API_KEY", None)

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402

api.load_dotenv = lambda *a, **k: None
client = TestClient(api.app)
FACTURA = next((Path(__file__).parent / "data" / "facturas").glob("*.png"))
H = {"X-API-Key": "clave-de-prueba"}


def subir(motor="ocr", headers=H, nombre=None, tipo="image/png"):
    with open(FACTURA, "rb") as f:
        return client.post(f"/v1/facturas/extraer?motor={motor}", headers=headers,
                           files={"archivo": (nombre or FACTURA.name, f, tipo)})


def test_salud():
    assert client.get("/salud").json()["estado"] == "ok"


def test_rechaza_sin_api_key():
    assert subir(headers={}).status_code == 401


def test_rechaza_formato_no_soportado():
    assert subir(nombre="x.pdf", tipo="application/pdf").status_code == 415


def test_extrae_con_ocr():
    r = subir("ocr")
    assert r.status_code == 200
    body = r.json()
    assert body["motor"] == "ocr"
    assert body["datos"]["total"] > 0


def test_llm_sin_clave_devuelve_503():
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-invalida"
    try:
        r = subir("llm")
    finally:
        os.environ.pop("ANTHROPIC_API_KEY", None)
    assert r.status_code in (502, 503)
