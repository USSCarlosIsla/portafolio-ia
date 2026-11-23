"""Tests del motor y de la API (sin gastar dinero: no llaman al LLM).

Ejecutar:  python -m pytest test_cotizador.py -v
Requiere haber corrido antes:  python pronostico.py  y  python generar_obra_grande.py
"""
import io
import os
from datetime import date
from pathlib import Path

import openpyxl
import pytest

os.environ["API_KEYS"] = "clave-test"

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
from catalogo import Catalogo  # noqa: E402
from cotizador import analizar_partida, cotizar  # noqa: E402

DATA = Path(__file__).parent / "data"
client = TestClient(api.app)
H = {"X-API-Key": "clave-test"}


# ---------------- Motor ----------------
def test_precio_unitario_es_suma_de_componentes():
    a = analizar_partida("P05")
    assert a["precio_unitario"] == pytest.approx(a["mano_obra"] + a["materiales"] + a["equipos"], abs=0.02)


def test_mano_de_obra_usa_rendimiento():
    # P01: 1 peón, rendimiento 4 m3/día → 8/4 = 2 hh por m3
    linea = next(d for d in analizar_partida("P01")["detalle"] if d["insumo"] == "MO-PEO")
    assert linea["cantidad"] == pytest.approx(2.0)


def test_totales_cuadran():
    c = cotizar([("P01", 10), ("P11", 50)])
    assert c["subtotal"] == pytest.approx(c["costo_directo"] + c["gastos_generales"] + c["utilidad"], abs=0.01)
    assert c["total"] == pytest.approx(c["subtotal"] * 1.18, abs=0.02)


def test_subtotal_de_titulos():
    c = cotizar([{"item": "01", "descripcion": "Obras"}, {"item": "01.01", "codigo": "P01", "metrado": 10},
                 {"item": "01.02", "codigo": "P11", "metrado": 5}, {"item": "02", "descripcion": "Otro"},
                 {"item": "02.01", "codigo": "P15", "metrado": 3}])
    titulo = c["partidas"][0]
    assert titulo["parcial"] == pytest.approx(c["partidas"][1]["parcial"] + c["partidas"][2]["parcial"])


def test_rechaza_partida_desconocida_y_metrado_invalido():
    with pytest.raises(ValueError):
        cotizar([("P99", 10)])
    with pytest.raises(ValueError):
        cotizar([("P01", -5)])


def test_catalogo_excel_ida_y_vuelta(tmp_path):
    Catalogo.ejemplo().a_excel(tmp_path / "cat.xlsx")
    cat = Catalogo.desde_excel(tmp_path / "cat.xlsx")
    assert cotizar([("P07", 12)], cat=cat)["total"] == cotizar([("P07", 12)])["total"]


def test_proyeccion_futura_cambia_el_total():
    c = api.cotizar_con_proyeccion([("P10", 1000), ("P11", 100)], date(2027, 6, 1))
    p = c["proyeccion"]
    assert p["rango_80"][0] <= p["total_proyectado"] <= p["rango_80"][1]
    assert c["total"] == p["total_proyectado"] != p["total_precios_hoy"]


# ---------------- API ----------------
def test_api_requiere_clave():
    assert client.post("/v1/cotizaciones", json={"items": []}).status_code == 401


def test_api_cotizacion_json():
    r = client.post("/v1/cotizaciones", headers=H, json={
        "items": [{"codigo": "P05", "metrado": 4.2}, {"codigo": "P10", "metrado": 520}],
        "fecha_ejecucion": "2027-03-01", "incluir_detalle": True})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] > 0 and body["proyeccion"]["meses_adelante"] == 5 and body["insumos"]


def test_api_error_422_con_partida_inexistente():
    r = client.post("/v1/cotizaciones", headers=H, json={"items": [{"codigo": "XX", "metrado": 1}]})
    assert r.status_code == 422


def test_api_excel_obra_grande_con_catalogo_propio():
    with open(DATA / "obra_grande_metrados.xlsx", "rb") as m, open(DATA / "plantilla_catalogo.xlsx", "rb") as c:
        r = client.post("/v1/cotizaciones/excel", headers=H,
                        files={"metrados": ("m.xlsx", m, api.XLSX), "catalogo": ("c.xlsx", c, api.XLSX)})
    assert r.status_code == 200
    assert float(r.headers["X-Total"]) > 1_000_000
    libro = openpyxl.load_workbook(io.BytesIO(r.content))
    assert {"Presupuesto", "APU", "Insumos"} <= set(libro.sheetnames)


def test_api_excel_rechaza_archivo_no_excel():
    r = client.post("/v1/cotizaciones/excel", headers=H, files={"metrados": ("m.xlsx", b"hola", api.XLSX)})
    assert r.status_code == 415
