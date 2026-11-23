"""Tests de valorizaciones (motor + API).

Ejecutar:  python -m pytest test_valorizaciones.py -v
Requiere: python pronostico.py, python generar_obra_grande.py y python simular_valorizaciones.py
"""
import os
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

os.environ["API_KEYS"] = "clave-test"

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
from cotizador import cotizar  # noqa: E402
from valorizaciones import formula_polinomica, valorizar_obra  # noqa: E402

DATA = Path(__file__).parent / "data"
client = TestClient(api.app)
H = {"X-API-Key": "clave-test"}
PRES = [{"item": "01", "codigo": "P01", "metrado": 100}, {"item": "02", "codigo": "P11", "metrado": 200}]


def indices_planos(valor_mes=None):
    """Índices constantes (K=1) salvo que se pida otro valor en un mes."""
    filas = []
    for cod in ["47", "39", "03", "21", "17", "04", "05", "43", "54", "49", "37"]:
        for m in pd.date_range("2026-01-01", "2027-12-01", freq="MS"):
            v = 100.0 if not (valor_mes and m.date() == valor_mes[0]) else valor_mes[1]
            filas.append({"fecha": m.date(), "codigo": cod, "indice": v, "proyectado": False})
    return pd.DataFrame(filas)


def test_formula_suma_uno_y_monomios_minimos():
    f = formula_polinomica(cotizar([(p["codigo"], p["metrado"]) for p in PRES]))
    assert sum(f["coeficientes"].values()) == pytest.approx(1.0, abs=1e-9)
    assert all(c >= 0.05 for c in f["coeficientes"].values()) and len(f["coeficientes"]) <= 8


def test_valorizacion_basica_sin_reajuste():
    cot = cotizar(PRES)
    res = valorizar_obra(cot, [{"mes": date(2026, 6, 1), "metrados": {"01": 50}}], date(2026, 5, 1),
                         indices=indices_planos())
    v = res["valorizaciones"][0]
    pu = cot["partidas"][0]["precio_unitario"]
    assert v["costo_directo"] == pytest.approx(50 * pu, abs=0.01)
    assert v["k"] == 1.0 and v["reajuste"] == 0
    assert v["neto_a_pagar"] == pytest.approx(v["valorizacion_bruta"] * 1.18, abs=0.02)


def test_reajuste_con_indices_10_por_ciento_mayores():
    cot = cotizar(PRES)
    res = valorizar_obra(cot, [{"mes": date(2026, 6, 1), "metrados": {"01": 50}}], date(2026, 5, 1),
                         indices=indices_planos((date(2026, 6, 1), 110.0)))
    v = res["valorizaciones"][0]
    assert v["k"] == pytest.approx(1.1, abs=0.001)
    assert v["reajuste"] == pytest.approx(v["valorizacion_bruta"] * 0.1, abs=0.05)


def test_mayor_metrado_se_alerta_y_no_se_paga():
    cot = cotizar(PRES)
    res = valorizar_obra(cot, [{"mes": date(2026, 6, 1), "metrados": {"01": 120}}], date(2026, 5, 1),
                         indices=indices_planos())
    v = res["valorizaciones"][0]
    assert any("mayor metrado" in a for a in v["alertas"])
    assert v["detalle"][0]["acumulado"] == 100


def test_adelanto_se_amortiza_completo_al_concluir():
    cot = cotizar(PRES)
    avance = [{"mes": date(2026, 6, 1), "metrados": {"01": 60, "02": 50}},
              {"mes": date(2026, 7, 1), "metrados": {"01": 40, "02": 150}}]
    res = valorizar_obra(cot, avance, date(2026, 5, 1), adelanto_directo_pct=0.1, indices=indices_planos())
    total_amort = sum(v["amortizacion_adelanto"] for v in res["valorizaciones"])
    assert total_amort == pytest.approx(res["adelanto_directo"], abs=0.01)
    assert res["adelanto_por_amortizar"] == 0


def test_alerta_atraso_bajo_80_por_ciento():
    cot = cotizar(PRES)
    res = valorizar_obra(cot, [{"mes": date(2026, 6, 1), "metrados": {"01": 10}}], date(2026, 5, 1),
                         programado={"2026-06": 1_000_000}, indices=indices_planos())
    assert any("Atraso" in a for a in res["valorizaciones"][0]["alertas"])


def test_item_inexistente_da_error():
    with pytest.raises(ValueError):
        valorizar_obra(cotizar(PRES), [{"mes": date(2026, 6, 1), "metrados": {"99": 1}}], date(2026, 5, 1),
                       indices=indices_planos())


# ---------------- API ----------------
def test_api_valorizacion_json():
    r = client.post("/v1/valorizaciones", headers=H, json={
        "presupuesto": PRES, "fecha_base": "2026-10-01", "adelanto_directo_pct": 0.1,
        "avance": [{"mes": "2026-11-01", "metrados": {"01": 30, "02": 40}}]})
    assert r.status_code == 200, r.text
    v = r.json()["valorizaciones"][0]
    assert v["neto_a_pagar"] > 0 and "detalle" not in v


def test_api_valorizacion_excel_obra_grande():
    with open(DATA / "obra_grande_metrados.xlsx", "rb") as m, open(DATA / "avance_obra_grande.xlsx", "rb") as a:
        r = client.post("/v1/valorizaciones/excel", headers=H, data={"fecha_base": "2026-10-01",
                                                                     "adelanto_directo_pct": "0.1"},
                        files={"metrados": ("m.xlsx", m, api.XLSX), "avance": ("a.xlsx", a, api.XLSX)})
    assert r.status_code == 200, r.text
    assert float(r.headers["X-Avance-Acumulado-Pct"]) == pytest.approx(100, abs=0.5)
    assert int(r.headers["X-Alertas"]) > 0


def test_api_formula_polinomica():
    r = client.post("/v1/formula-polinomica", headers=H, json=PRES)
    assert r.status_code == 200 and r.json()["texto"].startswith("K = ")
