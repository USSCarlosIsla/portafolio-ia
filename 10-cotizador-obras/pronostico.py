"""Modelo entrenado: pronóstico del precio de cada insumo a 1-12 meses.

¿Para qué? Una obra cotizada hoy se ejecuta en 3, 6 o 9 meses. Si cotizas con precios de hoy y el
acero sube 8%, pierdes margen. Este modelo proyecta el precio al mes de ejecución con un intervalo.

  * Datos: historial mensual de precios 2017-2026. Es SINTÉTICO pero con dinámicas reales: inflación,
    shock 2021-2022 (acero muy volátil), estacionalidad de agregados en temporada de lluvias y aumentos
    de jornal cada junio (convenio de construcción civil). Sustitúyelo por datos reales (INEI, revista
    Costos o tus compras) con la misma estructura de columnas.
  * Candidatos: (a) un Gradient Boosting GLOBAL para todos los insumos y horizontes (h = 1..12) que
    predice el cambio porcentual (log-retorno); (b) "sin cambio"; (c) seguir la tendencia de 12 meses.
  * Selección por segmento (jornales vs materiales/equipos) con un periodo de calibración, y reporte
    final en un periodo de test que no se usó para decidir nada.
  * Intervalos: conformal prediction, usando cuantiles de los errores en un periodo de calibración.

Entrenar:  python pronostico.py      (genera data/historial_precios.csv y modelos/pronostico_precios.joblib)
"""
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import catalogo as C

BASE = Path(__file__).resolve().parent
HIST = BASE / "data" / "historial_precios.csv"
MODELO = BASE / "modelos" / "pronostico_precios.joblib"
H_MAX = 12
FEATURES = ["h", "ret_1", "ret_3", "ret_6", "ret_12", "mercado_3", "mercado_12", "mes_objetivo", "cruza_junio",
            "es_mo", "es_eq", "volatilidad_12"]


# ---------------- 1. Historial sintético ----------------
def generar_historial(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    meses = pd.date_range("2017-01-01", "2026-10-01", freq="MS")
    n = len(meses)
    t = np.arange(n)
    shock = np.exp(-0.5 * ((t - 58) / 7) ** 2)  # pico ~ noviembre 2021
    mercado = np.cumsum(rng.normal(0.003, 0.006, n)) + 0.12 * np.cumsum(shock) / shock.sum()
    filas = []
    for ins in C.INSUMOS.values():
        if ins.tipo == "MO":  # jornal: sube cada junio entre 3% y 6%, sin ruido
            log_p = np.cumsum([(rng.uniform(0.03, 0.06) if m.month == 6 else 0.0) for m in meses])
        else:
            beta = {"MAT-ACE": 1.8, "MAT-CEM": 1.1, "MAT-MAD": 1.3}.get(ins.codigo, 0.9)
            propio = np.cumsum(rng.normal(0, 0.012 if ins.codigo == "MAT-ACE" else 0.006, n))
            extra = (0.25 * shock) if ins.codigo == "MAT-ACE" else 0  # el acero se dispara más en el shock
            estacional = 0.03 * np.isin(meses.month, [1, 2, 3]) if ins.codigo in ("MAT-ARG", "MAT-ARF",
                                                                                    "MAT-PIE", "MAT-HOR") else 0
            log_p = beta * mercado + propio + extra + estacional
        precio = ins.precio * np.exp(log_p - log_p[-1])  # el último mes coincide con el catálogo
        filas += [{"fecha": m.date(), "insumo": ins.codigo, "tipo": ins.tipo, "precio": round(p, 3)}
                  for m, p in zip(meses, precio)]
    return pd.DataFrame(filas)


# ---------------- 2. Features ----------------
def construir_features(hist: pd.DataFrame, con_objetivo=True) -> pd.DataFrame:
    piv = hist.pivot(index="fecha", columns="insumo", values="precio").sort_index()
    logp = np.log(piv)
    mat_cols = [c for c in piv.columns if c.startswith("MAT")]
    mercado = logp[mat_cols].mean(axis=1)
    tipos = hist.drop_duplicates("insumo").set_index("insumo")["tipo"]
    fechas = pd.to_datetime(piv.index)
    filas = []
    for ins in piv.columns:
        lp = logp[ins].to_numpy()
        ret1 = np.diff(lp, prepend=np.nan)
        for i in range(12, len(lp)):
            base = {
                "insumo": ins, "fecha_origen": piv.index[i],
                "ret_1": lp[i] - lp[i - 1], "ret_3": lp[i] - lp[i - 3], "ret_6": lp[i] - lp[i - 6],
                "ret_12": lp[i] - lp[i - 12], "mercado_3": mercado.iloc[i] - mercado.iloc[i - 3],
                "mercado_12": mercado.iloc[i] - mercado.iloc[i - 12],
                "volatilidad_12": np.nanstd(ret1[i - 11:i + 1]),
                "es_mo": int(tipos[ins] == "MO"), "es_eq": int(tipos[ins] == "EQ"),
            }
            for h in range(1, H_MAX + 1):
                if con_objetivo and i + h >= len(lp):
                    break
                mes_obj = (fechas[i].month - 1 + h) % 12 + 1
                cruza = int(any((fechas[i].month - 1 + k) % 12 + 1 == 6 for k in range(1, h + 1)))
                fila = {**base, "h": h, "mes_objetivo": mes_obj, "cruza_junio": cruza}
                if con_objetivo:
                    fila["y"] = lp[i + h] - lp[i]
                filas.append(fila)
    return pd.DataFrame(filas)


def nuevo_modelo():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, min_samples_leaf=30, random_state=0)


# ---------------- 3. Entrenamiento con validación temporal ----------------
def entrenar():
    HIST.parent.mkdir(exist_ok=True)
    MODELO.parent.mkdir(exist_ok=True)
    hist = generar_historial()
    hist.to_csv(HIST, index=False)
    df = construir_features(hist)
    origen = pd.to_datetime(df["fecha_origen"])

    # fit < 2023-07 | calibración 2023-07..2024-06 | test 2024-07.. (nunca mezcla futuro con pasado)
    fit, cal, test = origen < "2023-07-01", (origen >= "2023-07-01") & (origen < "2024-07-01"), origen >= "2024-07-01"
    modelo = nuevo_modelo().fit(df.loc[fit, FEATURES], df.loc[fit, "y"])
    df["seg"] = np.where(df["es_mo"] == 1, "MO", "MAT_EQ")
    df["p_modelo"] = modelo.predict(df[FEATURES])

    def predicciones(d):
        """Los 3 métodos candidatos, en log-retorno."""
        return {"modelo": d["p_modelo"], "sin_cambio": 0 * d["y"], "tendencia": d["ret_12"] / 12 * d["h"]}

    def ape(y, pred):
        return float((np.abs(np.exp(y) - np.exp(pred)) / np.exp(y)).mean() * 100)

    # 1) Elegir el método de cada segmento con el periodo de CALIBRACIÓN (no con test: eso sería hacer trampa)
    elegido, q_lo, q_hi = {}, {}, {}
    for seg in ["MO", "MAT_EQ"]:
        dc = df[cal & (df["seg"] == seg)]
        errores = {k: ape(dc["y"], v) for k, v in predicciones(dc).items()}
        elegido[seg] = min(errores, key=errores.get)
        res = dc["y"] - predicciones(dc)[elegido[seg]]
        qs = pd.DataFrame({"h": dc["h"], "r": res}).groupby("h")["r"].quantile([0.1, 0.9]).unstack()
        q_lo[seg], q_hi[seg] = qs[0.1].to_dict(), qs[0.9].to_dict()
    print("Método elegido por segmento (según calibración):", elegido)

    # 2) Reportar en TEST, que nunca se usó para decidir nada
    print("\nError porcentual medio del precio a 1-12 meses (periodo de test):")
    filas, cubre = [], []
    for seg in ["MO", "MAT_EQ"]:
        dt = df[test & (df["seg"] == seg)]
        pr = predicciones(dt)
        fila = {"segmento": seg, **{k: round(ape(dt["y"], v), 2) for k, v in pr.items()}}
        fila["sistema"] = fila[elegido[seg]]
        filas.append(fila)
        centro = pr[elegido[seg]]
        cubre.append(((dt["y"] >= centro + dt["h"].map(q_lo[seg])) & (dt["y"] <= centro + dt["h"].map(q_hi[seg]))))
    tabla = pd.DataFrame(filas).set_index("segmento")
    print(tabla.to_string())
    cobertura = float(pd.concat(cubre).mean())
    print(f"Cobertura del intervalo 80% en test: {cobertura:.0%}")

    # Modelo final: reentrena con todo el historial (método y cuantiles de calibración se conservan)
    final = nuevo_modelo().fit(df[FEATURES], df["y"])
    joblib.dump({"modelo": final, "elegido": elegido, "q_lo": q_lo, "q_hi": q_hi,
                 "ultimo_mes": str(hist["fecha"].max()),
                 "metricas": {"error_test_pct": tabla.to_dict(orient="index"), "cobertura_80": cobertura}}, MODELO)
    print(f"\nModelo guardado en {MODELO}")


# ---------------- 4. Uso en producción ----------------
_cache = {}


def cargar():
    if "m" not in _cache:
        if not MODELO.exists():
            raise FileNotFoundError("Modelo no entrenado. Ejecuta: python pronostico.py")
        _cache["m"] = joblib.load(MODELO)
        _cache["hist"] = pd.read_csv(HIST, parse_dates=["fecha"])
        _cache["hist"]["fecha"] = _cache["hist"]["fecha"].dt.date
    return _cache["m"], _cache["hist"]


def meses_hasta(fecha_ejecucion: date, ultimo_mes: str) -> int:
    u = date.fromisoformat(ultimo_mes)
    return (fecha_ejecucion.year - u.year) * 12 + fecha_ejecucion.month - u.month


def proyectar_precios(fecha_ejecucion: date) -> dict:
    """Precio esperado (y rango 80%) de cada insumo en el mes de ejecución."""
    m, hist = cargar()
    h = meses_hasta(fecha_ejecucion, m["ultimo_mes"])
    if h <= 0:
        return {"h": 0, "precios": {}, "bajo": {}, "alto": {}, "insumos": []}
    if h > H_MAX:
        raise ValueError(f"Solo se proyecta hasta {H_MAX} meses (la fecha pedida está a {h}).")
    feats = construir_features(hist[hist["fecha"] >= date(2024, 1, 1)], con_objetivo=False)
    feats = feats[(feats["fecha_origen"] == feats["fecha_origen"].max()) & (feats["h"] == h)]
    pred = m["modelo"].predict(feats[FEATURES])
    actual = hist[hist["fecha"] == hist["fecha"].max()].set_index("insumo")["precio"]
    out = {"h": h, "precios": {}, "bajo": {}, "alto": {}, "insumos": []}
    for (_, f), p in zip(feats.iterrows(), pred):
        ins, hoy = f["insumo"], float(actual[f["insumo"]])
        seg = "MO" if f["es_mo"] == 1 else "MAT_EQ"
        metodo = m["elegido"][seg]
        p = {"modelo": p, "sin_cambio": 0.0, "tendencia": f["ret_12"] / 12 * h}[metodo]
        centro, lo, hi = (float(hoy * np.exp(p + d)) for d in (0.0, m["q_lo"][seg][h], m["q_hi"][seg][h]))
        out["precios"][ins], out["bajo"][ins], out["alto"][ins] = round(centro, 3), round(lo, 3), round(hi, 3)
        out["insumos"].append({"insumo": ins, "descripcion": C.INSUMOS[ins].descripcion, "precio_hoy": hoy,
                               "precio_proyectado": round(centro, 2), "rango_bajo": round(lo, 2),
                               "rango_alto": round(hi, 2), "variacion_pct": round((centro / hoy - 1) * 100, 2),
                               "metodo": metodo})
    return out


if __name__ == "__main__":
    entrenar()
