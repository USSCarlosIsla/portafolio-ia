"""Valorizaciones mensuales de obra (Perú, contratos a precios unitarios).

Para cada mes calcula:
  metrados (anterior / actual / acumulado / saldo / % avance) por partida
  → valorización bruta = costo directo del mes + gastos generales + utilidad
  → reajuste por fórmula polinómica: V × (K − 1), con K = Σ coef_i × (Ir_i / Io_i)
  → amortización del adelanto directo (proporcional a lo valorizado)
  → IGV → neto a pagar
y genera alertas: mayores metrados (requieren aprobación) y atraso frente al calendario programado.

La fórmula polinómica se arma AUTOMÁTICAMENTE desde el presupuesto: incidencia de cada Índice
Unificado de Precios de la Construcción (IUPC, INEI) en el costo, con monomios mínimos de 5%.

⚠ Antes de usarlo en una obra real, verifica contra el reglamento vigente (Ley de Contrataciones del
Estado y su reglamento, DS 011-79-VC para fórmulas polinómicas):
  * NO se implementa la deducción del reajuste que no corresponde por adelantos (pendiente de norma).
  * El índice usado es el del mes valorizado (criterio "último mes de cada valorización").
  * Los índices de data/indices_iupc.csv son SINTÉTICOS: reemplázalos por los publicados por el INEI.
"""
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import catalogo as C
from cotizador import r2

BASE = Path(__file__).resolve().parent
INDICES_CSV = BASE / "data" / "indices_iupc.csv"

# Índices Unificados (INEI). Verifica los códigos con la tabla oficial vigente antes de usarlos.
NOMBRES_INDICE = {
    "47": "Mano de obra (incluye leyes sociales)", "21": "Cemento Portland tipo I",
    "03": "Acero de construcción corrugado", "04": "Agregado fino", "05": "Agregado grueso",
    "17": "Ladrillo", "43": "Madera nacional para encofrado y carpintería", "54": "Pintura látex",
    "49": "Maquinaria y equipo importado", "37": "Herramienta manual",
    "39": "Índice general de precios al consumidor",
}
# insumo del catálogo → índice unificado (para catálogos propios: los no mapeados van al 39)
MAPEO_INDICES = {
    "MO-OPE": "47", "MO-OFI": "47", "MO-PEO": "47", "MAT-CEM": "21", "MAT-ACE": "03", "MAT-ALA": "03",
    "MAT-CLA": "03", "MAT-ARG": "04", "MAT-ARF": "04", "MAT-PIE": "05", "MAT-PGR": "05", "MAT-HOR": "05",
    "MAT-AGU": "39", "MAT-LKK": "17", "MAT-MAD": "43", "MAT-PIN": "54", "MAT-IMP": "54",
    "EQ-MEZ": "49", "EQ-VIB": "49",
}
# serie de precios del historial que "imita" cada índice (solo para generar índices sintéticos)
SERIE_DE_INDICE = {"47": "MO-OPE", "21": "MAT-CEM", "03": "MAT-ACE", "04": "MAT-ARG", "05": "MAT-PIE",
                   "17": "MAT-LKK", "43": "MAT-MAD", "54": "MAT-PIN", "49": "EQ-MEZ", "37": "EQ-VIB"}


# ======================= Índices =======================
def generar_indices_sinteticos() -> pd.DataFrame:
    """Índices tipo IUPC a partir del historial de precios (sintético) + proyección a 12 meses."""
    from pronostico import HIST, cargar, proyectar_precios

    cargar()
    hist = pd.read_csv(HIST, parse_dates=["fecha"])
    piv = hist.pivot(index="fecha", columns="insumo", values="precio")
    ultimo = piv.index.max()
    futuros = []
    for h in range(1, 13):  # meses futuros con el modelo de pronóstico (útil para proyectar reajustes)
        mes = (ultimo + pd.DateOffset(months=h)).date()
        futuros.append(pd.Series(proyectar_precios(mes)["precios"], name=pd.Timestamp(mes)))
    piv = pd.concat([piv, pd.DataFrame(futuros)])
    filas = []
    for cod, ins in SERIE_DE_INDICE.items():
        serie = piv[ins] / piv[ins].iloc[0] * 300  # base arbitraria, como los IUPC (cientos)
        filas += [{"fecha": f.date(), "codigo": cod, "indice": round(v, 2), "proyectado": f > ultimo}
                  for f, v in serie.items()]
    general = piv.drop(columns=[c for c in piv.columns if c.startswith("MO")]).mean(axis=1)
    filas += [{"fecha": f.date(), "codigo": "39", "indice": round(v / general.iloc[0] * 300, 2),
               "proyectado": f > ultimo} for f, v in general.items()]
    df = pd.DataFrame(filas)
    INDICES_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(INDICES_CSV, index=False)
    return df


def cargar_indices() -> pd.DataFrame:
    if not INDICES_CSV.exists():
        return generar_indices_sinteticos()
    df = pd.read_csv(INDICES_CSV, dtype={"codigo": str}, parse_dates=["fecha"])
    df["fecha"] = df["fecha"].dt.date
    return df


def indice(indices: pd.DataFrame, codigo: str, mes: date) -> float:
    fila = indices[(indices["codigo"] == codigo) & (indices["fecha"] == date(mes.year, mes.month, 1))]
    if fila.empty:
        raise ValueError(f"No hay índice {codigo} para {mes:%Y-%m}. Carga los índices INEI de ese mes.")
    return float(fila["indice"].iloc[0])


# ======================= Fórmula polinómica =======================
def formula_polinomica(cot: dict, mapeo: Dict[str, str] = None, minimo: float = 0.05,
                       max_monomios: int = 8) -> dict:
    """Coeficientes por incidencia de cada índice en el presupuesto. GG y utilidad → índice 39.
    Monomios < `minimo` se agrupan en el 39; se redondea a 3 decimales y la suma queda en 1.000."""
    mapeo = {**MAPEO_INDICES, **(mapeo or {})}
    montos, sin_mapeo = {}, []
    for ins in cot["insumos"]:
        cod = mapeo.get(ins["insumo"])
        if cod is None:
            sin_mapeo.append(ins["insumo"])
            cod = "39"
        montos[cod] = montos.get(cod, 0.0) + ins["parcial"]
    herramientas = cot["costo_directo"] - sum(i["parcial"] for i in cot["insumos"])
    montos["37"] = montos.get("37", 0.0) + max(herramientas, 0.0)
    montos["39"] = montos.get("39", 0.0) + cot["gastos_generales"] + cot["utilidad"]
    total = sum(montos.values())
    coef = {k: v / total for k, v in montos.items()}

    for k in sorted(coef, key=coef.get):  # agrupa monomios pequeños (y excedentes) en el 39
        if k != "39" and (coef[k] < minimo or len(coef) > max_monomios):
            coef["39"] += coef.pop(k)
    coef = {k: round(v, 3) for k, v in coef.items()}
    mayor = max(coef, key=coef.get)
    coef[mayor] = round(coef[mayor] + 1 - sum(coef.values()), 3)  # cierra la suma en 1.000
    monomios = sorted(coef.items(), key=lambda kv: -kv[1])
    texto = "K = " + " + ".join(f"{c:.3f}·({k}r/{k}o)" for k, c in monomios)
    return {"coeficientes": dict(monomios), "texto": texto, "insumos_sin_indice": sin_mapeo,
            "descripcion": {k: NOMBRES_INDICE.get(k, "?") for k, _ in monomios}}


def factor_k(formula: dict, indices: pd.DataFrame, fecha_base: date, mes: date) -> float:
    return round(sum(c * indice(indices, k, mes) / indice(indices, k, fecha_base)
                     for k, c in formula["coeficientes"].items()), 3)


def usa_indices_proyectados(indices: pd.DataFrame, formula: dict, mes: date) -> bool:
    if "proyectado" not in indices.columns:
        return False
    m = indices[(indices["fecha"] == date(mes.year, mes.month, 1)) & indices["codigo"].isin(formula["coeficientes"])]
    return bool(m["proyectado"].astype(str).str.lower().isin(["true", "1"]).any())


# ======================= Valorizaciones =======================
def valorizar_obra(cot: dict, avance: List[dict], fecha_base: date, adelanto_directo_pct: float = 0.0,
                   programado: Optional[Dict[str, float]] = None, formula: Optional[dict] = None,
                   indices: Optional[pd.DataFrame] = None, permitir_mayores_metrados: bool = False,
                   tolerancia_metrado: float = 0.001) -> dict:
    """
    cot:        presupuesto contractual (salida de cotizador.cotizar)
    avance:     [{"mes": date, "metrados": {item: metrado ejecutado ese mes}}, ...] en orden
    fecha_base: mes del presupuesto (índices Io)
    programado: {"AAAA-MM": monto acumulado programado (valorización bruta)} → alerta de atraso
    tolerancia_metrado: exceso relativo que se considera redondeo (0.1%) y no mayor metrado
    """
    formula = formula or formula_polinomica(cot)
    indices = indices if indices is not None else cargar_indices()
    gg_pct, ut_pct = cot["parametros"]["gastos_generales_pct"], cot["parametros"]["utilidad_pct"]
    partidas = {p["item"]: p for p in cot["partidas"] if p["codigo"]}
    monto_contrato = cot["subtotal"]  # sin IGV
    adelanto = r2(monto_contrato * adelanto_directo_pct)
    acumulado = {it: 0.0 for it in partidas}
    amortizado, val_acum, salida = 0.0, 0.0, []

    for n, mes_av in enumerate(avance, 1):
        mes = mes_av["mes"] if isinstance(mes_av["mes"], date) else date.fromisoformat(str(mes_av["mes"]))
        desconocidos = set(mes_av["metrados"]) - set(partidas)
        if desconocidos:
            raise ValueError(f"Valorización {n}: ítems que no están en el presupuesto: {sorted(desconocidos)[:10]}")
        detalle, alertas, cd = [], [], 0.0
        for it, p in partidas.items():
            actual = float(mes_av["metrados"].get(it, 0.0))
            if actual < 0:
                raise ValueError(f"Valorización {n}, ítem {it}: metrado negativo.")
            anterior = acumulado[it]
            disponible = p["metrado"] - anterior
            if actual > disponible + max(1e-6, tolerancia_metrado * p["metrado"]):
                alertas.append(f"Ítem {it}: mayor metrado de {actual - disponible:,.2f} {p['unidad']} sobre lo "
                               f"contratado. Requiere aprobación de la Entidad antes de valorizarse.")
                if not permitir_mayores_metrados:
                    actual = max(disponible, 0.0)
            elif actual > disponible:  # exceso mínimo por redondeo: se ajusta sin alertar
                actual = max(disponible, 0.0)
            acumulado[it] = anterior + actual
            monto = r2(actual * p["precio_unitario"])
            cd += monto
            if actual or anterior:
                detalle.append({"item": it, "descripcion": p["descripcion"], "unidad": p["unidad"],
                                "metrado_contratado": p["metrado"], "precio_unitario": p["precio_unitario"],
                                "anterior": round(anterior, 2), "actual": round(actual, 2),
                                "acumulado": round(acumulado[it], 2), "saldo": round(p["metrado"] - acumulado[it], 2),
                                "avance_pct": round(acumulado[it] / p["metrado"] * 100, 2), "monto_actual": monto,
                                "monto_acumulado": r2(acumulado[it] * p["precio_unitario"])})
        cd = r2(cd)
        bruta = r2(cd * (1 + gg_pct + ut_pct))
        k = factor_k(formula, indices, fecha_base, mes)
        reajuste = r2(bruta * (k - 1))
        val_acum += bruta
        obra_concluida = all(acumulado[it] >= p["metrado"] - 1e-6 for it, p in partidas.items())
        if adelanto:  # proporcional a lo valorizado; el saldo se amortiza completo al concluir la obra
            amort = adelanto - amortizado if obra_concluida else min(adelanto * bruta / monto_contrato,
                                                                      adelanto - amortizado)
            amort = r2(amort)
        else:
            amort = 0.0
        amortizado += amort
        subtotal = r2(bruta + reajuste - amort)
        igv = r2(subtotal * C.IGV)
        if usa_indices_proyectados(indices, formula, mes):
            alertas.append("K provisional: usa índices proyectados por el modelo, no publicados por el INEI. "
                           "Recalcula el reajuste cuando el INEI publique los índices del mes.")
        clave = f"{mes:%Y-%m}"
        prog = (programado or {}).get(clave)
        if prog and val_acum < 0.8 * prog:
            alertas.append(f"Atraso: valorizado acumulado S/ {val_acum:,.2f} es {val_acum / prog:.1%} del programado "
                           f"(S/ {prog:,.2f}), menos del 80%. Revisa la exigencia de calendario acelerado.")
        salida.append({
            "numero": n, "mes": clave, "costo_directo": cd, "gastos_generales": r2(cd * gg_pct),
            "utilidad": r2(cd * ut_pct), "valorizacion_bruta": bruta, "k": k, "reajuste": reajuste,
            "amortizacion_adelanto": amort, "subtotal": subtotal, "igv": igv, "neto_a_pagar": r2(subtotal + igv),
            "avance_acumulado_pct": round(val_acum / monto_contrato * 100, 2),
            "programado_acumulado": prog, "alertas": alertas, "detalle": detalle,
        })
    return {"monto_contrato_sin_igv": monto_contrato, "adelanto_directo": adelanto,
            "adelanto_por_amortizar": r2(adelanto - amortizado), "fecha_base": f"{fecha_base:%Y-%m}",
            "formula_polinomica": formula, "valorizaciones": salida}


def leer_avance_excel(ruta) -> List[dict]:
    """Planilla de avance: columna `item` + una columna por mes (AAAA-MM) con el metrado ejecutado."""
    df = pd.read_excel(ruta, dtype={"item": str})
    if "item" not in df.columns:
        raise ValueError("La planilla de avance necesita una columna 'item'.")
    meses = [c for c in df.columns if c != "item"]
    out = []
    for m in meses:
        mes = pd.Timestamp(str(m)[:7] + "-01").date()
        serie = df.set_index("item")[m].dropna()
        out.append({"mes": mes, "metrados": {str(k): float(v) for k, v in serie.items() if v}})
    return out


def exportar_excel(res: dict, ruta) -> Path:
    ruta = Path(ruta)
    vals = res["valorizaciones"]
    cols = ["numero", "mes", "costo_directo", "gastos_generales", "utilidad", "valorizacion_bruta", "k", "reajuste",
            "amortizacion_adelanto", "subtotal", "igv", "neto_a_pagar", "avance_acumulado_pct", "programado_acumulado"]
    with pd.ExcelWriter(ruta) as w:
        pd.DataFrame([{c: v[c] for c in cols} for v in vals]).to_excel(w, sheet_name="Resumen", index=False)
        f = res["formula_polinomica"]
        pd.DataFrame([{"indice": k, "descripcion": f["descripcion"][k], "coeficiente": c}
                      for k, c in f["coeficientes"].items()]).to_excel(w, sheet_name="Fórmula polinómica", index=False)
        pd.DataFrame([{"mes": v["mes"], "alerta": a} for v in vals for a in v["alertas"]] or [{"mes": "", "alerta": "Sin alertas"}]
                     ).to_excel(w, sheet_name="Alertas", index=False)
        for v in vals[-6:]:  # detalle de las últimas 6 valorizaciones
            pd.DataFrame(v["detalle"]).to_excel(w, sheet_name=f"Val {v['numero']:02d} ({v['mes']})", index=False)
    return ruta
