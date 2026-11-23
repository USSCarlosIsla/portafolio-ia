"""Motor de presupuesto: APU → precio unitario → presupuesto jerárquico → resumen de insumos → Excel.

Todo es cálculo determinista y auditable (nada de LLM aquí): mismo input, mismo resultado.
Funciona con el catálogo de ejemplo o con TU base de APU (Catalogo.desde_excel), y recibe los metrados
como lista o como planilla Excel con ítems jerárquicos (01, 01.01, 01.01.01...).

Uso de ejemplo:  python cotizador.py
"""
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

import catalogo as C
from catalogo import Catalogo

EJEMPLO = Catalogo.ejemplo()


def r2(x: float) -> float:
    return round(x + 1e-9, 2)


def analizar_partida(codigo: str, precios: Optional[Dict[str, float]] = None, cat: Catalogo = EJEMPLO) -> dict:
    """Análisis de precio unitario de una partida (equivalente a la hoja APU de S10)."""
    p = cat.partidas[codigo]
    precios = precios or {}
    lineas = [(ins, n, n * C.HORAS_JORNADA / p.rendimiento) for ins, n in p.cuadrilla.items()]
    lineas += [(ins, None, cant) for ins, cant in p.materiales.items()]

    detalle, por_tipo = [], {"MO": 0.0, "MAT": 0.0, "EQ": 0.0}
    for ins, cuadrilla, cantidad in lineas:
        i = cat.insumos[ins]
        precio = precios.get(ins, i.precio)
        parcial = cantidad * precio
        por_tipo[i.tipo] = por_tipo.get(i.tipo, 0.0) + parcial
        detalle.append({"insumo": ins, "descripcion": i.descripcion, "tipo": i.tipo, "unidad": i.unidad,
                        "cuadrilla": cuadrilla, "cantidad": round(cantidad, 4), "precio": precio,
                        "parcial": r2(parcial)})
    if por_tipo["MO"] > 0:
        herramientas = C.HERRAMIENTAS_PCT_MO * por_tipo["MO"]
        detalle.append({"insumo": "EQ-HER", "descripcion": "Herramientas manuales (% MO)", "tipo": "EQ",
                        "unidad": "%MO", "cuadrilla": None, "cantidad": C.HERRAMIENTAS_PCT_MO,
                        "precio": r2(por_tipo["MO"]), "parcial": r2(herramientas)})
        por_tipo["EQ"] += herramientas
    return {"codigo": codigo, "descripcion": p.descripcion, "unidad": p.unidad, "rendimiento": p.rendimiento,
            "mano_obra": r2(por_tipo["MO"]), "materiales": r2(por_tipo["MAT"]), "equipos": r2(por_tipo["EQ"]),
            "precio_unitario": r2(sum(por_tipo.values())), "detalle": detalle}


def normalizar_items(items) -> List[dict]:
    """Acepta [(codigo, metrado), ...] o [{"item", "descripcion", "codigo", "metrado"}, ...].
    Las filas sin código son TÍTULOS (agrupadores); su subtotal se calcula automáticamente."""
    out = []
    for n, it in enumerate(items, 1):
        if isinstance(it, (tuple, list)):
            it = {"item": f"{n:02d}", "codigo": it[0], "metrado": it[1]}
        it = dict(it)
        cod = it.get("codigo")
        it["codigo"] = None if cod is None or pd.isna(cod) or str(cod).strip() == "" else str(cod).strip()
        it["item"] = str(it.get("item") or f"{n:02d}")
        out.append(it)
    return out


def cotizar(items, precios: Optional[Dict[str, float]] = None, gastos_generales: float = C.GASTOS_GENERALES,
            utilidad: float = C.UTILIDAD, cat: Catalogo = EJEMPLO) -> dict:
    items = normalizar_items(items)
    errores = []
    for it in items:
        if it["codigo"] is None:
            continue
        if it["codigo"] not in cat.partidas:
            errores.append(f"Ítem {it['item']}: partida desconocida '{it['codigo']}'")
        elif not isinstance(it.get("metrado"), (int, float)) or pd.isna(it["metrado"]) or it["metrado"] <= 0:
            errores.append(f"Ítem {it['item']}: metrado inválido ({it.get('metrado')})")
    if errores:
        raise ValueError("; ".join(errores[:50]) + (f" (+{len(errores) - 50} más)" if len(errores) > 50 else ""))

    filas, apus, insumos = [], {}, {}
    for it in items:
        if it["codigo"] is None:  # título
            filas.append({"item": it["item"], "descripcion": it.get("descripcion", ""), "codigo": None,
                          "unidad": None, "metrado": None, "precio_unitario": None, "parcial": None})
            continue
        apu = apus.get(it["codigo"])
        if apu is None:
            apu = apus[it["codigo"]] = analizar_partida(it["codigo"], precios, cat)
        metrado = float(it["metrado"])
        filas.append({"item": it["item"], "descripcion": it.get("descripcion") or apu["descripcion"],
                      "codigo": it["codigo"], "unidad": apu["unidad"], "metrado": metrado,
                      "precio_unitario": apu["precio_unitario"], "parcial": r2(metrado * apu["precio_unitario"])})
        for d in apu["detalle"]:  # resumen de insumos: cuánto hay que comprar de cada cosa
            if d["insumo"] == "EQ-HER":
                continue
            acc = insumos.get(d["insumo"])
            if acc is None:
                acc = insumos[d["insumo"]] = {k: d[k] for k in ("insumo", "descripcion", "tipo", "unidad", "precio")}
                acc["cantidad"] = 0.0
            acc["cantidad"] += d["cantidad"] * metrado

    # Subtotales de títulos: suma de las partidas cuyo ítem empieza con el ítem del título
    for f in filas:
        if f["codigo"] is None:
            pref = f["item"] + "."
            f["parcial"] = r2(sum(g["parcial"] for g in filas if g["codigo"] and g["item"].startswith(pref)))

    for acc in insumos.values():
        acc["cantidad"] = round(acc["cantidad"], 2)
        acc["parcial"] = r2(acc["cantidad"] * acc["precio"])

    cd = r2(sum(f["parcial"] for f in filas if f["codigo"]))
    gg, ut = r2(cd * gastos_generales), r2(cd * utilidad)
    subtotal = r2(cd + gg + ut)
    igv = r2(subtotal * C.IGV)
    return {
        "moneda": C.MONEDA, "partidas": filas, "apu": list(apus.values()),
        "insumos": sorted(insumos.values(), key=lambda x: (x["tipo"], x["insumo"])),
        "costo_directo": cd, "gastos_generales": gg, "utilidad": ut, "subtotal": subtotal,
        "igv": igv, "total": r2(subtotal + igv),
        "parametros": {"gastos_generales_pct": gastos_generales, "utilidad_pct": utilidad, "igv_pct": C.IGV},
    }


def cotizar_con_proyeccion(items, fecha_ejecucion, **kw) -> dict:
    """Cotiza con los precios PROYECTADOS al mes de ejecución (modelo de pronostico.py) y agrega
    el total a precios de hoy y un rango (escenario bajo/alto del intervalo 80% de cada insumo).
    Los insumos que el modelo no conoce (p. ej., de tu catálogo propio) se quedan a precio actual."""
    from pronostico import proyectar_precios

    proy = proyectar_precios(fecha_ejecucion)
    cot = cotizar(items, precios=proy["precios"] or None, **kw)
    if proy["h"] > 0:
        hoy = cotizar(items, **kw)["total"]
        bajo = cotizar(items, precios=proy["bajo"], **kw)["total"]
        alto = cotizar(items, precios=proy["alto"], **kw)["total"]
        cot["proyeccion"] = {"fecha_ejecucion": str(fecha_ejecucion), "meses_adelante": proy["h"],
                             "total_precios_hoy": hoy, "total_proyectado": cot["total"],
                             "rango_80": [bajo, alto], "variacion_pct": round((cot["total"] / hoy - 1) * 100, 2),
                             "insumos": proy["insumos"]}
    return cot


def leer_metrados_excel(ruta) -> List[dict]:
    """Planilla con columnas: item, descripcion, codigo, metrado. Filas sin código = títulos."""
    df = pd.read_excel(ruta, dtype={"item": str, "codigo": str})
    faltan = {"item", "codigo", "metrado"} - set(df.columns)
    if faltan:
        raise ValueError(f"A la planilla le faltan columnas: {sorted(faltan)}")
    if "descripcion" not in df.columns:
        df["descripcion"] = ""
    df = df.dropna(how="all")
    return [{"item": r.item, "descripcion": "" if pd.isna(r.descripcion) else r.descripcion,
             "codigo": r.codigo, "metrado": None if pd.isna(r.metrado) else float(r.metrado)}
            for r in df.itertuples()]


def exportar_excel(cot: dict, ruta: Union[str, Path]) -> Path:
    ruta = Path(ruta)
    resumen = pd.DataFrame([
        ("Costo directo", cot["costo_directo"]),
        (f"Gastos generales ({cot['parametros']['gastos_generales_pct']:.0%})", cot["gastos_generales"]),
        (f"Utilidad ({cot['parametros']['utilidad_pct']:.0%})", cot["utilidad"]),
        ("Subtotal", cot["subtotal"]), (f"IGV ({cot['parametros']['igv_pct']:.0%})", cot["igv"]),
        ("TOTAL PRESUPUESTO", cot["total"]),
    ], columns=["Concepto", f"Monto ({cot['moneda']})"])
    with pd.ExcelWriter(ruta) as w:
        pd.DataFrame(cot["partidas"]).to_excel(w, sheet_name="Presupuesto", index=False)
        resumen.to_excel(w, sheet_name="Presupuesto", index=False, startrow=len(cot["partidas"]) + 2)
        filas = []
        for apu in cot["apu"]:
            filas.append({"insumo": f"{apu['codigo']} - {apu['descripcion']}",
                          "descripcion": f"Rend.: {apu['rendimiento']} {apu['unidad']}/día",
                          "parcial": apu["precio_unitario"]})
            filas += apu["detalle"] + [{}]
        pd.DataFrame(filas).to_excel(w, sheet_name="APU", index=False)
        pd.DataFrame(cot["insumos"]).to_excel(w, sheet_name="Insumos", index=False)
        if cot.get("proyeccion"):
            p = cot["proyeccion"]
            pd.DataFrame([{"fecha_ejecucion": p["fecha_ejecucion"], "total_precios_hoy": p["total_precios_hoy"],
                           "total_proyectado": p["total_proyectado"], "rango_bajo": p["rango_80"][0],
                           "rango_alto": p["rango_80"][1], "variacion_pct": p["variacion_pct"]}]
                         ).to_excel(w, sheet_name="Proyección", index=False)
            pd.DataFrame(p["insumos"]).to_excel(w, sheet_name="Proyección", index=False, startrow=3)
    return ruta


if __name__ == "__main__":
    ejemplo = [("P01", 18.5), ("P03", 12.0), ("P05", 4.2), ("P08", 38.0), ("P10", 520.0), ("P11", 85.0),
               ("P13", 170.0), ("P15", 170.0)]
    cot = cotizar(ejemplo)
    for p in cot["partidas"]:
        print(f"{p['codigo']} {p['descripcion'][:45]:<45} {p['metrado']:>8.2f} {p['unidad']:<3} "
              f"x {p['precio_unitario']:>8.2f} = {p['parcial']:>11,.2f}")
    for k in ["costo_directo", "gastos_generales", "utilidad", "subtotal", "igv", "total"]:
        print(f"{k:>20}: S/ {cot[k]:>12,.2f}")
    out = Path(__file__).resolve().parent / "outputs"
    out.mkdir(exist_ok=True)
    print("Excel:", exportar_excel(cot, out / "cotizacion_ejemplo.xlsx"))
