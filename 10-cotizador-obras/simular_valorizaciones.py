"""Simula la ejecución de la obra grande durante 12 meses y genera todas sus valorizaciones.

  * Presupuesto contractual: data/obra_grande_metrados.xlsx (de generar_obra_grande.py), fecha base 2026-10.
  * Calendario programado en curva S. La ejecución real va ~15% más lenta entre los meses 4 y 7
    (dispara la alerta de atraso) y un ítem se pasa de su metrado contratado (alerta de mayor metrado).
  * Guarda la planilla de avance (formato para tu flujo) y el Excel de valorizaciones.

Uso:  python simular_valorizaciones.py
"""
import math
import random
import time
from datetime import date
from pathlib import Path

import pandas as pd

from cotizador import cotizar, leer_metrados_excel
from valorizaciones import exportar_excel, leer_avance_excel, valorizar_obra

BASE = Path(__file__).resolve().parent
MESES = [date(2026 + (10 + k) // 12, (10 + k) % 12 + 1, 1) for k in range(12)]  # nov-2026 .. oct-2027


def curva_s(k, n):  # fracción acumulada programada al final del mes k (1..n)
    return 1 / (1 + math.exp(-10 * (k / n - 0.5)))


def main():
    rng = random.Random(1)
    cot = cotizar(leer_metrados_excel(BASE / "data" / "obra_grande_metrados.xlsx"))
    partidas = [p for p in cot["partidas"] if p["codigo"]]
    n = len(MESES)
    s = [0] + [(curva_s(k, n) - curva_s(0, n)) / (curva_s(n, n) - curva_s(0, n)) for k in range(1, n + 1)]
    # fracción de lo programado que realmente se logra (atraso a mitad de obra, luego se recupera)
    cumplimiento = [1, 1, 0.9, 0.72, 0.65, 0.7, 0.8, 0.9, 0.95, 1, 1, 1]

    filas = {p["item"]: {"item": p["item"]} for p in partidas}
    ejec = {p["item"]: 0.0 for p in partidas}
    for k, mes in enumerate(MESES):
        for p in partidas:
            if k == n - 1:
                m = round(p["metrado"] - ejec[p["item"]], 2)  # último mes: cierra exacto lo que falta
            else:
                objetivo = p["metrado"] * s[k + 1] * cumplimiento[k]
                m = round(max(0.0, objetivo - ejec[p["item"]]) * rng.uniform(0.95, 1.05), 2)
                m = min(m, round(p["metrado"] - ejec[p["item"]], 2))
            ejec[p["item"]] += m
            filas[p["item"]][f"{mes:%Y-%m}"] = m
    filas[partidas[5]["item"]][f"{MESES[-1]:%Y-%m}"] += round(partidas[5]["metrado"] * 0.08, 2)  # mayor metrado

    ruta_avance = BASE / "data" / "avance_obra_grande.xlsx"
    pd.DataFrame(filas.values()).to_excel(ruta_avance, index=False)
    programado = {f"{m:%Y-%m}": round(cot["costo_directo"] * (1 + 0.18) * s[k + 1], 2) for k, m in enumerate(MESES)}

    t0 = time.perf_counter()
    res = valorizar_obra(cot, leer_avance_excel(ruta_avance), fecha_base=date(2026, 10, 1),
                         adelanto_directo_pct=0.10, programado=programado)
    t1 = time.perf_counter()
    out = BASE / "outputs"
    out.mkdir(exist_ok=True)
    exportar_excel(res, out / "valorizaciones_obra_grande.xlsx")

    print(f"Contrato (sin IGV): S/ {res['monto_contrato_sin_igv']:,.2f} | adelanto directo 10%: "
          f"S/ {res['adelanto_directo']:,.2f}")
    print(res["formula_polinomica"]["texto"])
    print(f"\n{'N°':>2} {'mes':<8} {'bruta':>15} {'K':>6} {'reajuste':>12} {'amortiz.':>13} {'neto a pagar':>15} {'avance':>7}")
    for v in res["valorizaciones"]:
        print(f"{v['numero']:>2} {v['mes']:<8} {v['valorizacion_bruta']:>15,.2f} {v['k']:>6.3f} {v['reajuste']:>12,.2f} "
              f"{v['amortizacion_adelanto']:>13,.2f} {v['neto_a_pagar']:>15,.2f} {v['avance_acumulado_pct']:>6.1f}%"
              + ("  ⚠" * bool(v["alertas"])))
    print(f"\nAdelanto pendiente de amortizar: S/ {res['adelanto_por_amortizar']:,.2f}")
    print("Alertas:")
    for v in res["valorizaciones"]:
        for a in v["alertas"]:
            print(f"  [{v['mes']}] {a}")
    print(f"\n12 valorizaciones de {len(partidas)} partidas calculadas en {t1 - t0:.2f}s → {out / 'valorizaciones_obra_grande.xlsx'}")


if __name__ == "__main__":
    main()
