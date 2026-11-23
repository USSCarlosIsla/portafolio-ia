"""Genera plantillas de ejemplo y una obra GRANDE para probar el cotizador a escala.

Crea en data/:
  * plantilla_catalogo.xlsx  → formato para cargar TU base de insumos y APU (hojas insumos / apu).
  * obra_grande_metrados.xlsx → planilla de metrados jerárquica de un conjunto habitacional:
      01 ESTRUCTURAS > 01.01 Bloque A > 01.01.01 Piso 1 > 01.01.01.01 partida ...
    (por defecto 10 bloques × 8 pisos ≈ 1.000 partidas, del orden de decenas de millones de soles).

Uso:  python generar_obra_grande.py --bloques 10 --pisos 8
"""
import argparse
import random
import time
from pathlib import Path

import pandas as pd

from catalogo import Catalogo
from cotizador import cotizar, exportar_excel, leer_metrados_excel

DATA = Path(__file__).resolve().parent / "data"
OUT = Path(__file__).resolve().parent / "outputs"

# partidas por piso: (código, metrado base) — se multiplican por un factor aleatorio por piso
ESTRUCTURAS = [("P05", 18), ("P06", 22), ("P07", 30), ("P08", 150), ("P09", 180), ("P10", 9500)]
ARQUITECTURA = [("P11", 140), ("P12", 260), ("P13", 800), ("P14", 420), ("P15", 800)]
CIMENTACION = [("P01", 380), ("P02", 160), ("P03", 210), ("P04", 520)]


def generar(bloques: int, pisos: int, seed=0):
    rng = random.Random(seed)
    filas = []

    def titulo(item, desc):
        filas.append({"item": item, "descripcion": desc, "codigo": None, "metrado": None})

    def partidas(prefijo, lista):
        for k, (cod, base) in enumerate(lista, 1):
            filas.append({"item": f"{prefijo}.{k:02d}", "descripcion": None, "codigo": cod,
                          "metrado": round(base * rng.uniform(0.85, 1.15), 2)})

    titulo("01", "ESTRUCTURAS")
    for b in range(1, bloques + 1):
        titulo(f"01.{b:02d}", f"Bloque {chr(64 + b)}")
        titulo(f"01.{b:02d}.00", "Cimentación")
        partidas(f"01.{b:02d}.00", CIMENTACION)
        for p in range(1, pisos + 1):
            titulo(f"01.{b:02d}.{p:02d}", f"Piso {p}")
            partidas(f"01.{b:02d}.{p:02d}", ESTRUCTURAS)
    titulo("02", "ARQUITECTURA")
    for b in range(1, bloques + 1):
        titulo(f"02.{b:02d}", f"Bloque {chr(64 + b)}")
        for p in range(1, pisos + 1):
            titulo(f"02.{b:02d}.{p:02d}", f"Piso {p}")
            partidas(f"02.{b:02d}.{p:02d}", ARQUITECTURA)
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bloques", type=int, default=10)
    ap.add_argument("--pisos", type=int, default=8)
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    Catalogo.ejemplo().a_excel(DATA / "plantilla_catalogo.xlsx")
    df = generar(args.bloques, args.pisos)
    df.to_excel(DATA / "obra_grande_metrados.xlsx", index=False)
    n = int(df["codigo"].notna().sum())
    print(f"Planilla: {len(df)} filas ({n} partidas, {len(df) - n} títulos)")

    t0 = time.perf_counter()
    cat = Catalogo.desde_excel(DATA / "plantilla_catalogo.xlsx")
    items = leer_metrados_excel(DATA / "obra_grande_metrados.xlsx")
    t1 = time.perf_counter()
    cot = cotizar(items, cat=cat)
    t2 = time.perf_counter()
    exportar_excel(cot, OUT / "presupuesto_obra_grande.xlsx")
    t3 = time.perf_counter()

    print(f"Total del presupuesto: S/ {cot['total']:,.2f}")
    for f in cot["partidas"]:
        if f["codigo"] is None and f["item"].count(".") == 0:
            print(f"  {f['item']} {f['descripcion']:<14} S/ {f['parcial']:>16,.2f}")
    print(f"Tiempos → leer Excel: {t1 - t0:.2f}s | calcular: {t2 - t1:.2f}s | exportar Excel: {t3 - t2:.2f}s")


if __name__ == "__main__":
    main()
