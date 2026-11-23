"""Evalúa un extractor campo por campo contra la verdad de referencia y exporta los resultados a Excel.

Uso:
  python evaluar.py --extractor ocr              # gratis, local
  python evaluar.py --extractor llm --limit 10   # usa la API de Claude (cuesta dinero; ver costo al final)

Las respuestas del LLM se guardan en outputs/cache_llm/ para no pagar dos veces por la misma factura.
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from esquema import Factura

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "facturas"
OUT = BASE / "outputs"
CAMPOS = ["serie", "correlativo", "fecha_emision", "emisor_ruc", "emisor_razon_social", "cliente_ruc",
          "cliente_razon_social", "moneda", "op_gravada", "igv", "importe_total"]


def norm_texto(s):
    s = "".join(c for c in unicodedata.normalize("NFD", str(s)) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]", "", s.lower())


def igual(campo, real, pred):
    if pred is None:
        return False
    if campo in ("op_gravada", "igv", "importe_total"):
        return abs(float(real) - float(pred)) < 0.02
    return norm_texto(real) == norm_texto(pred)


def lineas_correctas(real, pred):
    if len(real) != len(pred):
        return False
    return all(abs(r["valor_venta"] - p["valor_venta"]) < 0.02 and abs(r["cantidad"] - p["cantidad"]) < 1e-6
               for r, p in zip(real, pred))


def correr_llm(png):
    from extractor_llm import extraer_factura

    cache = OUT / "cache_llm" / (png.stem + ".json")
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    res = extraer_factura(str(png))
    data = {"factura": json.loads(res.factura.model_dump_json()) if res.factura else None,
            "intentos": res.intentos, "errores": res.errores, "costo_usd": res.costo_usd}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", choices=["ocr", "llm"], default="ocr")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    pngs = sorted(DATA.glob("*.png"))[: args.limit]
    if not pngs:
        raise SystemExit("No hay facturas. Ejecuta primero: python generar_facturas.py")

    filas, cabeceras, lineas, costo = [], [], [], 0.0
    for png in pngs:
        real = json.loads(png.with_suffix(".json").read_text(encoding="utf-8"))
        if args.extractor == "ocr":
            from extractor_ocr import extraer_factura_ocr
            pred = {k: (str(v) if k == "fecha_emision" and v else v) for k, v in extraer_factura_ocr(png).items()}
            pred["lineas"], extra = [], {}
        else:
            r = correr_llm(png)
            pred = r["factura"] or {}
            costo += r["costo_usd"]
            extra = {"intentos": r["intentos"], "revision_humana": bool(r["errores"] or not r["factura"])}

        fila = {"archivo": png.name, **{c: igual(c, real[c], pred.get(c)) for c in CAMPOS},
                "lineas": lineas_correctas(real["lineas"], pred.get("lineas", [])), **extra}
        fila["factura_perfecta"] = all(fila[c] for c in CAMPOS + ["lineas"])
        filas.append(fila)
        cabeceras.append({"archivo": png.name, **{c: pred.get(c) for c in CAMPOS}})
        lineas += [{"archivo": png.name, **l} for l in pred.get("lineas", [])]
        print(f"{png.name}: {'✓' if fila['factura_perfecta'] else '✗'}")

    df = pd.DataFrame(filas)
    print(f"\n=== Exactitud por campo — extractor '{args.extractor}' ({len(df)} facturas) ===")
    print((df[CAMPOS + ["lineas", "factura_perfecta"]].mean() * 100).round(1).to_string())
    if args.extractor == "llm":
        print(f"\nIntentos promedio: {df['intentos'].mean():.2f} | marcadas para revisión: "
              f"{df['revision_humana'].sum()} | costo total ≈ ${costo:.3f} USD "
              f"(${costo / len(df):.4f} por factura; incluye solo llamadas nuevas, no la caché)")

    excel = OUT / f"facturas_extraidas_{args.extractor}.xlsx"
    with pd.ExcelWriter(excel) as w:
        pd.DataFrame(cabeceras).to_excel(w, sheet_name="facturas", index=False)
        pd.DataFrame(lineas).to_excel(w, sheet_name="lineas", index=False)
        df.to_excel(w, sheet_name="evaluacion", index=False)
    print(f"\nResultados exportados a {excel}")


if __name__ == "__main__":
    main()
