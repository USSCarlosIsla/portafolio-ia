"""Genera facturas electrónicas peruanas sintéticas (PNG = representación impresa) + verdad (JSON).

La variedad es a propósito: 3 diseños, formatos de fecha (17/03/2027, 2027-03-17, 17-mar-2027), de
moneda (S/ 1,234.56 · S/. 1234.56 · US$ 1,234.56), etiquetas distintas ("OP. GRAVADA" / "Valor de venta" /
"SUBTOTAL"), rotación, ruido y desenfoque. Los RUC son válidos (dígito verificador SUNAT) pero inventados.
Los productos y precios vienen del catálogo del cotizador de obras (módulo 10), cuando está disponible.

Uso:  python generar_facturas.py --n 30
"""
import argparse
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from esquema import IGV, Factura, LineaFactura, digito_verificador_ruc

DATA_DIR = Path(__file__).resolve().parent / "data" / "facturas"
FONTS = {"sans": "arial.ttf", "sans_bold": "arialbd.ttf", "mono": "consola.ttf", "serif": "times.ttf"}

PROVEEDORES = ["Ferretería El Constructor S.A.C.", "Materiales Andinos E.I.R.L.", "Distribuidora La Roca S.A.C.",
               "Aceros y Concretos del Sur S.A.", "Depósito Obra Gris S.R.L.", "Eléctricos Volta S.A.C."]
CLIENTES = ["Constructora Horizonte S.A.C.", "Inversiones Altamira S.A.", "Consorcio Vial Lima Norte",
            "Edificaciones Modernas S.A.C.", "Arquitectura & Diseño E.I.R.L.", "Inmobiliaria Los Pinos S.A."]
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "set", "oct", "nov", "dic"]


def productos():
    """(descripción, precio sin IGV) desde el catálogo del cotizador; lista propia si no está."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "10-cotizador-obras"))
        from catalogo import INSUMOS
        return [(i.descripcion, i.precio) for i in INSUMOS.values() if i.tipo == "MAT"]
    except ImportError:
        return [("Cemento Portland tipo I (42.5 kg)", 30.5), ("Acero corrugado fy=4200 kg/cm²", 4.25),
                ("Ladrillo King Kong 18 huecos", 1.15), ("Arena gruesa", 62.0), ("Piedra chancada 1/2\"", 78.0)]


def ruc_aleatorio(rng, prefijo="20"):
    base = prefijo + "".join(str(rng.randint(0, 9)) for _ in range(8))
    return base + digito_verificador_ruc(base)


# ---------------- Formatos variados ----------------
def fmt_dinero(x, est):
    simbolo = "US$" if est["moneda"] == "USD" else est["simbolo"]
    return f"{simbolo} {x:,.2f}" if est["miles"] else f"{simbolo} {x:.2f}"


def fmt_fecha(d, estilo):
    return {"dmy": d.strftime("%d/%m/%Y"), "iso": d.isoformat(),
            "corto": f"{d.day:02d}-{MESES[d.month - 1]}-{d.year}"}[estilo]


def font(name, size):
    try:
        return ImageFont.truetype(FONTS[name], size)
    except OSError:
        return ImageFont.load_default(size)


# ---------------- Datos aleatorios coherentes ----------------
def factura_aleatoria(rng: random.Random, moneda: str) -> Factura:
    tc = 3.75 if moneda == "USD" else 1.0  # tipo de cambio referencial para facturas en dólares
    lineas = []
    for nombre, precio in rng.sample(productos(), rng.randint(2, 6)):
        cant = rng.choice([100, 250, 500, 1000]) if precio < 5 else rng.randint(1, 60)
        vu = round(precio * rng.uniform(0.95, 1.08) / tc, 2)
        lineas.append(LineaFactura(descripcion=nombre, cantidad=cant, valor_unitario=vu,
                                   valor_venta=round(cant * vu, 2)))
    op = round(sum(l.valor_venta for l in lineas), 2)
    igv = round(op * IGV, 2)
    return Factura(
        serie=rng.choice(["F001", "F002", "FA01", "E001"]), correlativo=f"{rng.randint(1, 99999):08d}",
        fecha_emision=date(2026, 1, 1) + timedelta(days=rng.randint(0, 600)),
        emisor_ruc=ruc_aleatorio(rng), emisor_razon_social=rng.choice(PROVEEDORES),
        cliente_ruc=ruc_aleatorio(rng, rng.choice(["20", "20", "10"])), cliente_razon_social=rng.choice(CLIENTES),
        moneda=moneda, lineas=lineas, op_gravada=op, igv=igv, importe_total=round(op + igv, 2),
    )


# ---------------- Diseños ----------------
def recuadro_sunat(d, x, y, f, fnt_b, fnt):
    """El recuadro típico de la representación impresa: RUC, tipo de comprobante y número."""
    d.rectangle((x, y, x + 400, y + 135), outline=0, width=3)
    d.text((x + 20, y + 12), f"R.U.C. {f.emisor_ruc}", font=fnt_b, fill=0)
    d.text((x + 20, y + 50), "FACTURA ELECTRÓNICA", font=fnt_b, fill=0)
    d.text((x + 20, y + 90), f"{f.serie} - {f.correlativo}", font=fnt, fill=0)


def diseno_clasico(f, rng, est):
    img = Image.new("L", (1100, 1450), 255)
    d = ImageDraw.Draw(img)
    d.text((60, 50), f.emisor_razon_social, font=font("sans_bold", 32), fill=0)
    d.text((60, 100), f"Av. {rng.choice(['Argentina', 'Colonial', 'Túpac Amaru', 'Industrial'])} "
                      f"{rng.randint(100, 3999)} - Lima", font=font("sans", 21), fill=60)
    recuadro_sunat(d, 640, 40, f, font("sans_bold", 24), font("sans_bold", 26))
    d.text((60, 210), "Fecha de emisión: " + fmt_fecha(f.fecha_emision, est["fecha"]), font=font("sans", 22), fill=0)
    d.text((60, 245), "Señor(es): " + f.cliente_razon_social, font=font("sans", 22), fill=0)
    d.text((60, 280), f"RUC: {f.cliente_ruc}", font=font("sans", 22), fill=0)
    d.text((640, 210), "Moneda: " + ("SOLES" if f.moneda == "PEN" else "DÓLARES AMERICANOS"),
           font=font("sans", 22), fill=0)
    y, cols = 350, [60, 560, 680, 880]
    d.rectangle((50, y - 10, 1050, y + 30), fill=225)
    for x, h in zip(cols, ["Descripción", "Cant.", "V. Unitario", "V. Venta"]):
        d.text((x, y), h, font=font("sans_bold", 21), fill=0)
    y += 50
    for l in f.lineas:
        for x, v in zip(cols, [l.descripcion[:40], f"{l.cantidad:g}", f"{l.valor_unitario:,.2f}",
                               f"{l.valor_venta:,.2f}"]):
            d.text((x, y), v, font=font("sans", 20), fill=0)
        y += 40
        d.line((50, y - 8, 1050, y - 8), fill=200)
    y += 30
    for label, v in [("OP. GRAVADA", f.op_gravada), ("I.G.V. 18%", f.igv), ("IMPORTE TOTAL", f.importe_total)]:
        d.text((640, y), label, font=font("sans_bold", 22), fill=0)
        d.text((870, y), fmt_dinero(v, est), font=font("sans_bold", 22), fill=0)
        y += 40
    d.text((60, y + 30), "Representación impresa de la Factura Electrónica", font=font("sans", 18), fill=80)
    return img.crop((0, 0, 1100, y + 90))


def diseno_centrado(f, rng, est):
    img = Image.new("L", (1100, 1450), 255)
    d = ImageDraw.Draw(img)
    w = d.textlength(f.emisor_razon_social.upper(), font=font("serif", 34))
    d.text(((1100 - w) / 2, 35), f.emisor_razon_social.upper(), font=font("serif", 34), fill=0)
    recuadro_sunat(d, 350, 95, f, font("serif", 26), font("serif", 26))
    d.text((80, 260), "Cliente: " + f.cliente_razon_social, font=font("serif", 25), fill=0)
    d.text((80, 298), f"R.U.C. N° {f.cliente_ruc}", font=font("serif", 25), fill=0)
    d.text((660, 260), "Emisión: " + fmt_fecha(f.fecha_emision, est["fecha"]), font=font("serif", 24), fill=0)
    d.text((660, 298), "Moneda: " + f.moneda, font=font("serif", 24), fill=0)
    y, cols = 370, [80, 190, 700, 880]  # columnas en otro orden
    for x, h in zip(cols, ["Cant", "Descripción", "Valor unit.", "Importe"]):
        d.text((x, y), h, font=font("serif", 23), fill=0)
    d.line((80, y + 35, 1020, y + 35), fill=0)
    y += 50
    for l in f.lineas:
        for x, v in zip(cols, [f"{l.cantidad:g}", l.descripcion[:38], f"{l.valor_unitario:.2f}", f"{l.valor_venta:.2f}"]):
            d.text((x, y), v, font=font("serif", 22), fill=0)
        y += 38
    d.line((80, y + 5, 1020, y + 5), fill=0)
    y += 25
    for label, v in [("Valor de venta:", f.op_gravada), ("IGV (18%):", f.igv), ("Total a pagar:", f.importe_total)]:
        d.text((600, y), label, font=font("serif", 24), fill=0)
        d.text((830, y), fmt_dinero(v, est), font=font("serif", 24), fill=0)
        y += 40
    return img.crop((0, 0, 1100, y + 70))


def diseno_tirilla(f, rng, est):
    img = Image.new("L", (760, 1700), 255)
    d = ImageDraw.Draw(img)
    lines = [f.emisor_razon_social, f"RUC {f.emisor_ruc}", "FACTURA ELECTRONICA", f"{f.serie}-{f.correlativo}",
             "FECHA: " + fmt_fecha(f.fecha_emision, est["fecha"]), "-" * 48, f"CLIENTE: {f.cliente_razon_social}"[:48],
             f"RUC: {f.cliente_ruc}", "-" * 48]
    for l in f.lineas:
        lines.append(l.descripcion[:46])
        lines.append(f"  {l.cantidad:g} x {l.valor_unitario:.2f}".ljust(30) + f"{l.valor_venta:,.2f}".rjust(18))
    lines += ["-" * 48, "SUBTOTAL".ljust(26) + fmt_dinero(f.op_gravada, est).rjust(22),
              "IGV 18%".ljust(26) + fmt_dinero(f.igv, est).rjust(22),
              "TOTAL".ljust(26) + fmt_dinero(f.importe_total, est).rjust(22), "",
              "Consulte su comprobante en SUNAT"]
    y = 30
    for line in lines:
        d.text((30, y), line, font=font("mono", 22), fill=0)
        y += 32
    return img.crop((0, 0, 760, y + 40))


def degradar(img, rng):
    """Simula una foto/escaneo: rotación leve, desenfoque y ruido."""
    img = img.rotate(rng.uniform(-1.5, 1.5), expand=True, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 0.9)))
    ruido = np.random.default_rng(rng.randint(0, 10**6)).normal(0, rng.uniform(0, 12), img.size[::-1])
    return Image.fromarray(np.clip(np.asarray(img, dtype=float) + ruido, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for viejo in DATA_DIR.glob("factura_*"):
        viejo.unlink()
    disenos = [diseno_clasico, diseno_centrado, diseno_tirilla]
    for i in range(args.n):
        moneda = "USD" if rng.random() < 0.15 else "PEN"
        f = factura_aleatoria(rng, moneda)
        est = {"moneda": moneda, "simbolo": rng.choice(["S/", "S/."]), "miles": rng.random() < 0.7,
               "fecha": rng.choice(["dmy", "dmy", "iso", "corto"])}
        degradar(disenos[i % 3](f, rng, est), rng).save(DATA_DIR / f"factura_{i:03d}.png")
        (DATA_DIR / f"factura_{i:03d}.json").write_text(f.model_dump_json(indent=2), encoding="utf-8")
    print(f"{args.n} facturas generadas en {DATA_DIR}")


if __name__ == "__main__":
    main()
