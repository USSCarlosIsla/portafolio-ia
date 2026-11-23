"""Genera facturas sintéticas (PNG) + su verdad de referencia (JSON) para evaluar extractores.

La variedad es a propósito: 3 diseños, varios formatos de fecha, de números (1.234.567 vs 1,234,567.00)
y de NIT, rotación, ruido y desenfoque. Es lo que rompe a las reglas fijas (regex) y donde un LLM brilla.

Uso:  python generar_facturas.py --n 30
"""
import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from esquema import Factura, LineaFactura, digito_verificacion_nit

DATA_DIR = Path(__file__).resolve().parent / "data" / "facturas"
FONTS = {"sans": "arial.ttf", "sans_bold": "arialbd.ttf", "mono": "consola.ttf", "serif": "times.ttf"}

PROVEEDORES = ["Ferretería El Constructor S.A.S.", "Materiales Andinos Ltda.", "Depósito La Roca S.A.S.",
               "Aceros y Concretos del Norte S.A.", "Distribuidora Obra Gris S.A.S.", "Eléctricos Volta Ltda."]
CLIENTES = ["Constructora Horizonte S.A.S.", "Inversiones Altamira", "Consorcio Vial 2024",
            "Juan Pérez Rodríguez", "Edificaciones Modernas S.A.", "Arquitectura & Diseño Ltda."]
PRODUCTOS = [("Cemento gris 50 kg", 32000, 38000), ("Varilla corrugada 1/2\" x 6 m", 28000, 35000),
             ("Arena lavada (m³)", 85000, 110000), ("Ladrillo tolete (und)", 650, 900),
             ("Bloque de concreto 15 cm", 2800, 3600), ("Grava triturada (m³)", 95000, 125000),
             ("Malla electrosoldada 6 m", 145000, 180000), ("Tubo PVC sanitario 4\"", 52000, 68000),
             ("Cable THHN #12 (rollo 100 m)", 210000, 260000), ("Mano de obra oficial (día)", 120000, 150000),
             ("Alquiler formaleta (día)", 15000, 22000), ("Impermeabilizante 5 gal", 180000, 240000)]
MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre",
         "octubre", "noviembre", "diciembre"]


# ---------------- Formatos variados ----------------
def fmt_dinero(x, estilo):
    if estilo == "co":
        return "$ " + f"{x:,.0f}".replace(",", ".")
    if estilo == "co_dec":
        return "$" + f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"${x:,.2f}"  # estilo "us"


def fmt_fecha(d, estilo):
    return {"iso": d.isoformat(), "dmy": d.strftime("%d/%m/%Y"),
            "largo": f"{d.day} de {MESES[d.month - 1]} de {d.year}",
            "corto": f"{d.day:02d}-{MESES[d.month - 1][:3]}-{d.year}"}[estilo]


def fmt_nit(nit, dv, estilo):
    puntos = f"{int(nit):,}".replace(",", ".")
    return {"puntos": f"NIT: {puntos}-{dv}", "plano": f"NIT {nit}-{dv}",
            "espacios": f"N.I.T. {nit[:3]} {nit[3:6]} {nit[6:]} - {dv}"}[estilo]


def font(name, size):
    try:
        return ImageFont.truetype(FONTS[name], size)
    except OSError:
        return ImageFont.load_default(size)


# ---------------- Datos aleatorios coherentes ----------------
def factura_aleatoria(rng: random.Random) -> Factura:
    nit = str(rng.randint(800_000_000, 901_999_999))
    lineas = []
    for nombre, lo, hi in rng.sample(PRODUCTOS, rng.randint(2, 6)):
        cant = rng.randint(1, 40) if lo > 10000 else rng.choice([100, 250, 500, 1000])
        precio = round(rng.randint(lo, hi), -2)
        lineas.append(LineaFactura(descripcion=nombre, cantidad=cant, precio_unitario=precio,
                                   total_linea=cant * precio))
    subtotal = sum(l.total_linea for l in lineas)
    iva = round(subtotal * 0.19)
    return Factura(
        numero_factura=rng.choice(["FE", "FV", "F", "SETT"]) + "-" + str(rng.randint(100, 99999)),
        fecha_emision=date(2024, 1, 1) + timedelta(days=rng.randint(0, 600)),
        proveedor_nombre=rng.choice(PROVEEDORES), proveedor_nit=nit, proveedor_dv=digito_verificacion_nit(nit),
        cliente_nombre=rng.choice(CLIENTES), lineas=lineas, subtotal=subtotal, iva=iva, total=subtotal + iva,
    )


# ---------------- Diseños ----------------
def diseno_clasico(f, rng, est):
    img = Image.new("L", (1100, 1400), 255)
    d = ImageDraw.Draw(img)
    d.text((60, 50), f.proveedor_nombre, font=font("sans_bold", 34), fill=0)
    d.text((60, 100), fmt_nit(f.proveedor_nit, f.proveedor_dv, est["nit"]), font=font("sans", 24), fill=0)
    d.text((60, 135), "Calle " + str(rng.randint(10, 150)) + " # " + str(rng.randint(1, 99)) + "-" +
           str(rng.randint(1, 99)) + ", Bogotá", font=font("sans", 22), fill=60)
    d.rectangle((640, 50, 1040, 170), outline=0, width=2)
    d.text((660, 65), "FACTURA ELECTRÓNICA DE VENTA", font=font("sans_bold", 20), fill=0)
    d.text((660, 100), f"No. {f.numero_factura}", font=font("sans_bold", 28), fill=0)
    d.text((660, 140), "Fecha: " + fmt_fecha(f.fecha_emision, est["fecha"]), font=font("sans", 20), fill=0)
    d.text((60, 220), "Cliente: " + f.cliente_nombre, font=font("sans", 24), fill=0)
    y = 300
    cols = [60, 600, 720, 900]
    d.rectangle((50, y - 10, 1050, y + 30), fill=225)
    for x, h in zip(cols, ["Descripción", "Cant.", "Vr. Unitario", "Vr. Total"]):
        d.text((x, y), h, font=font("sans_bold", 22), fill=0)
    y += 50
    for l in f.lineas:
        vals = [l.descripcion, f"{l.cantidad:g}", fmt_dinero(l.precio_unitario, est["dinero"]),
                fmt_dinero(l.total_linea, est["dinero"])]
        for x, v in zip(cols, vals):
            d.text((x, y), v, font=font("sans", 21), fill=0)
        y += 42
        d.line((50, y - 8, 1050, y - 8), fill=200)
    y += 30
    for label, v in [("Subtotal", f.subtotal), ("IVA 19%", f.iva), ("TOTAL A PAGAR", f.total)]:
        d.text((700, y), label, font=font("sans_bold", 23), fill=0)
        d.text((900, y), fmt_dinero(v, est["dinero"]), font=font("sans_bold", 23), fill=0)
        y += 40
    return img.crop((0, 0, 1100, y + 80))


def diseno_centrado(f, rng, est):
    img = Image.new("L", (1100, 1400), 255)
    d = ImageDraw.Draw(img)

    def centro(y, txt, fnt):
        w = d.textlength(txt, font=fnt)
        d.text(((1100 - w) / 2, y), txt, font=fnt, fill=0)

    centro(40, f.proveedor_nombre.upper(), font("serif", 38))
    centro(95, fmt_nit(f.proveedor_nit, f.proveedor_dv, est["nit"]) + "  ·  Régimen común", font("serif", 24))
    d.line((80, 140, 1020, 140), fill=0, width=3)
    d.text((80, 165), f"Factura de venta N° {f.numero_factura}", font=font("serif", 28), fill=0)
    d.text((680, 165), "Expedida: " + fmt_fecha(f.fecha_emision, est["fecha"]), font=font("serif", 24), fill=0)
    d.text((80, 215), "Señores: " + f.cliente_nombre, font=font("serif", 26), fill=0)
    y = 290
    cols = [80, 190, 700, 870]  # orden de columnas DISTINTO al diseño clásico
    for x, h in zip(cols, ["Cant", "Concepto", "V/r unit.", "V/r total"]):
        d.text((x, y), h, font=font("serif", 24), fill=0)
    d.line((80, y + 35, 1020, y + 35), fill=0)
    y += 50
    for l in f.lineas:
        for x, v in zip(cols, [f"{l.cantidad:g}", l.descripcion, fmt_dinero(l.precio_unitario, est["dinero"]),
                               fmt_dinero(l.total_linea, est["dinero"])]):
            d.text((x, y), v, font=font("serif", 23), fill=0)
        y += 40
    d.line((80, y + 5, 1020, y + 5), fill=0)
    y += 25
    for label, v in [("Valor bruto:", f.subtotal), ("Impuesto IVA:", f.iva), ("Valor neto:", f.total)]:
        d.text((620, y), label, font=font("serif", 25), fill=0)
        d.text((850, y), fmt_dinero(v, est["dinero"]), font=font("serif", 25), fill=0)
        y += 40
    return img.crop((0, 0, 1100, y + 80))


def diseno_tirilla(f, rng, est):
    img = Image.new("L", (760, 1600), 255)
    d = ImageDraw.Draw(img)
    fm = font("mono", 22)
    lines = [f.proveedor_nombre, fmt_nit(f.proveedor_nit, f.proveedor_dv, est["nit"]), "-" * 48,
             f"FACTURA {f.numero_factura}", fmt_fecha(f.fecha_emision, est["fecha"]),
             f"CLIENTE: {f.cliente_nombre}", "-" * 48]
    for l in f.lineas:
        lines.append(l.descripcion[:44])
        lines.append(f"  {l.cantidad:g} x {fmt_dinero(l.precio_unitario, est['dinero'])}".ljust(30) +
                     fmt_dinero(l.total_linea, est["dinero"]).rjust(18))
    lines += ["-" * 48, "SUBTOTAL".ljust(30) + fmt_dinero(f.subtotal, est["dinero"]).rjust(18),
              "IVA".ljust(30) + fmt_dinero(f.iva, est["dinero"]).rjust(18),
              "TOTAL".ljust(30) + fmt_dinero(f.total, est["dinero"]).rjust(18), "", "GRACIAS POR SU COMPRA"]
    y = 30
    for line in lines:
        d.text((30, y), line, font=fm, fill=0)
        y += 32
    return img.crop((0, 0, 760, y + 40))


def degradar(img, rng):
    """Simula una foto/escaneo: rotación leve, desenfoque y ruido."""
    img = img.rotate(rng.uniform(-1.5, 1.5), expand=True, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 0.9)))
    arr = np.asarray(img, dtype=float) + np.random.default_rng(rng.randint(0, 10**6)).normal(0, rng.uniform(0, 12), img.size[::-1])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    disenos = [diseno_clasico, diseno_centrado, diseno_tirilla]
    for i in range(args.n):
        f = factura_aleatoria(rng)
        est = {"dinero": rng.choice(["co", "co_dec", "us"]), "fecha": rng.choice(["iso", "dmy", "largo", "corto"]),
               "nit": rng.choice(["puntos", "plano", "espacios"])}
        img = degradar(disenos[i % 3](f, rng, est), rng)
        img.save(DATA_DIR / f"factura_{i:03d}.png")
        (DATA_DIR / f"factura_{i:03d}.json").write_text(f.model_dump_json(indent=2), encoding="utf-8")
    print(f"{args.n} facturas generadas en {DATA_DIR}")


if __name__ == "__main__":
    main()
