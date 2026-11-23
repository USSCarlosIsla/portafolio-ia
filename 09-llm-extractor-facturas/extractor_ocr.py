"""Baseline SIN LLM: OCR clásico (RapidOCR, corre local y gratis) + reglas con expresiones regulares.

Sirve para responder la pregunta que hará cualquier entrevistador: "¿de verdad necesitas un LLM?".
Las reglas funcionan con formatos conocidos, pero cada diseño, etiqueta o formato nuevo exige escribir
otra regla. No extrae las líneas de productos (haría falta reconstruir la tabla).
"""
import re
import unicodedata
from datetime import date

from rapidocr_onnxruntime import RapidOCR

_ocr = None
MESES = {m: i for i, m in enumerate(["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "set", "oct",
                                     "nov", "dic"], 1)}
MESES["sep"] = 9
DINERO = r"(?:S/\.?|US\$)\s?\d[\d,]*\.\d{2}"


def _sin_tildes(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def parse_dinero(s):
    return float(re.sub(r"[^\d.]", "", re.sub(r"^(S/\.?|US\$)", "", s.strip())))


def parse_fecha(t):
    t = _sin_tildes(t.lower())
    if m := re.search(r"(\d{4})-(\d{2})-(\d{2})", t):
        return date(int(m[1]), int(m[2]), int(m[3]))
    if m := re.search(r"(\d{2})/(\d{2})/(\d{4})", t):
        return date(int(m[3]), int(m[2]), int(m[1]))
    if (m := re.search(r"(\d{1,2})-([a-z]{3})-(\d{4})", t)) and m[2] in MESES:
        return date(int(m[3]), MESES[m[2]], int(m[1]))
    return None


def leer_cajas(path):
    global _ocr
    _ocr = _ocr or RapidOCR()
    res, _ = _ocr(str(path))
    cajas = [(b[0][0], b[0][1], txt) for b, txt, _conf in (res or [])]
    return sorted(cajas, key=lambda c: (round(c[1] / 15), c[0]))  # orden de lectura aproximado


def valor_de_etiqueta(cajas, patron):
    """Busca la etiqueta y toma el monto de la misma caja o de la caja a su derecha en la misma fila."""
    for x, y, txt in cajas:
        norm = _sin_tildes(txt.lower()).replace(" ", "")
        if re.search(patron, norm):
            if m := re.search(DINERO, txt):
                return parse_dinero(m[0])
            vecinos = [(cx, t) for cx, cy, t in cajas if abs(cy - y) < 20 and cx > x and re.search(DINERO, t)]
            if vecinos:
                return parse_dinero(re.search(DINERO, min(vecinos)[1])[0])
    return None


def extraer_factura_ocr(path):
    cajas = leer_cajas(path)
    texto = "\n".join(t for _, _, t in cajas)
    plano = _sin_tildes(texto).replace(" ", "")
    out = {"emisor_razon_social": next((t for _, _, t in cajas if not re.search(r"\d{6}", t)), None)}

    rucs = re.findall(r"(?<!\d)((?:10|15|16|17|20)\d{9})(?!\d)", plano)  # el 1.º es el emisor (recuadro)
    out["emisor_ruc"] = rucs[0] if rucs else None
    out["cliente_ruc"] = rucs[1] if len(rucs) > 1 else None
    if m := re.search(r"([FE][A-Z0-9]{3})-(\d{1,8})(?!\d)", plano):
        out["serie"], out["correlativo"] = m[1], m[2]
    if m := re.search(r"(?:Senor\(es\)|Cliente|CLIENTE):\s*(.+)", _sin_tildes(texto)):
        out["cliente_razon_social"] = m[1].strip()
    out["moneda"] = "USD" if re.search(r"US\$|DOLARES|USD", plano.upper()) else "PEN"
    out["fecha_emision"] = next((f for line in texto.splitlines() if (f := parse_fecha(line))), None)
    out["op_gravada"] = valor_de_etiqueta(cajas, r"op\.?gravada|valordeventa|subtotal")
    out["igv"] = valor_de_etiqueta(cajas, r"^i\.?g\.?v")
    out["importe_total"] = valor_de_etiqueta(cajas, r"importetotal|totalapagar|^total")
    return out
