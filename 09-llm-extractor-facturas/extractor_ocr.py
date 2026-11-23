"""Baseline SIN LLM: OCR clásico (RapidOCR, corre local y gratis) + reglas con expresiones regulares.

Sirve para responder la pregunta que hará cualquier entrevistador: "¿de verdad necesitas un LLM?".
Las reglas funcionan bien con un formato conocido, pero cada diseño, etiqueta o formato de número
nuevo exige escribir otra regla. No extrae las líneas de productos (haría falta reconstruir la tabla).
"""
import re
import unicodedata
from datetime import date

from rapidocr_onnxruntime import RapidOCR

_ocr = None
MESES = {m: i for i, m in enumerate(["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct",
                                     "nov", "dic"], 1)}
DINERO = r"\$\s?[\d.,]+\d"


def _sin_tildes(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def parse_dinero(s):
    s = re.sub(r"[^\d.,]", "", s)
    if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d{1,2})?|\d+,\d{1,2}", s):  # 1.234.567,00 (Colombia)
        return float(s.replace(".", "").replace(",", "."))
    return float(s.replace(",", ""))  # 1,234,567.00 o 1234567


def parse_fecha(t):
    t = _sin_tildes(t.lower())
    if m := re.search(r"(\d{4})-(\d{2})-(\d{2})", t):
        return date(int(m[1]), int(m[2]), int(m[3]))
    if m := re.search(r"(\d{2})/(\d{2})/(\d{4})", t):
        return date(int(m[3]), int(m[2]), int(m[1]))
    if m := re.search(r"(\d{1,2})\s*-?\s*(?:de)?\s*([a-z]{3})[a-z]*\s*-?\s*(?:de)?\s*(\d{4})", t):
        if m[2] in MESES:
            return date(int(m[3]), MESES[m[2]], int(m[1]))
    return None


def leer_cajas(path):
    global _ocr
    _ocr = _ocr or RapidOCR()
    res, _ = _ocr(str(path))
    return [(b[0][0], b[0][1], txt) for b, txt, _conf in (res or [])]


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
    out = {"proveedor_nombre": cajas[0][2] if cajas else None}

    if m := re.search(r"N\.?I\.?T\.?:?([\d.]+)-(\d)", plano):
        out["proveedor_nit"], out["proveedor_dv"] = m[1].replace(".", ""), m[2]
    if m := re.search(r"(?:No\.|N°|FACTURA)((?:FE|FV|F|SETT)-\d+)", plano):
        out["numero_factura"] = m[1]
    if m := re.search(r"(?:Cliente|Senores|CLIENTE):\s*(.+)", _sin_tildes(texto)):
        out["cliente_nombre"] = m[1].strip()
    out["fecha_emision"] = next((f for line in texto.splitlines() if (f := parse_fecha(line))), None)
    out["subtotal"] = valor_de_etiqueta(cajas, r"subtotal|valorbruto")
    out["iva"] = valor_de_etiqueta(cajas, r"^iva|impuesto")
    out["total"] = valor_de_etiqueta(cajas, r"totalapagar|valorneto|^total")
    return out
