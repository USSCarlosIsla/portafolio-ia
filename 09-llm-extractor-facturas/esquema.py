"""Esquema de una factura electrónica peruana (SUNAT) + reglas de negocio para validar la extracción.

El LLM rellena este esquema (salida estructurada); luego `validar_factura` revisa RUC, serie, IGV y
sumas. Si algo no cuadra, los errores se le devuelven al LLM para que corrija (ver extractor_llm.py).
"""
import re
from datetime import date
from typing import List, Literal

from pydantic import BaseModel, Field

IGV = 0.18
PREFIJOS_RUC = ("10", "15", "16", "17", "20")  # 10 persona natural, 20 persona jurídica, etc.


class LineaFactura(BaseModel):
    descripcion: str
    cantidad: float
    valor_unitario: float = Field(description="Valor unitario SIN IGV")
    valor_venta: float = Field(description="Cantidad × valor unitario, SIN IGV")


class Factura(BaseModel):
    serie: str = Field(description="Serie del comprobante, p. ej. F001 o E001")
    correlativo: str = Field(description="Número correlativo tal cual aparece, sin la serie, p. ej. 00012345")
    fecha_emision: date = Field(description="Fecha de emisión en formato ISO AAAA-MM-DD")
    emisor_ruc: str = Field(description="RUC del emisor (proveedor): 11 dígitos")
    emisor_razon_social: str
    cliente_ruc: str = Field(description="RUC del cliente (adquiriente): 11 dígitos")
    cliente_razon_social: str
    moneda: Literal["PEN", "USD"] = Field(description="PEN si el monto está en soles (S/), USD si en dólares (US$)")
    lineas: List[LineaFactura]
    op_gravada: float = Field(description="Total de operaciones gravadas (sin IGV)")
    igv: float = Field(description="Monto del IGV en dinero, no el porcentaje")
    importe_total: float


# ---------------- Reglas de negocio ----------------
def digito_verificador_ruc(ruc10: str) -> str:
    """Algoritmo de SUNAT (módulo 11) sobre los 10 primeros dígitos del RUC."""
    pesos = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]
    resto = 11 - sum(int(d) * p for d, p in zip(ruc10, pesos)) % 11
    return str({10: 0, 11: 1}.get(resto, resto))


def ruc_valido(ruc: str) -> bool:
    return bool(re.fullmatch(r"\d{11}", ruc or "")) and ruc[:2] in PREFIJOS_RUC and \
        digito_verificador_ruc(ruc[:10]) == ruc[10]


def validar_factura(f: Factura, tolerancia: float = 0.05) -> List[str]:
    """Devuelve una lista de errores en lenguaje natural (vacía = factura consistente)."""
    errores = []
    if not re.fullmatch(r"[FE][A-Z0-9]{3}", f.serie):
        errores.append(f"La serie '{f.serie}' no tiene formato de factura (4 caracteres que empiezan con F o E).")
    if not re.fullmatch(r"\d{1,8}", f.correlativo):
        errores.append(f"El correlativo '{f.correlativo}' debe tener entre 1 y 8 dígitos.")
    for rol, ruc in [("emisor", f.emisor_ruc), ("cliente", f.cliente_ruc)]:
        if not ruc_valido(ruc):
            errores.append(f"El RUC del {rol} '{ruc}' no es válido (11 dígitos, prefijo 10/15/16/17/20 y "
                           f"dígito verificador SUNAT). Revisa cada dígito en la imagen.")
    for i, l in enumerate(f.lineas, 1):
        if abs(l.cantidad * l.valor_unitario - l.valor_venta) > tolerancia:
            errores.append(f"Línea {i}: cantidad × valor unitario ({l.cantidad * l.valor_unitario:,.2f}) "
                           f"no coincide con su valor de venta ({l.valor_venta:,.2f}).")
    suma = sum(l.valor_venta for l in f.lineas)
    if abs(suma - f.op_gravada) > tolerancia:
        errores.append(f"La suma de las líneas ({suma:,.2f}) no coincide con la op. gravada ({f.op_gravada:,.2f}).")
    if abs(f.op_gravada * IGV - f.igv) > max(tolerancia, 0.01 * f.igv):
        errores.append(f"El IGV ({f.igv:,.2f}) no es el 18% de la op. gravada ({f.op_gravada * IGV:,.2f}).")
    if abs(f.op_gravada + f.igv - f.importe_total) > tolerancia:
        errores.append(f"Op. gravada + IGV ({f.op_gravada + f.igv:,.2f}) no coincide con el importe total "
                       f"({f.importe_total:,.2f}).")
    return errores
