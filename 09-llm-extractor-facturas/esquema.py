"""Esquema de datos de una factura + reglas de negocio para validar lo que extrae cualquier modelo.

El LLM rellena este esquema (salida estructurada); luego `validar_factura` revisa que los números
cuadren. Si no cuadran, el error se le devuelve al LLM para que se corrija (ver extractor_llm.py).
"""
from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field


class LineaFactura(BaseModel):
    descripcion: str
    cantidad: float
    precio_unitario: float
    total_linea: float


class Factura(BaseModel):
    numero_factura: str = Field(description="Número o código de la factura, tal cual aparece")
    fecha_emision: date = Field(description="Fecha de emisión en formato ISO AAAA-MM-DD")
    proveedor_nombre: str
    proveedor_nit: str = Field(description="NIT del proveedor SIN dígito de verificación, solo dígitos")
    proveedor_dv: Optional[str] = Field(default=None, description="Dígito de verificación del NIT, si aparece")
    cliente_nombre: str
    lineas: List[LineaFactura]
    subtotal: float
    iva: float = Field(description="Valor del IVA (impuesto), no el porcentaje")
    total: float
    moneda: str = Field(default="COP", description="Código ISO de la moneda")


# ---------------- Reglas de negocio ----------------
def digito_verificacion_nit(nit: str) -> str:
    """Algoritmo oficial de la DIAN (Colombia) para el dígito de verificación del NIT."""
    pesos = [3, 7, 13, 17, 19, 23, 29, 37, 41, 43, 47, 53, 59, 67, 71]
    digitos = [int(c) for c in reversed(nit)]
    s = sum(d * p for d, p in zip(digitos, pesos)) % 11
    return str(s if s in (0, 1) else 11 - s)


def validar_factura(f: Factura, tolerancia: float = 1.0) -> List[str]:
    """Devuelve una lista de errores en lenguaje natural (vacía = factura consistente)."""
    errores = []
    for i, l in enumerate(f.lineas, 1):
        if abs(l.cantidad * l.precio_unitario - l.total_linea) > tolerancia:
            errores.append(f"Línea {i}: cantidad × precio unitario ({l.cantidad * l.precio_unitario:,.2f}) "
                           f"no coincide con su total ({l.total_linea:,.2f}).")
    suma = sum(l.total_linea for l in f.lineas)
    if abs(suma - f.subtotal) > tolerancia:
        errores.append(f"La suma de las líneas ({suma:,.2f}) no coincide con el subtotal ({f.subtotal:,.2f}).")
    if abs(f.subtotal + f.iva - f.total) > tolerancia:
        errores.append(f"Subtotal + IVA ({f.subtotal + f.iva:,.2f}) no coincide con el total ({f.total:,.2f}).")
    if not f.proveedor_nit.isdigit():
        errores.append(f"El NIT '{f.proveedor_nit}' debe contener solo dígitos.")
    elif f.proveedor_dv is not None and digito_verificacion_nit(f.proveedor_nit) != f.proveedor_dv:
        errores.append(f"El dígito de verificación {f.proveedor_dv} no corresponde al NIT {f.proveedor_nit} "
                       f"(debería ser {digito_verificacion_nit(f.proveedor_nit)}). Revisa ambos valores.")
    return errores
