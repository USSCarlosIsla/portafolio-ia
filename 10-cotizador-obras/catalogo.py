"""Catálogo de insumos y Análisis de Precios Unitarios (APU) de partidas típicas de edificación.

IMPORTANTE: precios y rendimientos son REFERENCIALES y aproximados (orden de magnitud Lima, 2026),
pensados para el portafolio. En un uso real se reemplazan por la base de datos de la empresa
(por ejemplo, exportada de S10) o por los precios de la revista Costos / CAPECO.
"""
from dataclasses import dataclass, field
from typing import Dict

# ---------------- Configuración regional ----------------
MONEDA = "PEN"
IGV = 0.18                  # impuesto general a las ventas (Perú)
GASTOS_GENERALES = 0.10     # % del costo directo
UTILIDAD = 0.08             # % del costo directo
HERRAMIENTAS_PCT_MO = 0.03  # herramientas manuales = 3% de la mano de obra
HORAS_JORNADA = 8


@dataclass(frozen=True)
class Insumo:
    codigo: str
    descripcion: str
    unidad: str
    tipo: str      # "MO" mano de obra, "MAT" material, "EQ" equipo
    precio: float  # precio unitario sin IGV


INSUMOS: Dict[str, Insumo] = {i.codigo: i for i in [
    Insumo("MO-OPE", "Operario", "hh", "MO", 28.50),
    Insumo("MO-OFI", "Oficial", "hh", "MO", 22.60),
    Insumo("MO-PEO", "Peón", "hh", "MO", 20.40),
    Insumo("MAT-CEM", "Cemento Portland tipo I (42.5 kg)", "bls", "MAT", 30.50),
    Insumo("MAT-ACE", "Acero corrugado fy=4200 kg/cm²", "kg", "MAT", 4.25),
    Insumo("MAT-ARG", "Arena gruesa", "m3", "MAT", 62.00),
    Insumo("MAT-ARF", "Arena fina", "m3", "MAT", 55.00),
    Insumo("MAT-PIE", "Piedra chancada 1/2\"", "m3", "MAT", 78.00),
    Insumo("MAT-PGR", "Piedra grande de 8\"", "m3", "MAT", 60.00),
    Insumo("MAT-HOR", "Hormigón", "m3", "MAT", 58.00),
    Insumo("MAT-AGU", "Agua puesta en obra", "m3", "MAT", 6.00),
    Insumo("MAT-LKK", "Ladrillo King Kong 18 huecos", "und", "MAT", 1.15),
    Insumo("MAT-MAD", "Madera tornillo", "p2", "MAT", 6.80),
    Insumo("MAT-ALA", "Alambre negro recocido #16", "kg", "MAT", 6.20),
    Insumo("MAT-CLA", "Clavos para madera c/c 3\"", "kg", "MAT", 6.00),
    Insumo("MAT-PIN", "Pintura látex", "gal", "MAT", 48.00),
    Insumo("MAT-IMP", "Imprimante para muros", "gal", "MAT", 22.00),
    Insumo("EQ-MEZ", "Mezcladora de concreto 9-11 p3", "hm", "EQ", 22.00),
    Insumo("EQ-VIB", "Vibrador de concreto 4 HP", "hm", "EQ", 12.50),
]}


@dataclass(frozen=True)
class Partida:
    codigo: str
    descripcion: str
    unidad: str
    rendimiento: float                 # unidades por jornada de la cuadrilla
    cuadrilla: Dict[str, float]        # insumo MO o EQ -> número de personas/equipos
    materiales: Dict[str, float] = field(default_factory=dict)  # insumo MAT -> cantidad por unidad


PARTIDAS: Dict[str, Partida] = {p.codigo: p for p in [
    Partida("P01", "Excavación manual de zanjas en terreno normal", "m3", 4.0, {"MO-PEO": 1}),
    Partida("P02", "Relleno compactado con material propio", "m3", 8.0, {"MO-PEO": 2},
            {"MAT-AGU": 0.10}),
    Partida("P03", "Concreto ciclópeo cimientos corridos 1:10 + 30% PG", "m3", 25.0,
            {"MO-OPE": 2, "MO-OFI": 2, "MO-PEO": 8, "EQ-MEZ": 1},
            {"MAT-CEM": 2.90, "MAT-HOR": 0.83, "MAT-PGR": 0.48, "MAT-AGU": 0.18}),
    Partida("P04", "Falso piso de concreto 1:8 e=4\"", "m2", 120.0,
            {"MO-OPE": 2, "MO-OFI": 1, "MO-PEO": 8, "EQ-MEZ": 1},
            {"MAT-CEM": 0.43, "MAT-HOR": 0.11, "MAT-AGU": 0.02}),
    Partida("P05", "Concreto f'c=210 kg/cm² en columnas", "m3", 12.0,
            {"MO-OPE": 2, "MO-OFI": 1, "MO-PEO": 8, "EQ-MEZ": 1, "EQ-VIB": 1},
            {"MAT-CEM": 9.73, "MAT-ARG": 0.52, "MAT-PIE": 0.53, "MAT-AGU": 0.18}),
    Partida("P06", "Concreto f'c=210 kg/cm² en vigas", "m3", 14.0,
            {"MO-OPE": 2, "MO-OFI": 1, "MO-PEO": 8, "EQ-MEZ": 1, "EQ-VIB": 1},
            {"MAT-CEM": 9.73, "MAT-ARG": 0.52, "MAT-PIE": 0.53, "MAT-AGU": 0.18}),
    Partida("P07", "Concreto f'c=210 kg/cm² en losa aligerada", "m3", 20.0,
            {"MO-OPE": 2, "MO-OFI": 2, "MO-PEO": 8, "EQ-MEZ": 1, "EQ-VIB": 1},
            {"MAT-CEM": 9.73, "MAT-ARG": 0.52, "MAT-PIE": 0.53, "MAT-AGU": 0.18}),
    Partida("P08", "Encofrado y desencofrado de columnas", "m2", 12.0, {"MO-OPE": 1, "MO-OFI": 1},
            {"MAT-ALA": 0.30, "MAT-CLA": 0.17, "MAT-MAD": 4.50}),
    Partida("P09", "Encofrado y desencofrado de vigas", "m2", 10.0, {"MO-OPE": 1, "MO-OFI": 1},
            {"MAT-ALA": 0.30, "MAT-CLA": 0.20, "MAT-MAD": 5.20}),
    Partida("P10", "Acero de refuerzo fy=4200 kg/cm²", "kg", 250.0, {"MO-OPE": 1, "MO-OFI": 1},
            {"MAT-ACE": 1.07, "MAT-ALA": 0.03}),
    Partida("P11", "Muro de ladrillo King Kong de cabeza, mezcla 1:5", "m2", 6.5,
            {"MO-OPE": 1, "MO-PEO": 0.5}, {"MAT-LKK": 72, "MAT-CEM": 0.33, "MAT-ARG": 0.06, "MAT-AGU": 0.01}),
    Partida("P12", "Muro de ladrillo King Kong de soga, mezcla 1:5", "m2", 10.0,
            {"MO-OPE": 1, "MO-PEO": 0.5}, {"MAT-LKK": 39, "MAT-CEM": 0.18, "MAT-ARG": 0.03, "MAT-AGU": 0.01}),
    Partida("P13", "Tarrajeo de muros interiores 1:5 e=1.5 cm", "m2", 15.0, {"MO-OPE": 1, "MO-PEO": 0.5},
            {"MAT-CEM": 0.13, "MAT-ARF": 0.02, "MAT-AGU": 0.01}),
    Partida("P14", "Contrapiso de 48 mm", "m2", 100.0, {"MO-OPE": 2, "MO-OFI": 1, "MO-PEO": 4},
            {"MAT-CEM": 0.36, "MAT-ARG": 0.04, "MAT-AGU": 0.01}),
    Partida("P15", "Pintura látex en muros, 2 manos con imprimante", "m2", 30.0, {"MO-OPE": 1, "MO-PEO": 0.25},
            {"MAT-PIN": 0.06, "MAT-IMP": 0.05}),
]}


def catalogo_texto() -> str:
    """Catálogo legible para el LLM (código, descripción, unidad)."""
    return "\n".join(f"{p.codigo} | {p.descripcion} | {p.unidad}" for p in PARTIDAS.values())


# ---------------- Catálogos propios (la base de APU de tu empresa) ----------------
@dataclass
class Catalogo:
    insumos: Dict[str, Insumo]
    partidas: Dict[str, Partida]

    @classmethod
    def ejemplo(cls) -> "Catalogo":
        return cls(dict(INSUMOS), dict(PARTIDAS))

    def a_excel(self, ruta):
        """Exporta el catálogo con el formato de plantilla (sirve como ejemplo para llenar el tuyo)."""
        import pandas as pd

        ins = pd.DataFrame([vars(i) for i in self.insumos.values()])
        filas = []
        for p in self.partidas.values():
            for codigo, n in {**p.cuadrilla, **p.materiales}.items():
                es_cuadrilla = codigo in p.cuadrilla
                filas.append({"partida": p.codigo, "descripcion": p.descripcion, "unidad": p.unidad,
                              "rendimiento": p.rendimiento, "insumo": codigo,
                              "cuadrilla": n if es_cuadrilla else None, "cantidad": None if es_cuadrilla else n})
        with pd.ExcelWriter(ruta) as w:
            ins.to_excel(w, sheet_name="insumos", index=False)
            pd.DataFrame(filas).to_excel(w, sheet_name="apu", index=False)

    @classmethod
    def desde_excel(cls, ruta) -> "Catalogo":
        """Lee un Excel con hojas `insumos` (codigo, descripcion, unidad, tipo, precio) y `apu`
        (partida, descripcion, unidad, rendimiento, insumo, cuadrilla, cantidad): una fila por insumo.
        Para MO/EQ se llena `cuadrilla`; para materiales, `cantidad` por unidad de partida."""
        import pandas as pd

        hojas = pd.read_excel(ruta, sheet_name=["insumos", "apu"], dtype={"codigo": str, "partida": str,
                                                                         "insumo": str})
        insumos = {str(r.codigo): Insumo(str(r.codigo), str(r.descripcion), str(r.unidad), str(r.tipo).upper(),
                                         float(r.precio)) for r in hojas["insumos"].itertuples()}
        errores, partidas = [], {}
        for cod, g in hojas["apu"].groupby("partida", sort=False):
            f = g.iloc[0]
            cuadrilla, materiales = {}, {}
            for r in g.itertuples():
                if r.insumo not in insumos:
                    errores.append(f"Partida {cod}: insumo '{r.insumo}' no está en la hoja insumos.")
                elif insumos[r.insumo].tipo in ("MO", "EQ"):
                    cuadrilla[r.insumo] = float(r.cuadrilla)
                else:
                    materiales[r.insumo] = float(r.cantidad)
            if f.rendimiento <= 0 and cuadrilla:
                errores.append(f"Partida {cod}: rendimiento debe ser > 0.")
            partidas[str(cod)] = Partida(str(cod), str(f.descripcion), str(f.unidad), float(f.rendimiento),
                                         cuadrilla, materiales)
        if errores:
            raise ValueError("Catálogo inválido:\n" + "\n".join(errores[:50]))
        return cls(insumos, partidas)
