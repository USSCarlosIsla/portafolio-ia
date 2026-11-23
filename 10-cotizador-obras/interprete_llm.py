"""Intérprete con LLM: pedido en lenguaje natural → partidas del catálogo + metrados.

Ejemplo de entrada (como lo escribiría un cliente por WhatsApp):
  "Quiero levantar un cerco de ladrillo de cabeza de 25 m de largo y 2.4 m de alto, con su cimiento
   corrido de 0.6 de ancho por 0.8 de profundidad, tarrajeado y pintado por un lado."

División de responsabilidades (la clave del diseño):
  * Claude INTERPRETA: elige partidas del catálogo y escribe el cálculo del metrado como expresión
    aritmética ("25*2.4"), con sus supuestos y las preguntas que habría que hacerle al cliente.
  * El CÓDIGO CALCULA: evalúa la expresión de forma segura (sin eval()), valida códigos y unidades, y
    el motor de presupuesto pone los precios. El LLM nunca inventa un precio.
"""
import ast
import operator
from pathlib import Path
from typing import List

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field

import catalogo as C

load_dotenv(Path(__file__).resolve().parent / ".env")
MODEL = "claude-opus-5-5"


class PartidaInterpretada(BaseModel):
    codigo: str = Field(description="Código de partida del catálogo, p. ej. P11")
    fragmento: str = Field(description="Parte del pedido del cliente que origina esta partida")
    calculo: str = Field(description="Expresión aritmética del metrado usando solo números y + - * / ( )")
    unidad: str = Field(description="Unidad de la partida según el catálogo")
    supuestos: str = Field(description="Supuestos tomados para el metrado (vacío si no hubo)")


class Interpretacion(BaseModel):
    partidas: List[PartidaInterpretada]
    no_cubierto: List[str] = Field(description="Trabajos pedidos que NO existen en el catálogo")
    preguntas: List[str] = Field(description="Datos faltantes que habría que confirmar con el cliente")


SYSTEM = f"""Eres un ingeniero de costos peruano que prepara metrados para un presupuesto de obra.
Convierte el pedido del cliente en partidas del CATÁLOGO (usa solo estos códigos y sus unidades):

{C.catalogo_texto()}

Reglas:
- `calculo` es una expresión aritmética con números y + - * / ( ) — por ejemplo "25*0.6*0.8".
  No escribas el resultado, solo la expresión: el sistema la evaluará.
- Incluye las partidas implícitas típicas (p. ej., un cimiento corrido implica excavación; un muro
  tarrajeado y pintado implica tarrajeo y pintura sobre el área indicada).
- Si falta un dato, asume un valor estándar razonable, explícalo en `supuestos` y agrega la pregunta
  correspondiente en `preguntas`.
- Lo que no tenga partida en el catálogo va en `no_cubierto`; no lo fuerces en otra partida."""

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def evaluar_expresion(expr: str) -> float:
    """Evalúa aritmética básica de forma segura: rechaza nombres, llamadas y cualquier otra cosa."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError(f"Expresión no permitida: {expr!r}")
    return ev(ast.parse(expr.replace(",", "."), mode="eval").body)


def validar(interp: Interpretacion):
    """Convierte la interpretación en items (código, metrado) y lista de observaciones."""
    items, obs = [], []
    for p in interp.partidas:
        if p.codigo not in C.PARTIDAS:
            obs.append(f"Código inexistente {p.codigo} ('{p.fragmento}'): se descarta.")
            continue
        if p.unidad != C.PARTIDAS[p.codigo].unidad:
            obs.append(f"{p.codigo}: unidad '{p.unidad}' no coincide con '{C.PARTIDAS[p.codigo].unidad}'.")
        try:
            metrado = round(evaluar_expresion(p.calculo), 2)
        except (ValueError, SyntaxError, ZeroDivisionError) as e:
            obs.append(f"{p.codigo}: no se pudo evaluar '{p.calculo}' ({e}).")
            continue
        if metrado <= 0:
            obs.append(f"{p.codigo}: metrado no positivo ({metrado}).")
            continue
        items.append((p.codigo, metrado))
    return items, obs


def interpretar(pedido: str, client: anthropic.Anthropic = None) -> Interpretacion:
    client = client or anthropic.Anthropic()
    resp = client.beta.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": pedido}],
        output_format=Interpretacion,
        output_config={"effort": "medium"},  # hay razonamiento geométrico: un poco más que en extracción
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if resp.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"El modelo no completó la respuesta (stop_reason={resp.stop_reason}).")
    return resp.parsed_output


if __name__ == "__main__":
    import sys

    from cotizador import cotizar

    pedido = " ".join(sys.argv[1:]) or (
        "Quiero levantar un cerco de ladrillo de cabeza de 25 m de largo y 2.4 m de alto, con su cimiento "
        "corrido de 0.6 de ancho por 0.8 de profundidad, tarrajeado y pintado por un lado. También quiero "
        "una puerta metálica de 3 m.")
    interp = interpretar(pedido)
    items, obs = validar(interp)
    for p in interp.partidas:
        print(f"{p.codigo}  {p.calculo:<22} {p.unidad:<3} | {p.fragmento} | {p.supuestos}")
    print("No cubierto:", interp.no_cubierto)
    print("Preguntas para el cliente:", interp.preguntas)
    print("Observaciones:", obs)
    print(f"TOTAL: S/ {cotizar(items)['total']:,.2f}")
