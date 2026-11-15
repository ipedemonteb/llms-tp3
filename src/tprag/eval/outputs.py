"""Salidas de las evaluaciones: un resultado ya corrido no se pisa sin querer.

Los nombres salen de la configuración de la corrida, así que repetirla con otra versión del evaluador (o del código)
escribiría encima de la anterior. Por eso `eval_generation.py` y `analyze_generation.py` se niegan a escribir si la
salida existe: una variante nueva va con `--tag <nombre>` (sufijo `__<nombre>`, como las corridas `__v1`, `__v2`,
`__v3`) y queda al lado; `--force` reemplaza a propósito.
"""

import argparse
import re
from pathlib import Path

_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def tag_arg(value: str) -> str:
    """Tipo de argparse para --tag: letras, números, punto, guion y guion bajo (va en el nombre de los archivos)."""
    if not _TAG.fullmatch(value):
        raise argparse.ArgumentTypeError(f"tag inválido: {value!r} (letras, números, '.', '-' o '_')")
    return value


def tagged(stem: str, tag: str | None) -> str:
    return f"{stem}__{tag}" if tag else stem


def refuse_overwrite(paths: list[Path], force: bool) -> str | None:
    """Mensaje si alguna salida ya existe y no se pidió --force; None si se puede escribir."""
    existing = [p for p in paths if p.exists()]
    if not existing or force:
        return None
    return (f"ya existe {', '.join(p.name for p in existing)}: no se pisa. Usar --tag <nombre> para escribir la "
            "variante al lado, o --force para reemplazarla")
