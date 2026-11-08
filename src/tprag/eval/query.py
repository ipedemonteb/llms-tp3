"""Texto de la consulta de cada target (paso 7, docs/agents/evaluacion.md §2.1).

La consulta no se guarda en el dataset: se arma al evaluar a partir del `code`. Una por target; si el módulo se importa
en más de una línea (`import mock` y `from mock import patch`, o `from PyPDF2.errors import ...`), se juntan las
líneas de todos esos imports.

- **A (base):** las líneas de A2 (`context_lines`) en orden, sin sangría, sin limpieza ni encabezado, sin tope.

Variantes previstas para la ablación (no implementadas): A1 (`usages`), B (sin comentarios ni strings), C (encabezado
en lenguaje natural), A3 (la función entera) y `task`.
"""

from collections.abc import Callable

from tprag.validate.imports import ImportRef, extract_imports


def target_refs(code: str, module: str) -> list[ImportRef]:
    """Imports del módulo del target: el mismo módulo o un submódulo (`PyPDF2` → `PyPDF2.errors`)."""
    refs = [r for r in extract_imports(code) if r.module == module or r.module.startswith(module + ".")]
    if not refs:
        raise ValueError(f"el código no importa {module}")
    return refs


def query_a(code: str, module: str) -> str:
    lines = code.splitlines()
    numbers = sorted({n for r in target_refs(code, module) for n in r.context_lines})
    return "\n".join(s for n in numbers if (s := lines[n - 1].strip()))


QUERIES: dict[str, Callable[[str, str], str]] = {"A": query_a}
