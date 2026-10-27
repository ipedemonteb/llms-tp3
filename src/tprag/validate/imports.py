"""Extracción de imports con `ast` (validador.md §1, pasos 1 y 6).

Un `ImportRef` por módulo importado: `import a, b` da dos. Se incluyen los imports dentro de funciones o `try`; los
relativos (`from . import x`) se ignoran porque son del propio proyecto. Un script con error de sintaxis levanta
`SyntaxError` (lo maneja quien llama).
"""

import ast
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ImportRef:
    module: str  # `google.oauth2.service_account`; en `from x import y`, x
    line: int
    names: list[str]  # nombres que el import deja definidos en el script: `np`, `PdfReader`, `google`
    usages: list[int] = field(default_factory=list)  # líneas donde se usan esos nombres (sin la del import)


def _bound_name(node: ast.Import | ast.ImportFrom, alias: ast.alias) -> str:
    if alias.asname:
        return alias.asname
    if isinstance(node, ast.Import):
        return alias.name.split(".")[0]  # `import a.b` deja definido `a`
    return alias.name  # `from x import y` → `y` (o `*`)


def extract_imports(source: str) -> list[ImportRef]:
    tree = ast.parse(source)
    uses = defaultdict(set)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            uses[node.id].add(node.lineno)

    refs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                refs.append(ImportRef(alias.name, node.lineno, [_bound_name(node, alias)]))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            refs.append(ImportRef(node.module, node.lineno, [_bound_name(node, a) for a in node.names]))
    for ref in refs:
        ref.usages = sorted({line for n in ref.names for line in uses.get(n, ()) if line != ref.line})
    return sorted(refs, key=lambda r: r.line)  # estable: `import a, b` queda en el orden del código
