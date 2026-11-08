"""Extracción de imports con `ast` (validador.md §1, pasos 1 y 6).

Un `ImportRef` por módulo importado: `import a, b` da dos. Se incluyen los imports dentro de funciones o `try`; los
relativos (`from . import x`) se ignoran porque son del propio proyecto. Un script con error de sintaxis levanta
`SyntaxError` (lo maneja quien llama).

Dos formas de juntar las líneas de uso (dataset_eval.md §2.2, §3.1):
- `usages` (A1): líneas donde aparecen los nombres del import.
- `context_lines` (A2, consulta base del paso 7): además sigue los nombres derivados hasta `MAX_DEPTH` saltos
  (`reader = PdfReader(p)` → `for page in reader.pages`) y toma la sentencia entera (las de varias líneas también; de
  un `for`/`with`/`if`/`def` solo la parte donde aparece el nombre, no el cuerpo). Incluye la línea del import. El
  rastro es por nombre en todo el script, no por función, y no sigue valores de retorno de funciones propias.
"""

import ast
from collections import defaultdict
from dataclasses import dataclass, field

MAX_DEPTH = 2


@dataclass
class ImportRef:
    module: str  # `google.oauth2.service_account`; en `from x import y`, x
    line: int
    names: list[str]  # nombres que el import deja definidos en el script: `np`, `PdfReader`, `google`
    usages: list[int] = field(default_factory=list)  # líneas donde se usan esos nombres (sin la del import)
    context_lines: list[int] = field(default_factory=list)  # A2: import + usos siguiendo las asignaciones


def _bound_name(node: ast.Import | ast.ImportFrom, alias: ast.alias) -> str:
    if alias.asname:
        return alias.asname
    if isinstance(node, ast.Import):
        return alias.name.split(".")[0]  # `import a.b` deja definido `a`
    return alias.name  # `from x import y` → `y` (o `*`)


def _target_names(target: ast.AST) -> list[str]:
    """Nombres que define una asignación: `a`, `a, b`, `[a, *b]`; no `self.x` ni `d[k]`."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for elt in target.elts for n in _target_names(elt)]
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return []


def _bindings(tree: ast.AST) -> list[tuple[list[str], set[str]]]:
    """(nombres que se definen, nombres que aparecen en el valor) de cada asignación, `for`, `with`, comprensión y `:=`."""
    pairs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            pairs += [(t, node.value) for t in node.targets]
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)) and node.value is not None:
            pairs.append((node.target, node.value))
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            pairs.append((node.target, node.iter))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            pairs += [(i.optional_vars, i.context_expr) for i in node.items if i.optional_vars is not None]
    return [(_target_names(t), {n.id for n in ast.walk(v) if isinstance(n, ast.Name)}) for t, v in pairs]


def _traced(names: list[str], bindings: list[tuple[list[str], set[str]]]) -> set[str]:
    """Nombres del import (nivel 0) y los que se asignan a partir de ellos, hasta MAX_DEPTH."""
    depth = {n: 0 for n in names}
    changed = True
    while changed:
        changed = False
        for targets, used in bindings:
            levels = [depth[u] for u in used if u in depth]
            if not levels or min(levels) + 1 > MAX_DEPTH:
                continue
            for t in targets:
                if depth.get(t, MAX_DEPTH + 1) > min(levels) + 1:
                    depth[t] = min(levels) + 1
                    changed = True
    return set(depth)


def _span(node: ast.AST) -> range:
    return range(node.lineno, (node.end_lineno or node.lineno) + 1)


def _name_spans(tree: ast.AST) -> dict[str, set[int]]:
    """nombre → líneas de las sentencias donde aparece. Sentencia simple: entera. Sentencia compuesta (`for`, `with`,
    `if`, `def`, ...): solo la expresión donde aparece (el iterable, el decorador, la condición), no el cuerpo."""
    parent = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    out = defaultdict(set)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name):
            continue
        child = node
        while not isinstance(parent.get(child), (ast.stmt, ast.ExceptHandler)) and child in parent:
            child = parent[child]
        if isinstance(parent.get(child), ast.ExceptHandler):  # `except X as e:`: la línea del except, no el cuerpo
            out[node.id].update(_span(child))
            continue
        stmt = parent.get(child)
        if stmt is None:
            continue
        compound = any(isinstance(getattr(stmt, f, None), list) and getattr(stmt, f) and
                       isinstance(getattr(stmt, f)[0], ast.stmt) for f in ("body", "orelse", "finalbody", "handlers"))
        if not compound:
            out[node.id].update(_span(stmt))
        else:  # `withitem` no tiene posición: se usa la del nombre
            out[node.id].update(_span(child if hasattr(child, "lineno") else node))
    return out


def extract_imports(source: str) -> list[ImportRef]:
    tree = ast.parse(source)
    uses = defaultdict(set)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            uses[node.id].add(node.lineno)

    refs, import_spans = [], {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                refs.append(ImportRef(alias.name, node.lineno, [_bound_name(node, alias)]))
                import_spans[id(refs[-1])] = _span(node)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            refs.append(ImportRef(node.module, node.lineno, [_bound_name(node, a) for a in node.names]))
            import_spans[id(refs[-1])] = _span(node)

    bindings, spans = _bindings(tree), _name_spans(tree)
    for ref in refs:
        ref.usages = sorted({line for n in ref.names for line in uses.get(n, ()) if line != ref.line})
        traced = _traced(ref.names, bindings)
        ref.context_lines = sorted(set(import_spans[id(ref)]) | {line for n in traced for line in spans.get(n, ())})
    return sorted(refs, key=lambda r: r.line)  # estable: `import a, b` queda en el orden del código
