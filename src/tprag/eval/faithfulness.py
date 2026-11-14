"""Faithfulness por heurística (metricas.md §3.2), sin API: ¿la API nueva que usa el script corregido sale de los chunks
que recibió el LLM, o de su memoria?

Claims = los identificadores de la API nueva en las líneas cambiadas: el módulo importado, los nombres importados y lo
que cuelga de ellos (atributos y argumentos por nombre), siguiendo las asignaciones como A2 en `validate/imports.py`
(`reader = PdfReader(f)` → `reader.pages`). Un claim es fiel si aparece como palabra completa en el texto de los
chunks. Se cuentan identificadores distintos.

Limitaciones (metricas.md §3.2): no ve si un nombre correcto se usa con argumentos equivocados; un import nuevo de la
biblioteca estándar que no aparece en el contexto se toma como auxiliar (`import json`) y no se cuenta; los métodos de
los tipos básicos (`split`, `append`...) se descartan aunque cuelguen de la API nueva.
"""

import ast
import builtins
import difflib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Métodos y atributos de los tipos básicos: un `text.split()` sobre lo que devolvió la API nueva no es un claim de esa API
_BASIC_ATTRS = {a for t in (str, bytes, list, dict, set, tuple, int, float, bool) for a in dir(t)}
_STOP = _BASIC_ATTRS | set(dir(builtins))


@dataclass(frozen=True)
class Claim:
    name: str
    line: int
    kind: str  # module | import | attr | kwarg


def changed_lines(original: str, fixed: str) -> set[int]:
    """Líneas (desde 1) del script corregido que no estaban en el original: insertadas o reemplazadas."""
    sm = difflib.SequenceMatcher(a=original.splitlines(), b=fixed.splitlines(), autojunk=False)
    return {j + 1 for tag, _, _, j1, j2 in sm.get_opcodes() if tag in ("replace", "insert") for j in range(j1, j2)}


def _import_keys(tree: ast.AST) -> set[tuple]:
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            keys |= {("import", a.name, a.asname) for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            keys |= {("from", node.module, a.name, a.asname) for a in node.names}
    return keys


def _root(node: ast.AST) -> str | None:
    """Nombre en la raíz de una cadena `a.b(c)[0].d`; None si no termina en un nombre."""
    while True:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred, ast.Await)):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        else:
            return None


def _target_names(target: ast.AST) -> list[str]:
    return [n.id for n in ast.walk(target) if isinstance(n, ast.Name)]


def is_helper_import(module: str, context: str) -> bool:
    """Import nuevo de la biblioteca estándar que no aparece en el contexto: auxiliar, no el reemplazo."""
    return module.split(".")[0] in sys.stdlib_module_names and not mentions(context, module.split(".")[0])


def claims(original: str, fixed: str, context: str = "") -> list[Claim]:
    """Claims de la API nueva en las líneas cambiadas de `fixed`. `context` solo se usa para descartar imports
    auxiliares de la biblioteca estándar."""
    tree = ast.parse(fixed)
    old_keys = _import_keys(ast.parse(original))
    changed = changed_lines(original, fixed)
    out: list[Claim] = []
    bound: set[str] = set()  # nombres que vienen de un import nuevo (o derivados de ellos)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if ("import", a.name, a.asname) in old_keys or is_helper_import(a.name, context):
                    continue
                out.append(Claim(a.name, node.lineno, "module"))
                bound.add(a.asname or a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            new = [a for a in node.names if ("from", node.module, a.name, a.asname) not in old_keys]
            if not new or is_helper_import(node.module, context):
                continue
            out.append(Claim(node.module, node.lineno, "module"))
            for a in new:
                if a.name != "*":
                    out.append(Claim(a.name, node.lineno, "import"))
                    bound.add(a.asname or a.name)

    # Asignaciones: `reader = PdfReader(f)`, `for page in reader.pages`, `with X() as y` (hasta que no cambie)
    for _ in range(10):
        before = len(bound)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and _root(node.value) in bound:
                bound.update(n for t in node.targets for n in _target_names(t))
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value and _root(node.value) in bound:
                bound.update(_target_names(node.target))
            elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)) and _root(node.iter) in bound:
                bound.update(_target_names(node.target))
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for item in node.items:
                    if item.optional_vars is not None and _root(item.context_expr) in bound:
                        bound.update(_target_names(item.optional_vars))
            elif isinstance(node, ast.NamedExpr) and _root(node.value) in bound:
                bound.add(node.target.id)
        if len(bound) == before:
            break

    for node in ast.walk(tree):
        line = getattr(node, "lineno", None)
        if line not in changed:
            continue
        if isinstance(node, ast.Attribute) and _root(node.value) in bound and node.attr not in _STOP \
                and not node.attr.startswith("__"):
            out.append(Claim(node.attr, line, "attr"))
        elif isinstance(node, ast.Call) and _root(node.func) in bound:
            out.extend(Claim(k.arg, line, "kwarg") for k in node.keywords if k.arg and k.arg not in _STOP)
    return out


def mentions(context: str, name: str) -> bool:
    """¿`name` aparece como palabra completa en el contexto? (`PdfReader` no matchea con `PdfReaderX`)."""
    return re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", context) is not None


def faithfulness(original: str, fixed: str, context: str) -> dict:
    """Fracción de identificadores distintos de la API nueva que aparecen en el contexto. `score` None: no hay claims
    (no hubo cambio, o el cambio no usa una API nueva)."""
    if fixed == original:
        return {"claims": 0, "faithful": 0, "score": None, "unfaithful": []}
    try:
        names = sorted({c.name for c in claims(original, fixed, context)})
    except SyntaxError:
        return {"claims": 0, "faithful": 0, "score": None, "unfaithful": [], "error": "no parsea"}
    unfaithful = [n for n in names if not mentions(context, n)]
    n = len(names)
    return {"claims": n, "faithful": n - len(unfaithful), "score": (n - len(unfaithful)) / n if n else None,
            "unfaithful": unfaithful}


def load_chunk_texts(path: Path, ids: set[str]) -> dict[str, str]:
    """Texto de los chunks por id desde `data/processed/<modo>/<splitter>/chunks.jsonl` (para corridas guardadas sin el
    texto: opción C de metricas.md §3.2)."""
    out = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if '"id"' in line:
                c = json.loads(line)
                if c["id"] in ids:
                    out[c["id"]] = c["text"]
    return out


def record_context(record: dict, texts: dict[str, str] | None = None) -> str:
    """Texto de todos los chunks que recibió el LLM en ese script (uno por import que pasó el umbral). Usa el `text`
    guardado en el detalle; si falta, lo busca en `texts` (id → texto)."""
    parts = []
    for ctx in record.get("contexts", []):
        for ch in ctx["chunks"]:
            text = ch.get("text") or (texts or {}).get(ch["id"])
            if text is None:
                raise KeyError(f"{record['id']}: falta el texto del chunk {ch['id']} (pasar `texts`)")
            parts.append(text)
    return "\n\n".join(parts)
