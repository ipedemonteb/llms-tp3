"""Nombres específicos de un bloque gold, para el dataset sintético (plan_experimentos_indice.md §2, prompt synthetic_v2).

- `forbidden_names`: identificadores de la API real (atributos, llamadas, nombres importados, argumentos con nombre) y
  datos de ejemplo (strings cortos) del bloque. Es la lista NOMBRES PROHIBIDOS que reciben los agentes.
- `shared_names`: cuáles de esos nombres aparecen en una consulta. Con 3 o más, la "librería inventada" copia la real y
  la búsqueda queda regalada.

Se descartan builtins, palabras clave, módulos y métodos de la biblioteca estándar más comunes y nombres de menos de 3
caracteres: no identifican a ninguna librería.
"""

import ast
import builtins
import keyword
import re
import sys

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_GENERIC = (
    set(dir(builtins)) | set(keyword.kwlist) | set(sys.stdlib_module_names)
    | {n for t in (str, list, dict, set, tuple, bytes, int, float) for n in dir(t)}
    | {"self", "cls", "args", "kwargs", "main", "data", "path", "name", "value", "values", "result", "results",
       "file", "files", "text", "key", "keys", "item", "items", "line", "lines", "func", "obj", "output", "input",
       "size", "kind", "mode", "type", "config", "options", "default", "params", "info", "debug", "error",
       "warning", "logger", "log", "run", "start", "stop", "close", "read", "write", "load", "save", "dump",
       "dumps", "loads", "get", "put", "post", "delete", "add", "remove", "create", "init", "new", "test", "tests",
       "utf", "utf-8", "true", "false", "none", "rb", "wb", "w", "r", "a", "x", "y", "i", "j", "k", "n", "df"}
)
MAX_STRING = 30  # strings más largos suelen ser frases o rutas, no un dato de ejemplo


def _keep(name: str) -> bool:
    # `Config` o `Document` con mayúscula son clases de la API, no la palabra genérica
    if name.startswith("__") and name.endswith("__"):  # `__main__`, `__init__`: Python, no la librería
        return False
    return len(name) >= 3 and name not in _GENERIC and (name[0].isupper() or name.lower() not in _GENERIC)


def _names_ast(tree: ast.AST) -> set[str]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            out.add(node.func.id)
        elif isinstance(node, ast.keyword) and node.arg:
            out.add(node.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                out.update(alias.name.split("."))
                if alias.asname:
                    out.add(alias.asname)
            if isinstance(node, ast.ImportFrom) and node.module:
                out.update(node.module.split("."))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value.strip()
            if s and len(s) <= MAX_STRING and "\n" not in s:
                out.add(s)
    return out


def _names_regex(code: str) -> set[str]:
    """Para bloques de la doc que no parsean (`...`, fragmentos): `.attr`, `nombre(` y `arg=`."""
    out = set(re.findall(r"\.([A-Za-z_]\w*)", code)) | set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", code))
    out |= set(re.findall(r"[(,]\s*([A-Za-z_]\w*)\s*=(?!=)", code))
    out |= {m for line in re.findall(r"^\s*(?:from|import)\s+(.+)$", code, re.M) for m in _IDENT.findall(line)}
    out |= {s for s in re.findall(r"[\"']([^\"'\n]{1,%d})[\"']" % MAX_STRING, code)}
    return out


def forbidden_names(code: str) -> list[str]:
    try:
        names = _names_ast(ast.parse(code))
    except SyntaxError:
        names = _names_regex(code)
    return sorted(n for n in names if _keep(n))


def _stdlib_api() -> set[str]:
    """Nombres públicos de los módulos de la stdlib que más usan los scripts (`uuid4`, `mkdir`, `perf_counter`...)."""
    import importlib
    names = set()
    for mod in ("os", "os.path", "sys", "time", "datetime", "pathlib", "json", "logging", "re", "uuid", "random",
                "math", "collections", "itertools", "functools", "typing", "dataclasses", "argparse", "csv",
                "sqlite3", "subprocess", "shutil", "tempfile", "hashlib", "base64", "asyncio", "contextlib", "io"):
        m = importlib.import_module(mod)
        names |= {n for n in dir(m) if not n.startswith("_")}
        for attr in ("Path", "datetime", "date", "Logger", "Connection", "Cursor"):
            if hasattr(m, attr) and isinstance(getattr(m, attr), type):
                names |= {n for n in dir(getattr(m, attr)) if not n.startswith("_")}
    return names


_STDLIB_API = _stdlib_api()


def _usage_lines(query: str, module: str) -> str:
    """El import del módulo inventado y las líneas que usan el módulo o algún nombre importado de él."""
    roots = {module.split(".")[0]}
    for m in re.finditer(rf"^\s*from\s+{re.escape(module.split('.')[0])}[\w.]*\s+import\s+\(?([^)\n]+)", query, re.M):
        roots |= {p.split(" as ")[-1].strip() for p in m.group(1).split(",") if p.strip()}
    for m in re.finditer(rf"^\s*import\s+{re.escape(module.split('.')[0])}[\w.]*\s+as\s+(\w+)", query, re.M):
        roots.add(m.group(1))
    pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(roots))) + r")\b")
    return "\n".join(line for line in query.split("\n") if pattern.search(line))


def shared_names(query: str, gold: str, module: str) -> list[str]:
    """Identificadores de la API del gold que aparecen en las líneas de uso de `module` (el módulo inventado) en la
    consulta. No cuenta strings, dunders ni nombres de la stdlib común."""
    tokens = set(_IDENT.findall(_usage_lines(query, module)))
    return sorted(n for n in forbidden_names(gold)
                  if _IDENT.fullmatch(n) and not n.startswith("__") and n not in _STDLIB_API and n in tokens)
