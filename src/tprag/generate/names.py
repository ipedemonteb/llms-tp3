"""Nombres sin definir en un script (generador.md §2.6): los usa la post-validación del generador (el reintento de la
iteración, `pipeline.answer` con `retry=True`) y el chequeo `undefined_names` del nivel 1 (`tprag.eval.generation`).
Solo usa el código original y el corregido: se puede calcular en producción, sin el dataset."""

import ast
import builtins
import symtable

_MODULE_NAMES = {"__file__", "__builtins__", "__annotations__", "__path__", "__cached__"}


def undefined_names(code: str) -> set[str] | None:
    """Nombres que el código usa y no define en ningún lado (import, asignación, def/class, parámetro ni builtin): darían
    NameError al ejecutarse. Con `symtable` (biblioteca estándar): un nombre usado dentro de una función o clase que no
    es local ni de una función que la contiene tiene que estar definido en el módulo. None si no se puede saber (no
    parsea, o tiene `from x import *`)."""
    try:
        tree = ast.parse(code)
        table = symtable.symtable(code, "<script>", "exec")
    except SyntaxError:
        return None
    if any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in ast.walk(tree)):
        return None
    defined, used = set(dir(builtins)) | _MODULE_NAMES, set()

    def visit(t: symtable.SymbolTable, top: bool) -> None:
        for s in t.get_symbols():
            if (top or s.is_declared_global()) and (s.is_assigned() or s.is_imported()):
                defined.add(s.get_name())
            if s.is_referenced() and (top or s.is_global()):
                used.add(s.get_name())
        for child in t.get_children():
            visit(child, False)

    visit(table, True)
    return used - defined


def new_undefined_names(original: str, fixed: str) -> list[str]:
    """Nombres sin definir que aparecen en el corregido y no en el original (lo que ya venía mal no se cuenta)."""
    after, before = undefined_names(fixed), undefined_names(original)
    return sorted(after - (before or set())) if after is not None and before is not None else []
