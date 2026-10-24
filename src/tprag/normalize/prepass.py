"""Paso previo sobre el texto, antes de pandoc: lo que pandoc rompe o pierde (loaders.md §3.6).

rst:
- Directivas con código que pandoc no conoce (`ipython`, `plot`, `testcode`, `doctest`…) → `code-block`.
  Sin esto, pandoc junta las líneas del código en un párrafo.
- `code-tab` (sphinx-tabs) → pestaña con título + `code-block`.
- Directivas de descripción de API (`class::`, `function::`, `data::`…) → párrafo con la firma como código y
  la descripción como texto normal. Sin esto, pandoc pierde la firma (y `class::` choca con la directiva
  básica de rst que asigna una clase CSS).
- `testsetup` / `testcleanup` se eliminan: el sitio publicado no los muestra.
"""

import re

DIRECTIVE = re.compile(r"^(\s*)\.\.\s+(?:([\w-]+):)?([\w-]+)::(?:\s+(.*?))?\s*$")
OPTION = re.compile(r"^\s+:[\w-]+:")
RST_LITERAL = re.compile(r"^(?!\s*\.\.\s).*::\s*$")

# Directivas cuyo contenido es código, con el lenguaje que se le asigna (None: el argumento de la directiva).
CODE_DIRECTIVES = {"code-block": None, "code": None, "sourcecode": None, "ipython": "python", "plot": "python",
                   "testcode": "python", "jupyter-execute": "python", "doctest": "pycon", "testoutput": "text"}
HIDDEN_DIRECTIVES = {"testsetup", "testcleanup"}
API_DIRECTIVES = {
    "function", "class", "method", "attribute", "data", "exception", "decorator", "decoratormethod",
    "classmethod", "staticmethod", "property", "type", "member", "macro", "var", "envvar", "option", "confval",
    "describe", "object",
    "setting", "templatetag", "templatefilter", "fieldlookup", "django-admin", "django-admin-option",  # django
}
# Pseudo-decoradores de la directiva ipython (`@savefig x.png`, `@suppress`): no son código.
IPYTHON_DECORATOR = re.compile(r"^\s*@(savefig|suppress|verbatim|doctest|okexcept|okwarning)\b")


def _block_end(lines: list[str], i: int) -> int:
    """Índice siguiente al cuerpo indentado de la directiva de la línea i."""
    base = len(lines[i]) - len(lines[i].lstrip())
    j = i + 1
    while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > base):
        j += 1
    while j > i + 1 and not lines[j - 1].strip():
        j -= 1  # las líneas en blanco del final no son parte del cuerpo
    return j


def _body(lines: list[str], i: int, end: int) -> list[str]:
    """Cuerpo de la directiva sin sus opciones (`:linenos:`, `:caption: x`, `:suppress:`…)."""
    body = lines[i + 1:end]
    k = 0
    while k < len(body) and not body[k].strip():
        k += 1
    while k < len(body) and OPTION.match(body[k]):
        k += 1
    while k < len(body) and not body[k].strip():
        k += 1
    return body[k:]


def _reindent(body: list[str], indent: str) -> list[str]:
    widths = [len(l) - len(l.lstrip()) for l in body if l.strip()]
    cut = min(widths) if widths else 0
    return [indent + l[cut:] if l.strip() else "" for l in body]


def prepass_rst(text: str) -> str:
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        m = DIRECTIVE.match(lines[i])
        if not m:
            # Párrafo que termina en "::": lo que sigue es un bloque literal (código) y no se recorre.
            i = _block_end(lines, i) if RST_LITERAL.match(lines[i]) else i + 1
            continue
        indent, domain, name, arg = m.group(1), m.group(2), m.group(3), (m.group(4) or "").strip()
        end = _block_end(lines, i)
        body = _body(lines, i, end)

        if name in HIDDEN_DIRECTIVES:
            lines[i:end] = []
            continue

        if name in CODE_DIRECTIVES and domain is None:
            if name == "plot" and arg and not any(l.strip() for l in body):
                lines[i:end] = []  # `.. plot:: ruta/al/script.py`: el código está en otro archivo
                continue
            lang = arg.split()[0] if CODE_DIRECTIVES[name] is None and arg else (CODE_DIRECTIVES[name] or "")
            if name == "ipython":
                body = [l for l in body if not IPYTHON_DECORATOR.match(l)]
            new = [f"{indent}.. code-block:: {lang}".rstrip(), "", *_reindent(body, indent + "   ")]
            lines[i:end] = new
            i += len(new)  # el código no se recorre: puede mostrar rst (`.. note::`) como ejemplo
            continue

        if name == "code-tab":  # `.. code-tab:: python Synchronous` (sphinx-tabs)
            lang, _, label = arg.partition(" ")
            inner = indent + "   "
            new = [f"{indent}.. tab:: {label or lang}", "", f"{inner}.. code-block:: {lang}", "",
                   *_reindent(body, inner + "   ")]
            lines[i:end] = new
            i += len(new)
            continue

        if name in API_DIRECTIVES:
            # Los métodos dentro de una clase quedan al nivel de la clase y se procesan en las vueltas siguientes.
            signature = [f"{indent}``{arg}``", ""] if arg else []
            lines[i:end] = [*signature, *_reindent(body, indent)]
            i += len(signature) if signature else 0
            continue

        i += 1
    return "\n".join(lines)
