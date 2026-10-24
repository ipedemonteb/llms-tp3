"""Paso previo sobre el texto, antes de pandoc: lo que pandoc rompe o pierde (loaders.md §3.6).

rst:
- Directivas con código que pandoc no conoce (`ipython`, `plot`, `testcode`, `doctest`…) → `code-block`.
  Sin esto, pandoc junta las líneas del código en un párrafo.
- `code-tab` (sphinx-tabs) → pestaña con título + `code-block`.
- Directivas de descripción de API (`class::`, `function::`, `data::`…) → párrafo con la firma como código y
  la descripción como texto normal. Sin esto, pandoc pierde la firma (y `class::` choca con la directiva
  básica de rst que asigna una clase CSS).
- `testsetup` / `testcleanup` se eliminan: el sitio publicado no los muestra.

Markdown: cada dialecto marca avisos, pestañas y directivas a su manera, y casi ninguna la entiende pandoc. Se
traducen todos a la misma forma, los contenedores de pandoc (`::: {.note}` … `:::`), con el título o el
argumento como primer párrafo (igual que pandoc deja las directivas de rst). Así el árbol los trata igual sin
importar de dónde vienen:
- MkDocs: avisos `!!! note "Título"` / `??? note` y pestañas `=== "Título"` (contenido indentado);
  `::: paquete.modulo` (mkdocstrings, referencia autogenerada) se elimina.
- fastapi / typer: bloques `/// tip | Título` … `///` y pestañas `//// tab | Título` … `////`.
- MyST: directivas ```` ```{note} ```` y `:::{note}`; las de código (```` ```{code-block} python ````) quedan
  como bloques de código y ```` ```{eval-rst} ```` como bloque `eval-rst`, que el árbol procesa como rst.
- Docusaurus / MDX (playwright): `:::note`, `<Tabs>` / `<TabItem>`, `import … from …` y otros componentes.
- Títulos con atributos (`# First Steps { #first-steps }`) y encabezado YAML (el `title` pasa a ser el H1).
"""

import re

from .codeblocks import MYST_CODE

DIRECTIVE = re.compile(r"^(\s*)\.\.\s+(?:([\w-]+):)?([\w-]+)::(?:\s+(.*?))?\s*$")
OPTION = re.compile(r"^\s+:[\w-]+:")
RST_LITERAL = re.compile(r"^(?!\s*\.\.\s).*::\s*$")
RST_UNDERLINE = re.compile(r"^\s*([=\-~^\"'`#*+])\1{3,}\s*$")  # subrayado de título (numpydoc: "--------")

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
            if lines[i].lstrip().startswith(">>>") and (i == 0 or not lines[i - 1].strip()
                                                        or RST_UNDERLINE.match(lines[i - 1])):
                # Bloque doctest nativo (párrafo que empieza con >>>, sin "::"): pandoc lo lee como texto.
                indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
                end = i
                while end < len(lines) and lines[end].strip():
                    end += 1
                new = [f"{indent}.. code-block:: pycon", "", *_reindent(lines[i:end], indent + "   ")]
                if i and lines[i - 1].strip():
                    new = ["", *new]  # pegado al subrayado del título: se separa con una línea en blanco
                lines[i:end] = new
                i += len(new)
                continue
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


# --- Markdown ---------------------------------------------------------------------------------------

FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
MYST_DIRECTIVE = re.compile(r"^\{([\w-]+)\}\s*(.*)$")  # info string de una directiva MyST: {note} Título
MYST_COLON = re.compile(r"^(\s*)(:{3,})\{([\w-]+)\}\s*(.*)$")  # :::{note} Título
DOCUSAURUS = re.compile(r"^(\s*)(:{3,})([a-z]+)(?:\[(.*?)\]|\s+(.*?))?\s*$")  # :::note Título / :::note[Título]
MKDOCSTRINGS = re.compile(r"^\s*:::\s+[\w.]+\s*$")  # ::: pydantic.BaseModel
MKDOCS_ADMONITION = re.compile(r'^(\s*)(?:!!!|\?\?\?\+?)\s+([\w-]+)(?:\s+"([^"]*)")?\s*$')
MKDOCS_TAB = re.compile(r'^(\s*)===[+!]?\s+"([^"]*)"\s*$')
SLASH_BLOCK = re.compile(r"^(/{3,})\s*([\w-]+)(?:\s*\|\s*(.*?))?\s*$")  # /// tip | Título  (fastapi)
MDX_IMPORT = re.compile(r"^(?:import\s.+\sfrom\s+['\"].+['\"]|import\s+['\"].+['\"]|export\s.+);?\s*$")
JSX_TAB_ITEM = re.compile(r"^\s*<TabItem\b([^>]*)>\s*$")
JSX_CLOSE = re.compile(r"^\s*</(Tabs|TabItem)>\s*$")
# <HTMLCard …>, </Admonition>… Sin indentación: una línea indentada como `<Response [200]>` es salida de código.
JSX_COMPONENT = re.compile(r"^</?[A-Z][\w.]*(?=[\s>/]|$)")
HEADER_ATTRS = re.compile(r"^(#{1,6}\s.*?)\s*\{\s*[#.:][^}]*\}\s*$")
MYST_OPTION = re.compile(r"^:[\w-]+:")

# Directivas que el lector no ve como texto (o que resuelven otras piezas, como las inclusiones).
MD_DROPPED = {"toctree", "contents", "image", "figure", "include", "literalinclude", "autodoc", "automodule",
              "autoclass", "autofunction", "automethod", "autosummary", "raw", "todo"}
MD_CODE = {**MYST_CODE, "code-cell": "python"}  # code-cell: notebooks en MyST


def _div(cls: str, title: str | None, body: list[str], title_div: bool = False) -> list[str]:
    """Contenedor de pandoc; el título va como primer párrafo (o como `::: {.title}` para avisos)."""
    head = ["", f"::: {{.{cls}}}"]
    if title:
        head += ["::: {.title}", title, ":::", ""] if title_div else [title, ""]
    return head + body + ["", ":::", ""]


def _indented(lines: list[str], i: int, indent: int) -> int:
    """Índice siguiente al contenido indentado más que `indent` que empieza en la línea i."""
    j = i
    while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > indent):
        j += 1
    while j > i and not lines[j - 1].strip():
        j -= 1
    return j


def _dedent(lines: list[str]) -> list[str]:
    widths = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    cut = min(widths) if widths else 0
    return [l[cut:] for l in lines]


def _myst_body(body: list[str]) -> list[str]:
    """Cuerpo de una directiva MyST sin sus opciones (`:caption: x` o un bloque YAML entre `---`)."""
    k = 0
    if body and body[0].strip() == "---":
        k = next((j + 1 for j in range(1, len(body)) if body[j].strip() == "---"), 0)
    while k < len(body) and MYST_OPTION.match(body[k].strip()):
        k += 1
    while k < len(body) and not body[k].strip():
        k += 1
    return body[k:]


def _myst_directive(name: str, arg: str, body: list[str], fence: str, indent: str) -> list[str]:
    body = _myst_body(body)
    if name in MD_CODE:
        lang = arg.split()[0] if arg else MD_CODE[name]
        return [f"{indent}{fence}{lang}", *body, f"{indent}{fence}"]
    if name == "eval-rst":
        return [f"{indent}{fence}eval-rst", *body, f"{indent}{fence}"]
    if name in MD_DROPPED:
        return []
    if name == "admonition":
        return _div("admonition", arg, _md_lines(body), title_div=True)
    if name in ("tab-set", "tabs"):
        return _div("tabs", None, _md_lines(body))
    if name in ("tab-item", "tab"):
        return _div("tab", arg, _md_lines(body))
    return _div(name, arg or None, _md_lines(body))  # avisos, versiones (el argumento es la versión), otras


def _md_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        m = FENCE.match(line)
        if m:  # bloque de código (o directiva MyST con backticks): se busca su cierre sin recorrer el interior
            indent, fence, info = m.group(1), m.group(2), m.group(3).strip()
            close = re.compile(rf"^\s*{re.escape(fence[0])}{{{len(fence)},}}\s*$")
            j = i + 1
            while j < len(lines) and not close.match(lines[j]):
                j += 1
            d = MYST_DIRECTIVE.match(info)
            if d:
                out += _myst_directive(d.group(1), d.group(2).strip(), _dedent(lines[i + 1:j]), fence, indent)
            else:
                out += lines[i:j + 1]
            i = j + 1
            continue

        colon, docusaurus = MYST_COLON.match(line), DOCUSAURUS.match(line)
        if (colon or docusaurus) and not MKDOCSTRINGS.match(line):
            if colon:
                indent, colons, name, arg = colon.groups()
            else:
                indent, colons, name, bracket_title, arg = docusaurus.groups()
                arg = bracket_title or arg
            close = re.compile(rf"^\s*:{{{len(colons)}}}\s*$")
            j = i + 1
            while j < len(lines) and not close.match(lines[j]):
                j += 1
            body = _dedent(lines[i + 1:j])
            if docusaurus and arg and name in ("note", "tip", "info", "caution", "danger", "warning"):
                out += _div(name, arg.strip(), _md_lines(body), title_div=True)  # el argumento es el título
            else:
                out += _myst_directive(name, (arg or "").strip(), body, "```", indent)
            i = j + 1
            continue

        if re.match(r"^<HTMLCard\b", line):  # playwright: vista previa del HTML + el mismo HTML como código
            j = i + 1
            while j < len(lines) and not lines[j].startswith("</HTMLCard>"):
                j += 1
            card, inside = [], None
            for l in lines[i + 1:j]:  # se conservan solo los bloques de código: la vista previa repite el código
                f = FENCE.match(l)
                if f and inside is None:
                    inside = f.group(2)
                    card.append(l)
                elif inside is not None:
                    card.append(l)
                    if l.strip().startswith(inside) and l.strip().strip(inside[0]) == "":
                        inside = None
            out += ["", *card, ""]
            i = j + 1
            continue

        if MKDOCSTRINGS.match(line):  # referencia autogenerada: la línea y sus opciones indentadas
            i = _indented(lines, i + 1, len(line) - len(line.lstrip()))
            continue

        m = MKDOCS_ADMONITION.match(line)
        if m:
            end = _indented(lines, i + 1, len(m.group(1)))
            out += _div(m.group(2), m.group(3), _md_lines(_dedent(lines[i + 1:end])), title_div=bool(m.group(3)))
            i = end
            continue

        m = MKDOCS_TAB.match(line)
        if m:  # pestañas seguidas (separadas solo por líneas en blanco) forman un grupo
            indent, tabs = len(m.group(1)), []
            while m:
                end = _indented(lines, i + 1, indent)
                tabs += _div("tab", m.group(2), _md_lines(_dedent(lines[i + 1:end])))
                i = end
                k = i
                while k < len(lines) and not lines[k].strip():
                    k += 1
                m = MKDOCS_TAB.match(lines[k]) if k < len(lines) else None
                if m and len(m.group(1)) == indent:
                    i = k
                else:
                    m = None
            out += _div("tabs", None, tabs)
            continue

        m = SLASH_BLOCK.match(line)
        if m:  # fastapi: /// tip | Título … ///  y  //// tab | Título … ////
            tabs = []
            while m:
                slashes, name, title = m.groups()
                j = i + 1
                while j < len(lines) and lines[j].strip() != slashes:
                    j += 1
                body = _md_lines(lines[i + 1:j])
                i = j + 1
                if name != "tab":
                    out += _div(name, title, body, title_div=True)
                    break
                tabs += _div("tab", title, body)
                k = i
                while k < len(lines) and not lines[k].strip():
                    k += 1
                m = SLASH_BLOCK.match(lines[k]) if k < len(lines) else None
                if m and m.group(2) == "tab":
                    i = k
                else:
                    m = None
            if tabs:
                out += _div("tabs", None, tabs)
            continue

        if re.match(r"^\s*<Tabs\b", line):  # MDX: la etiqueta de apertura puede ocupar varias líneas
            while i < len(lines) and not lines[i].rstrip().endswith(">"):
                i += 1
            out += ["", "::: {.tabs}", ""]
            i += 1
            continue
        m = JSX_TAB_ITEM.match(line)
        if m:
            attrs = dict(re.findall(r'(\w+)=["\']([^"\']*)["\']', m.group(1)))
            out += ["", "::: {.tab}", attrs.get("label") or attrs.get("value", ""), ""]
            i += 1
            continue
        if JSX_CLOSE.match(line):
            out += ["", ":::", ""]
            i += 1
            continue
        if MDX_IMPORT.match(line):
            i += 1
            continue
        if JSX_COMPONENT.match(line):  # otros componentes: se quitan las etiquetas, el contenido queda
            if "/>" not in line and not line.rstrip().endswith(">"):
                while i + 1 < len(lines) and not re.search(r"/?>\s*$", lines[i]):
                    i += 1
            i += 1
            continue

        m = HEADER_ATTRS.match(line)
        out.append(m.group(1) if m else line)
        i += 1
    return out


def prepass_md(text: str) -> str:
    lines = text.split("\n")
    title = None
    if lines and lines[0].strip() == "---":  # encabezado YAML (docusaurus, mkdocs): se saca; el title va de H1
        end = next((j for j in range(1, len(lines)) if lines[j].strip() == "---"), None)
        if end is not None:
            meta = lines[1:end]
            title = next((re.sub(r"^title:\s*", "", l).strip().strip("'\"") for l in meta
                          if l.startswith("title:")), None)
            lines = lines[end + 1:]
    out = _md_lines(lines)
    if title and not _has_h1(out):
        out = [f"# {title}", ""] + out
    return "\n".join(out)


def _has_h1(lines: list[str]) -> bool:
    """¿Hay un título `# …` fuera de los bloques de código? (Un `# comentario` de Python no cuenta.)"""
    fence = None
    for l in lines:
        f = FENCE.match(l)
        if f:
            fence = None if fence and f.group(2).startswith(fence) else (fence or f.group(2))
        elif fence is None and re.match(r"^#\s", l):
            return True
    return False
