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


# --- Referencias (rst `:func:`x`` en el árbol; MyST `{func}`x`` en el paso previo de Markdown) ---------------

API_ROLES = {"func", "meth", "class", "attr", "exc", "data", "mod", "obj", "const", "type", "member", "var",
             "macro", "function", "method", "property", "decorator", "deco",
             # `texto` sin rol en rst: Sphinx lo resuelve con el rol por defecto, que en estas docs es casi
             # siempre `py:obj` o `any` y se ve como código (flask, scikit-learn, matplotlib).
             "title-reference"}
TEXT_ROLES = {"ref", "doc", "term", "numref", "download", "abbr", "menuselection", "guilabel"}
# Alias de módulo que usan las referencias de la doc de sqlalchemy (`:class:`_orm.Session``), ~3.900 en 67 archivos.
SQLALCHEMY_ALIASES = {
    "orm": "sqlalchemy.orm", "schema": "sqlalchemy.schema", "sql": "sqlalchemy.sql", "engine": "sqlalchemy.engine",
    "asyncio": "sqlalchemy.ext.asyncio", "sa": "sqlalchemy", "dml": "sqlalchemy.sql.dml", "types": "sqlalchemy.types",
    "postgresql": "sqlalchemy.dialects.postgresql", "expression": "sqlalchemy.sql.expression",
    "query": "sqlalchemy.orm.query", "functions": "sqlalchemy.sql.functions", "pool": "sqlalchemy.pool",
    "events": "sqlalchemy.events", "result": "sqlalchemy.engine.result", "sqltypes": "sqlalchemy.types",
    "mysql": "sqlalchemy.dialects.mysql", "reflection": "sqlalchemy.engine.reflection",
    "sqlite": "sqlalchemy.dialects.sqlite", "baked": "sqlalchemy.ext.baked", "session": "sqlalchemy.orm.session",
    "ormevents": "sqlalchemy.orm.events", "mssql": "sqlalchemy.dialects.mssql", "exc": "sqlalchemy.exc",
}


def role_text(role: str, text: str) -> tuple[str, bool]:
    """(texto a mostrar, si va como código) de una referencia de Sphinx o MyST."""
    m = re.match(r"^(.*?)\s*<([^<>]+)>$", text, re.S)
    title, target = (m.group(1).strip(), m.group(2).strip()) if m else (None, text.strip())
    role = role.split(":")[-1]  # py:class → class
    if role in API_ROLES:
        return expand_alias(target.lstrip("~!.")), True  # nombre completo: librería y módulo dentro del texto
    if role == "pep":
        return f"PEP {target}", False
    if role == "rfc":
        return f"RFC {target}", False
    if role in TEXT_ROLES:
        return (title or target).lstrip("~!"), False
    # envvar, setting, file, command, option, kbd, paramref (sqlalchemy: `_orm.relationship.foreign_keys`)…
    return expand_alias((title or target).lstrip("~!")), True


def expand_alias(name: str) -> str:
    """`_orm.Session` → `sqlalchemy.orm.Session` (alias de módulo de la doc de sqlalchemy)."""
    alias = re.match(r"_(\w+)\.", name)
    if alias and alias.group(1) in SQLALCHEMY_ALIASES:
        return SQLALCHEMY_ALIASES[alias.group(1)] + name[alias.end() - 1:]
    return name


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
# <Note>, </Flex>… Sin indentación: una línea indentada como `<Response [200]>` es salida de código (los componentes
# anidados quedan sin indentación al desindentar el contenido del que los contiene).
JSX_COMPONENT = re.compile(r"^</?[A-Z][\w.]*(?=[\s>/]|$)")
HEADER_ATTRS = re.compile(r"^(#{1,6}\s.*?)\s*\{\s*[#.:][^}]*\}\s*$")
MYST_OPTION = re.compile(r"^:[\w-]+:")
MYST_ROLE = re.compile(r"\{([\w:.-]+)\}`([^`]+)`")  # {func}`attrs.define`
MYST_LABEL = re.compile(r"^\s*\([\w.:\-]+\)=\s*$")  # (extending-metadata)=
# Links de MkDocs (autorefs) `[`Annotated`][typing.Annotated]`: el destino no está definido en la página. El `[` no
# puede venir pegado a un nombre o a otro `]`, para no tocar índices como `x[0][1]`.
AUTOREF = re.compile(r"(?<![\w\]`)])\[([^\[\]]+)\]\[[\w.:\-]*\]")
# Hugging Face doc-builder (transformers)
HF_LINK = re.compile(r"(?<![\w\]`)])\[`~?([\w.]+(?:\(\))?)`\](?![\[(])")  # [`Pipeline`], [`~GenerationMixin.generate`]
HF_DIRECTIVE = re.compile(r"^\s*\[\[(autodoc|open-in-colab)\]\]")
HF_OPTIONS = re.compile(r"^\s*<hfoptions\b[^>]*>\s*$")
HF_OPTION = re.compile(r'^\s*<hfoption\s+id="([^"]*)"\s*>\s*$')
HF_CLOSE = re.compile(r"^\s*</hfoptions?>\s*$")
# Componentes JSX (streamlit, Hugging Face, Docusaurus)
JSX_ADMONITIONS = {"Note": "note", "Tip": "tip", "Important": "important", "Warning": "warning", "Info": "info",
                   "Caution": "caution", "Danger": "danger", "Admonition": "note"}
JSX_NAVIGATION = {"InlineCallout", "InlineCalloutContainer", "RefCard", "Tile", "TileContainer", "DataSourcesCard",
                  "CodeTile", "Cloud", "Image", "YouTube", "Youtube", "GithubLink", "ExampleGithubLink",
                  "Autofunction"}  # tarjetas de links, capturas y referencia autogenerada
MATERIAL_ICON = re.compile(r"<i\b[^>]*material-icons[^>]*>[^<]*</i>\s*")  # streamlit
GRADIO_META = re.compile(r"^(Tags|Related spaces|Contributed by)\b.*$")

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


def _reindented(lines: list[str], indent: str) -> list[str]:
    """Vuelve a poner la indentación original: una directiva dentro de un ítem de lista tiene que quedar adentro."""
    return [indent + l if l.strip() else "" for l in lines]


def _inline_md(line: str) -> str:
    """Referencias de MyST (`{func}`attrs.define``) como las de rst; links de MkDocs `[texto][ref]` y de Hugging
    Face `[`~a.b.C`]` → su texto; íconos de streamlit (`<i … material-icons…>warning</i>`) afuera."""
    def role(m):
        shown, as_code = role_text(m.group(1), m.group(2))
        return f"`{shown}`" if as_code else shown
    line = AUTOREF.sub(r"\1", MYST_ROLE.sub(role, line))
    line = HF_LINK.sub(lambda m: f"`{m.group(1)}`", line)
    return MATERIAL_ICON.sub("", line)


def _jsx_component(lines: list[str], i: int, out: list[str]) -> int:
    """Componente de MDX/JSX (streamlit, Docusaurus, Hugging Face): avisos como avisos, navegación afuera, el resto
    con su contenido desindentado (si no, pandoc lee el contenido indentado como un bloque de código)."""
    if lines[i].startswith("</"):  # cierre suelto
        return i + 1
    name = JSX_COMPONENT.match(lines[i]).group(0)[1:]
    end = i
    while end < len(lines) and ">" not in lines[end]:  # la etiqueta de apertura puede ocupar varias líneas
        end += 1
    tag = "\n".join(lines[i:end + 1])
    head, _, rest = tag.partition(">")
    if head.rstrip().endswith("/"):  # <Image … />, <Autofunction … />
        return end + 1
    # Cierre: en la misma línea (`<Tip>texto</Tip>`) o en una línea propia, contando anidamientos del mismo nombre.
    close = f"</{name}>"
    if close in rest:
        inner, k = [rest.split(close)[0]], end
    else:
        depth, k = 1, end + 1
        while k < len(lines):
            depth += len(re.findall(rf"<{name}(?=[\s>])", lines[k])) - lines[k].count(close)
            if depth <= 0:
                break
            k += 1
        if k >= len(lines):  # sin cierre (etiqueta mal escrita): solo se saca la etiqueta, no el resto de la página
            return end + 1
        inner = ([rest] if rest.strip() else []) + lines[end + 1:k]
        before = lines[k].split(close)[0]
        if before.strip():
            inner.append(before)
    body = _md_lines(_dedent(inner))
    if name in JSX_NAVIGATION:
        return k + 1
    if name in JSX_ADMONITIONS:
        cls = "warning" if name == "Tip" and "warning={true}" in head else JSX_ADMONITIONS[name]
        out += _div(cls, None, body)
    elif name == "Collapse":  # streamlit: <Collapse title="…"> se ve como un desplegable con ese título
        title = re.search(r'title="([^"]*)"', head)
        out += ["", f"**{title.group(1)}**" if title else "", "", *body, ""]
    else:  # <Flex>, <div>-like y desconocidos: se quitan las etiquetas, el contenido queda
        out += ["", *body, ""]
    return k + 1


def _myst_directive(name: str, arg: str, body: list[str], fence: str) -> list[str]:
    """Directiva MyST sin indentación (quien llama la vuelve a indentar)."""
    body = _myst_body(body)
    if name in MD_CODE:
        # code-block/code/sourcecode: el argumento es el lenguaje. Las de lenguaje fijo (doctest, ipython…) usan el
        # argumento para otra cosa: ```{doctest} namedtuples (cattrs) es el nombre de un grupo de tests.
        lang = (arg.split()[0] if arg else "") if MD_CODE[name] == "" else MD_CODE[name]
        return [f"{fence}{lang}", *body, fence]
    if name == "eval-rst":
        return [f"{fence}eval-rst", *body, fence]
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
                out += _reindented(_myst_directive(d.group(1), d.group(2).strip(), _dedent(lines[i + 1:j]), fence),
                                   indent)
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
                new = _div(name, arg.strip(), _md_lines(body), title_div=True)  # el argumento es el título
            else:
                new = _myst_directive(name, (arg or "").strip(), body, "```")
            out += _reindented(new, indent)
            i = j + 1
            continue

        if MYST_LABEL.match(line):  # `(extending-metadata)=`: destino de links de MyST, el lector no lo ve
            i += 1
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
        # Hugging Face (transformers): pestañas <hfoptions>/<hfoption id="…"> como las demás. Las líneas en blanco
        # alrededor de las marcas arreglan los bloques de código pegados a la etiqueta.
        if HF_OPTIONS.match(line):
            out += ["", "::: {.tabs}", ""]
            i += 1
            continue
        m = HF_OPTION.match(line)
        if m:
            out += ["", "::: {.tab}", m.group(1), ""]
            i += 1
            continue
        if HF_CLOSE.match(line):
            out += ["", ":::", ""]
            i += 1
            continue
        m = HF_DIRECTIVE.match(line)
        if m:  # [[autodoc]] Clase (y sus métodos indentados debajo) y [[open-in-colab]]: el lector no los ve así
            i += 1
            while m.group(1) == "autodoc" and i < len(lines) and re.match(r"^\s+-\s+\S", lines[i]):
                i += 1
            continue
        if GRADIO_META.match(line):  # gradio: "Tags: …", "Related spaces: …", "Contributed by …"
            i += 1
            continue

        if JSX_COMPONENT.match(line):
            i = _jsx_component(lines, i, out)
            continue

        m = HEADER_ATTRS.match(line)
        out.append(_inline_md(m.group(1) if m else line))
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


# --- README ---------------------------------------------------------------------------------------

# Secciones de un README que no explican cómo usar la librería (loaders.md §3.5, cuarta excepción).
README_NOISE = re.compile(r"^(change\s*log|changes|release\s*notes|release\s*history|history|what'?s\s*new|"
                          r"contribut\w*|license|licence|authors?|credits|acknowledg\w*|sponsors?|backers|"
                          r"code\s*of\s*conduct|security|reporting\s+(a\s+)?security)\b", re.I)
CHANGELOG = re.compile(r"^(change\s*log|changes|release\s*notes|release\s*history|history|what'?s\s*new)\b", re.I)
MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def clean_readme(text: str, fmt: str) -> str:
    """Saca las secciones de ruido (changelog, licencia, contribuciones…) hasta el próximo título de su nivel."""
    lines = text.split("\n")
    rst_levels: dict[str, int] = {}  # en rst el nivel lo da el orden de aparición de cada subrayado
    out, skip_level, fence = [], None, None
    for i, line in enumerate(lines):
        if fmt == "md" and (f := FENCE.match(line)):  # un `# License` dentro de un bloque de código no es título
            fence = None if fence and f.group(2).startswith(fence) else (fence or f.group(2))
        level, title = (None, "") if fence else _heading(lines, i, fmt, rst_levels)
        if level is not None:
            title = title.strip("*_`: ")
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if skip_level is None and CHANGELOG.match(title):
                # El changelog va al final del README y sus entradas son títulos de cualquier nivel (pydantic:
                # `## v2.14.1`, `## New Contributors`…): se corta hasta el final.
                break
            if skip_level is None and README_NOISE.match(title):
                skip_level = level
        if skip_level is None:
            out.append(line)
    return "\n".join(out)


def _heading(lines: list[str], i: int, fmt: str, rst_levels: dict[str, int]) -> tuple[int | None, str]:
    """(nivel, texto) si la línea i es un título."""
    if fmt == "md":
        m = MD_HEADING.match(lines[i])
        if m:
            return len(m.group(1)), m.group(2)
        if i + 1 < len(lines) and lines[i].strip() and re.match(r"^(=+|-+)\s*$", lines[i + 1]):
            return (1 if lines[i + 1].startswith("=") else 2), lines[i]  # título subrayado (Title\n=====)
        return None, ""
    if (i + 1 < len(lines) and lines[i].strip() and RST_UNDERLINE.match(lines[i + 1])
            and not RST_UNDERLINE.match(lines[i]) and len(lines[i + 1].strip()) >= len(lines[i].strip()) - 2):
        char = lines[i + 1].strip()[0]
        return rst_levels.setdefault(char, len(rst_levels) + 1), lines[i]
    return None, ""
