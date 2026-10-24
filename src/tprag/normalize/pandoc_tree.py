"""rst/md → árbol JSON de pandoc → transformaciones en Python → Markdown (loaders.md §3.5 y §3.6).

El árbol de pandoc es una lista de bloques {"t": tipo, "c": contenido}. Transformaciones:
- Referencias de Sphinx (`:class:`X <a.b.X>``) → `a.b.X` (nombre completo); a páginas o secciones → texto.
- Bloques de código → lenguaje final + conversión de consola (codeblocks.process_block).
- Avisos (`note`, `warning`, `seealso`…) y versiones (`versionadded`…) → párrafo con prefijo.
- Pestañas → todas con su título, salvo variantes de versión de Python y otros lenguajes (§3.5).
- Directivas sin contenido para el lector (`module`, `toctree`, `index`…), imágenes y figuras → se eliminan.
- Links → su texto; HTML crudo → su texto visible (sin etiquetas); citas y contenedores restantes → su contenido.
- MyST `eval-rst` → se procesa como rst.
"""

import html
import json
import re
from collections import Counter
from dataclasses import dataclass, field

import pypandoc

from .codeblocks import ConsoleMode, normalize_lang, process_block
from .prepass import expand_alias, prepass_md, prepass_rst, role_text as _role_text

# Markdown sin las extensiones que cambian el texto: comillas tipográficas, emojis (`:tada:`), y superíndices y
# subíndices (`z^*…^*` en una fórmula se volvía <sup>). Las fórmulas `$…$` sí se leen como fórmulas: así no se
# rompen (transformers, torch); un `$` de terminal o de precio no cumple las reglas de apertura y cierre.
MD_FORMAT = "commonmark_x-smart-emoji-superscript-subscript"
MD_ESCAPE = re.compile(r"\\([\[\]<>$_*~])")

ADMONITIONS = {"note": "Note", "warning": "Warning", "tip": "Tip", "hint": "Hint", "important": "Important",
               "caution": "Caution", "attention": "Attention", "danger": "Danger", "error": "Error",
               "seealso": "See also", "admonition": "Note",
               # MkDocs, fastapi y Docusaurus
               "info": "Info", "check": "Check", "details": "Details", "abstract": "Abstract", "summary": "Summary",
               "success": "Success", "question": "Question", "failure": "Failure", "bug": "Bug",
               "example": "Example", "quote": "Quote", "danger-zone": "Danger"}
HTML_TAG = re.compile(r"<[^>]+>")
# textual: ```{.textual path="docs/examples/x.py"} se publica como captura SVG de la app corriendo.
SCREENSHOT_BLOCKS = {"textual"}
# Etiquetas de formato dentro de ejemplos de terminal (los `<font color=…>` de fastapi), no de código Python.
HTML_STYLE_TAG = re.compile(r"</?(font|span|b|strong|i|em|u)\b[^>]*>", re.I)
TERMINAL_LANGS = {"console", "bash", "text", "powershell", "batch", "doscon", "cmd", "commandline"}
VERSIONS = {"versionadded": "New in version", "versionchanged": "Changed in version",
            "deprecated": "Deprecated since version", "versionremoved": "Removed in version"}
DROPPED_DIVS = {"module", "currentmodule", "highlight", "toctree", "index", "autosummary", "contents",
                "sectionauthor", "codeauthor", "moduleauthor", "tabularcolumns", "raw", "include", "todo",
                "image", "figure", "only-latex", "default-domain", "default-role", "role", "program"}
TAB_GROUPS = {"tabs", "tab-set"}
TABS = {"tab", "group-tab", "tab-item"}
PY_VARIANT = re.compile(r"python\s*3\.\d+|py3\d+|annotated", re.I)
OTHER_LANG = re.compile(r"\b(java|c#|csharp|\.net|javascript|typescript|js|ts|node(\.js)?|go|rust|ruby|c\+\+|cpp)\b",
                        re.I)
PYTHON_TITLE = re.compile(r"\bpython\b", re.I)
TEMPLATE_VAR = re.compile(r"\{\{[^{}]*\}\}|\{%[^{}]*%\}")  # párrafo que es solo una variable o etiqueta Jinja


@dataclass
class Normalized:
    markdown: str
    block_langs: list[str] = field(default_factory=list)  # lenguaje de cada bloque, en orden (numeración estable)
    dropped: Counter = field(default_factory=Counter)  # contenedores eliminados, por clase (para revisar)


# --- pandoc -----------------------------------------------------------------------------------------

def to_tree(text: str, fmt: str) -> dict:
    # --quiet: sin advertencias como "Reference not found for 'version'" (sustituciones que completa Sphinx).
    return json.loads(pypandoc.convert_text(text, "json", format=fmt, extra_args=["--quiet"]))


def to_markdown(tree: dict) -> str:
    md = pypandoc.convert_text(json.dumps(tree), "gfm", format="json", extra_args=["--wrap=none"])
    out, fence = [], None
    for line in md.splitlines():
        m = re.match(r"^\s*(`{3,}|~{3,})", line)
        if m and (fence is None or m.group(1).startswith(fence)):
            fence = None if fence else m.group(1)
        elif fence is None:
            if line.strip() == "<!-- -->":  # pandoc lo pone para separar listas (quedan al sacar imágenes)
                continue
            # pandoc escapa caracteres para que el Markdown se vea bien al renderizarlo (`\[`, `\<`, `\$`…); este
            # texto lo leen un modelo de embeddings y un LLM, así que el `\` es ruido. `\|` (tablas) y `` \` ``
            # (código en línea) se dejan. Un `\` al final de la línea es un salto de línea forzado: también se saca.
            line = re.sub(r"(?<!\\)\\$", "", MD_ESCAPE.sub(r"\1", line))
        out.append(line.rstrip())
    return "\n".join(out).strip() + "\n"


# --- ayudas sobre el árbol --------------------------------------------------------------------------

def _str(text: str) -> list[dict]:
    """Texto plano como inlines de pandoc (palabras separadas por Space)."""
    out = []
    for k, word in enumerate(text.split()):
        if k:
            out.append({"t": "Space"})
        out.append({"t": "Str", "c": word})
    return out


def stringify(node) -> str:
    """Texto visible de un nodo o lista de nodos."""
    if isinstance(node, list):
        return "".join(stringify(n) for n in node)
    if not isinstance(node, dict):
        return ""
    t, c = node.get("t"), node.get("c")
    if t == "Str":
        return c
    if t in ("Space", "SoftBreak", "LineBreak"):
        return " "
    if t in ("Code", "CodeBlock", "Math", "RawInline"):  # Math: [tipo, fórmula]; sin esto, un párrafo con solo
        return c[1] if t != "RawInline" else ""          # una fórmula parecía vacío y se eliminaba
    return stringify(c)


def _classes(attr) -> list[str]:
    return attr[1]


def _flatten(blocks: list[dict]) -> list[dict]:
    """Bloques de una celda de tabla como una sola línea de inlines (sin procesar todavía)."""
    out: list[dict] = []
    for b in blocks:
        t, c = b["t"], b.get("c")
        if t in ("Para", "Plain"):
            part = [{"t": "Space"} if x["t"] in ("SoftBreak", "LineBreak") else x for x in c]
        elif t == "CodeBlock":
            part = [{"t": "Code", "c": [["", [], []], " ".join(c[1].split())]}]
        elif t in ("BulletList", "OrderedList"):
            items = c if t == "BulletList" else c[1]
            part = []
            for item in items:
                part += ([{"t": "Str", "c": ";"}, {"t": "Space"}] if part else []) + _flatten(item)
        elif t in ("Div", "BlockQuote"):
            part = _flatten(c[1] if t == "Div" else c)
        else:
            part = _str(stringify(b))
        if part:
            out += ([{"t": "Space"}] if out else []) + part
    return out


# --- transformaciones -------------------------------------------------------------------------------

class _Transformer:
    def __init__(self, console_mode: ConsoleMode):
        self.console_mode = console_mode
        self.block_langs: list[str] = []
        self.dropped: Counter = Counter()

    # Inlines
    def inlines(self, items: list[dict]) -> list[dict]:
        out = []
        for node in items:
            out.extend(self.inline(node))
        return out

    def inline(self, node: dict) -> list[dict]:
        t, c = node["t"], node.get("c")
        if t == "Code":
            attr, text = c
            role = dict(attr[2]).get("role")
            if role:
                shown, as_code = _role_text(role, text)
                return [{"t": "Code", "c": [["", [], []], shown]}] if as_code else _str(shown)
            return [{"t": "Code", "c": [["", [], []], text]}]
        if t == "Link":
            return self.inlines(c[1])
        if t in ("Image", "RawInline"):
            return []
        if t == "Span":
            if "title-ref" in _classes(c[0]):
                # `texto` sin rol en rst: Sphinx usa el rol por defecto (py:obj o any) y lo muestra como código
                return [{"t": "Code", "c": [["", [], []], expand_alias(stringify(c[1]).lstrip("~!"))]}]
            return self.inlines(c[1])
        if t in ("Superscript", "Subscript"):  # `2<sup>31</sup>` en HTML → `2^31`, más legible como texto
            return [{"t": "Str", "c": ("^" if t == "Superscript" else "_") + stringify(c)}]
        if t in ("Emph", "Strong", "Strikeout", "Superscript", "Subscript", "SmallCaps", "Underline"):
            return [{"t": t, "c": self.inlines(c)}]
        if t == "Quoted":
            return [{"t": t, "c": [c[0], self.inlines(c[1])]}]
        if t == "Cite":
            return self.inlines(c[1])
        if t == "Note":
            return [{"t": t, "c": self.blocks(c)}]
        return [node]

    # Bloques
    def blocks(self, items: list[dict]) -> list[dict]:
        out = []
        for node in items:
            out.extend(self.block(node))
        return out

    def block(self, node: dict) -> list[dict]:
        t, c = node["t"], node.get("c")
        if t in ("Para", "Plain"):
            inl = self.inlines(c)
            text = stringify(inl).strip()
            if TEMPLATE_VAR.fullmatch(text):  # `{{ header }}` de pandas: lo reemplaza Sphinx al construir el sitio
                self.dropped["variable de plantilla"] += 1
                return []
            return [{"t": t, "c": inl}] if text else []
        if t == "Header":
            return [{"t": t, "c": [c[0], ["", [], []], self.inlines(c[2])]}]
        if t == "CodeBlock":
            attr, text = c
            declared = next((k for k in _classes(attr) if k not in ("sourceCode", "numberLines")), None)
            if declared == "eval-rst":  # MyST: el contenido es rst y se procesa como tal
                return self.blocks(to_tree(prepass_rst(text), "rst")["blocks"])
            if normalize_lang(declared) in SCREENSHOT_BLOCKS:  # el sitio muestra una captura, no el texto del bloque
                self.dropped[f"captura ({normalize_lang(declared)})"] += 1
                return []
            if not text.strip():  # p. ej. un snippet de textual (.tcss) que no se pudo resolver
                self.dropped["bloque vacío"] += 1
                return []
            lang, code = process_block(declared, text, self.console_mode)
            if lang in TERMINAL_LANGS and HTML_STYLE_TAG.search(code):  # en un bloque html, <span> es el ejemplo
                code = html.unescape(HTML_STYLE_TAG.sub("", code))
            self.block_langs.append(lang)
            return [{"t": "CodeBlock", "c": [["", [lang] if lang else [], []], code]}]
        if t == "Div":
            return self.div(c[0], c[1])
        if t == "BlockQuote":  # en rst casi siempre es indentación, no una cita
            return self.blocks(c)
        if t in ("BulletList",):
            return [{"t": t, "c": [self.blocks(item) for item in c]}]
        if t == "OrderedList":
            return [{"t": t, "c": [c[0], [self.blocks(item) for item in c[1]]]}]
        if t == "DefinitionList":  # gfm no tiene listas de definiciones: término en negrita + definición
            out = []
            for term, definitions in c:
                out.append({"t": "Para", "c": [{"t": "Strong", "c": self.inlines(term)}]})
                for d in definitions:
                    out.extend(self.blocks(d))
            return out
        if t == "Figure":
            self.dropped[t] += 1
            return []
        if t == "RawBlock":  # HTML crudo: se quitan las etiquetas y queda el texto visible
            fmt, raw = c
            raw = re.sub(r"<(script|style)\b.*?</\1>|<!--.*?-->", "", raw, flags=re.S | re.I)
            text = " ".join(html.unescape(HTML_TAG.sub(" ", raw)).split()) if fmt == "html" else ""
            if not text:
                self.dropped["html"] += 1
                return []
            return [{"t": "Para", "c": _str(text)}]
        if t == "Table":
            return self.table(node)
        if t == "LineBlock":
            return [{"t": t, "c": [self.inlines(line) for line in c]}]
        return [node]

    def table(self, node: dict) -> list[dict]:
        """Tabla simple, que Markdown de GitHub puede escribir como tabla (`| a | b |`).

        Si una celda tiene varios párrafos, un bloque de código, una lista o celdas combinadas, pandoc escribe la tabla
        entera en HTML (`<table><td><span class="math inline">…`): 47 tablas en 33 archivos. Se aplana cada celda a una
        línea (conservando código, fórmulas y notas en línea), las celdas combinadas pasan a simples y el título de la
        tabla va como párrafo antes. Los bloques de código de las celdas pasan a código en línea y no se numeran.
        """
        _, caption, colspecs, thead, tbodies, tfoot = node["c"]
        ncols = len(colspecs)
        empty_attr = ["", [], []]

        def cell(c):
            return [empty_attr, c[1], 1, 1, [{"t": "Plain", "c": self.inlines(_flatten(c[4]))}]]

        def row(r):
            cells = [cell(c) for c in r[1]][:ncols]
            cells += [[empty_attr, {"t": "AlignDefault"}, 1, 1, []] for _ in range(ncols - len(cells))]
            return [empty_attr, cells]

        head = [row(r) for r in thead[1]]
        body = [row(r) for b in tbodies for r in b[2] + b[3]] + [row(r) for r in tfoot[1]]
        if not head and body:  # sin encabezado no hay tabla de Markdown: la primera fila pasa a serlo
            head, body = body[:1], body[1:]
        specs = [[spec[0], {"t": "ColWidthDefault"}] for spec in colspecs]
        simple = {"t": "Table", "c": [empty_attr, [None, []], specs, [empty_attr, head],
                                      [[empty_attr, 0, [], body]], [empty_attr, []]]}
        return self.blocks(caption[1]) + [simple]

    def div(self, attr, content: list[dict]) -> list[dict]:
        classes = _classes(attr)
        cls = classes[0] if classes else ""
        if cls in DROPPED_DIVS or cls.startswith("auto"):
            self.dropped[cls] += 1
            return []
        if cls in ADMONITIONS:
            label = ADMONITIONS[cls]
            if content and content[0]["t"] == "Div" and "title" in _classes(content[0]["c"][0]):
                label = stringify(content[0]["c"][1]).strip() or label
                content = content[1:]
            elif cls == "admonition" and content and content[0]["t"] in ("Para", "Plain"):
                # `.. admonition:: Django's Time Zone` (rst): el título llega como primer párrafo
                label, content = stringify(content[0]["c"]).strip() or label, content[1:]
            return self._labeled(label, self.blocks(content))
        if cls in VERSIONS:
            body = self.blocks(content)
            version = ""
            if body and body[0]["t"] in ("Para", "Plain") and body[0]["c"]:
                first = body[0]["c"]
                version = stringify(first[0])
                rest = first[2:] if len(first) > 1 and first[1]["t"] in ("Space", "SoftBreak") else first[1:]
                body = ([{"t": "Para", "c": rest}] if rest else []) + body[1:]
            return self._labeled(f"{VERSIONS[cls]} {version}".strip(), body)
        if cls in TAB_GROUPS:
            return self.tabs(content)
        return self.blocks(content)  # contenedor desconocido o sin clase: se conserva el contenido

    def _labeled(self, label: str, body: list[dict], suffix: str = ":") -> list[dict]:
        prefix = [{"t": "Strong", "c": _str(label + suffix)}]
        if body and body[0]["t"] in ("Para", "Plain"):
            return [{"t": "Para", "c": prefix + [{"t": "Space"}] + body[0]["c"]}] + body[1:]
        return [{"t": "Para", "c": prefix}] + body

    def tabs(self, content: list[dict]) -> list[dict]:
        # Primero se decide qué pestañas quedan y recién después se procesan: si no, los bloques de las
        # descartadas entrarían en la numeración.
        entries = []  # (título o None si no es una pestaña, bloques sin procesar)
        for node in content:
            if node["t"] != "Div" or not (set(_classes(node["c"][0])) & TABS):
                entries.append((None, [node]))
                continue
            inner = node["c"][1]
            title = ""
            if inner and inner[0]["t"] in ("Para", "Plain"):
                title, inner = stringify(inner[0]["c"]).strip(), inner[1:]
            entries.append((title, inner))
        titles = [t for t, _ in entries if t]
        has_python = any(PYTHON_TITLE.search(t) and not OTHER_LANG.search(t) for t in titles)
        out, seen_variant = [], False
        for title, inner in entries:
            if title and PY_VARIANT.search(title):
                if seen_variant:  # variantes del mismo ejemplo: se queda la primera (la más moderna)
                    self.dropped["tab (variante de Python)"] += 1
                    continue
                seen_variant = True
            elif title and has_python and OTHER_LANG.search(title) and not PYTHON_TITLE.search(title):
                self.dropped["tab (otro lenguaje)"] += 1
                continue
            body = self.blocks(inner)
            if title is not None and not body:  # p. ej. "Output" con solo una imagen
                self.dropped["tab (vacía)"] += 1
                continue
            out.extend(self._labeled(f"Tab: {title}", body, suffix="") if title else body)
        return out


def normalize_tree(tree: dict, console_mode: ConsoleMode = "outputs_as_comments") -> Normalized:
    tr = _Transformer(console_mode)
    tree = {**tree, "blocks": tr.blocks(tree["blocks"])}
    return Normalized(to_markdown(tree), tr.block_langs, tr.dropped)


def normalize_rst(text: str, console_mode: ConsoleMode = "outputs_as_comments") -> Normalized:
    return normalize_tree(to_tree(prepass_rst(text), "rst"), console_mode)


def normalize_md(text: str, console_mode: ConsoleMode = "outputs_as_comments") -> Normalized:
    return normalize_tree(to_tree(prepass_md(text), MD_FORMAT), console_mode)
