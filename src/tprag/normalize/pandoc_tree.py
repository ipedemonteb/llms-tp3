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

from .codeblocks import ConsoleMode, process_block
from .prepass import prepass_md, prepass_rst

# Markdown sin las extensiones que cambian el texto: comillas tipográficas, emojis (`:tada:`) y fórmulas con `$`
# (en la doc, `$` suele ser el prompt de una terminal o un precio).
MD_FORMAT = "commonmark_x-smart-emoji-tex_math_dollars"

API_ROLES = {"func", "meth", "class", "attr", "exc", "data", "mod", "obj", "const", "type", "member", "var",
             "macro", "function", "method", "property", "decorator", "deco"}
TEXT_ROLES = {"ref", "doc", "term", "numref", "download", "abbr", "menuselection", "guilabel", "title-reference"}
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
    return json.loads(pypandoc.convert_text(text, "json", format=fmt))


def to_markdown(tree: dict) -> str:
    md = pypandoc.convert_text(json.dumps(tree), "gfm", format="json", extra_args=["--wrap=none"])
    out, fence = [], None
    for line in md.splitlines():
        m = re.match(r"^\s*(`{3,}|~{3,})", line)
        if m and (fence is None or m.group(1).startswith(fence)):
            fence = None if fence else m.group(1)
        elif fence is None:
            line = line.replace("\\$", "$")  # pandoc escapa `$` para que no se lea como fórmula; acá es texto
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
    if t in ("Code", "CodeBlock"):
        return c[1]
    return stringify(c)


def _classes(attr) -> list[str]:
    return attr[1]


def _role_text(role: str, text: str) -> tuple[str, bool]:
    """(texto a mostrar, si va como código) de una referencia de Sphinx."""
    m = re.match(r"^(.*?)\s*<([^<>]+)>$", text, re.S)
    title, target = (m.group(1).strip(), m.group(2).strip()) if m else (None, text.strip())
    role = role.split(":")[-1]  # py:class → class
    if role in API_ROLES:
        return target.lstrip("~!."), True  # nombre completo: librería y módulo dentro del texto
    if role == "pep":
        return f"PEP {target}", False
    if role == "rfc":
        return f"RFC {target}", False
    if role in TEXT_ROLES:
        return (title or target).lstrip("~!"), False
    return (title or target).lstrip("~!"), True  # envvar, setting, file, command, option, kbd…


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
            return self.inlines(c[1])
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
            if declared in SCREENSHOT_BLOCKS:  # el sitio muestra una captura, no el texto del bloque
                self.dropped[f"captura ({declared})"] += 1
                return []
            lang, code = process_block(declared, text, self.console_mode)
            if lang != "python" and HTML_STYLE_TAG.search(code):
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
            return [self.table(node)]
        if t == "LineBlock":
            return [{"t": t, "c": [self.inlines(line) for line in c]}]
        return [node]

    def table(self, node: dict) -> dict:
        """Aplica las mismas transformaciones a los bloques e inlines de las celdas, en orden."""
        block_types = {"Para", "Plain", "CodeBlock", "Div", "BlockQuote", "BulletList", "OrderedList"}
        inline_types = {"Code", "Link", "Image", "Span", "RawInline"}

        def walk(x):
            if isinstance(x, list):
                out = []
                for y in x:
                    if isinstance(y, dict) and y.get("t") in block_types:
                        out.extend(self.block(y))
                    elif isinstance(y, dict) and y.get("t") in inline_types:
                        out.extend(self.inline(y))
                    else:
                        out.append(walk(y))
                return out
            if isinstance(x, dict):
                return {k: walk(v) for k, v in x.items()}
            return x
        return walk(node)

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
