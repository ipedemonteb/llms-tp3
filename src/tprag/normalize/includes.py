"""Inclusiones: páginas que pegan el contenido de otro archivo (loaders.md §3.3).

Lo usan dos lados, por eso vive acá:
- La descarga (`src/scripts/fetch_raw_docs.py --add-included`): encuentra los destinos y los resuelve contra la
  lista de archivos del repo, para bajar los que faltan.
- El normalizador: reemplaza cada instrucción por el contenido del archivo (antes del paso previo y de pandoc), y
  calcula qué archivos son destino de una inclusión, que entonces no se procesan como documento propio (§3.3, 3b).

Sintaxis: `.. literalinclude::` / `.. include::` (rst), ```` ```{literalinclude} ```` / ```` ```{include} ````
(MyST), `{* archivo.py hl[…] ln[a:b] *}` (fastapi, typer), `--8<-- "archivo"` (snippets de MkDocs, textual) y
`$code_nombre` (gradio: demo/nombre/run.py).
"""

import ast
import re
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

INCLUDE_PATTERNS = [
    re.compile(r"^\s*\.\.\s+(?:literalinclude|include)::\s*(\S+)", re.M),  # reST (también {eval-rst} en MyST)
    re.compile(r"^\s*(?:`{3,}|:{3,})\{(?:literalinclude|include)\}\s*(\S+)", re.M),  # MyST
    re.compile(r"^\{\*\s*(\S+?\.py)\b", re.M),  # fastapi y typer
    re.compile(r"^\s*-{2}8<-{2}\s*\"([^\"]+)\"", re.M),  # MkDocs snippets (textual)
]
EXT_LANG = {".py": "python", ".toml": "toml", ".cfg": "ini", ".ini": "ini", ".json": "json", ".yaml": "yaml",
            ".yml": "yaml", ".sh": "bash", ".html": "html", ".js": "js", ".css": "css", ".tcss": "css",
            ".txt": "text", ".sql": "sql", ".xml": "xml"}
MAX_DEPTH = 5  # inclusiones anidadas (también corta ciclos: A incluye B, B incluye A)


# gradio: `$code_hello_world_4` en una guía se publica como el código de demo/hello_world_4/run.py;
# `$demo_hello_world_4` es la app embebida (no es texto).
GRADIO_CODE = re.compile(r"^\s*\$code_(\w+)\s*$", re.M)
GRADIO_DEMO = re.compile(r"^\s*\$demo_\w+\s*$")


def include_targets(text: str) -> list[str]:
    """Rutas que el archivo incluye, tal como están escritas (sin rango de líneas de los snippets)."""
    found = [_strip_range(m.strip("'\"")) for p in INCLUDE_PATTERNS for m in p.findall(text)]
    return found + [f"demo/{name}/run.py" for name in GRADIO_CODE.findall(text)]


def _strip_range(target: str) -> str:
    return re.sub(r"(:\d+){1,2}$", "", target)


def resolve_include(target: str, including: str, tree: set[str], doc_dirs: list[str]) -> str | None:
    """Ruta en el repo del archivo incluido. Cada herramienta resuelve desde un lugar distinto (la página,
    la carpeta de mkdocs.yml, la raíz), así que se prueba desde la carpeta de la página hacia arriba."""
    base = Path(including).parent
    if target.startswith("/"):
        # Sphinx: relativo a la carpeta de conf.py, que puede estar debajo de la de la doc (pytest: doc/en).
        candidates = [f"{d}/{target.lstrip('/')}" for d in doc_dirs] + [target.lstrip("/")]
        candidates += [(b / target.lstrip("/")).as_posix() for b in [base, *base.parents]]
    else:
        candidates = [(b / target).as_posix() for b in [base, *base.parents]]
    for c in candidates:
        c = _normpath(c)
        if c in tree:
            return c
    return None


def _normpath(path: str) -> str:
    """Normaliza una ruta del repo ("a/./b/../c" -> "a/c"); "" si se va fuera del repo."""
    parts = []
    for p in path.split("/"):
        if p == "..":
            if not parts:
                return ""  # se va fuera del repo
            parts.pop()
        elif p not in ("", "."):
            parts.append(p)
    return "/".join(parts)


# --- Resolución en el normalizador -----------------------------------------------------------------

@dataclass
class Corpus:
    """Archivos de una librería en data/raw (rutas del repo, como en el manifest)."""
    paths: set[str]
    read: Callable[[str], str]
    doc_dirs: list[str] = field(default_factory=list)


@dataclass
class Resolved:
    text: str
    used: set[str] = field(default_factory=set)  # archivos pegados
    unresolved: list[str] = field(default_factory=list)  # "página -> destino"


def included_paths(pages: dict[str, str], corpus: Corpus) -> set[str]:
    """Archivos que alguna página incluye: no se procesan como documento propio (3b)."""
    found = set()
    for page, text in pages.items():
        for target in include_targets(text):
            rel = resolve_include(target, page, corpus.paths, corpus.doc_dirs)
            if rel:
                found.add(rel)
    return found


def resolve_includes(text: str, fmt: str, page: str, corpus: Corpus, _depth: int = 0) -> Resolved:
    """Reemplaza cada inclusión por el contenido del archivo. fmt: "rst" o "md"."""
    res = Resolved(text)
    lines = text.split("\n")
    out = _rst(lines, page, corpus, res, _depth) if fmt == "rst" else _md(lines, page, corpus, res, _depth)
    res.text = "\n".join(out)
    return res


def _load(target: str, page: str, corpus: Corpus, res: Resolved) -> tuple[str | None, str]:
    """(contenido o None, ruta resuelta)."""
    rel = resolve_include(_strip_range(target), page, corpus.paths, corpus.doc_dirs)
    if rel is None:
        res.unresolved.append(f"{page} -> {target}")
        return None, ""
    res.used.add(rel)
    return corpus.read(rel), rel


def _nested(content: str, fmt: str, rel: str, corpus: Corpus, res: Resolved, depth: int) -> str:
    """Las inclusiones de un archivo incluido son relativas a ese archivo."""
    if depth + 1 >= MAX_DEPTH:
        res.unresolved.append(f"{rel} -> (inclusiones anidadas: límite {MAX_DEPTH})")
        return content
    inner = resolve_includes(content, fmt, rel, corpus, depth + 1)
    res.used |= inner.used
    res.unresolved += inner.unresolved
    return inner.text


def _select(content: str, opts: dict[str, str]) -> str:
    """Opciones de literalinclude / include: pyobject, lines, start/end, dedent."""
    if "pyobject" in opts:
        content = _pyobject(content, opts["pyobject"]) or content
    lines = content.split("\n")
    if "lines" in opts:
        lines = _pick_lines(lines, opts["lines"])
    for key, after in (("start-after", True), ("start-at", False)):
        if key in opts:
            idx = next((k for k, l in enumerate(lines) if opts[key] in l), None)
            if idx is not None:
                lines = lines[idx + 1 if after else idx:]
    for key, before in (("end-before", True), ("end-at", False)):
        if key in opts:
            idx = next((k for k, l in enumerate(lines) if opts[key] in l), None)
            if idx is not None:
                lines = lines[:idx if before else idx + 1]
    text = "\n".join(lines)
    if "dedent" in opts:
        text = textwrap.dedent(text)
    return text.strip("\n")


def _pick_lines(lines: list[str], spec: str) -> list[str]:
    """`1-5,8,10-` (Sphinx) o `1:5,8` (fastapi ln[…]): números de línea desde 1, rangos inclusivos."""
    picked = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        a, sep, b = part.replace(":", "-").partition("-")
        start = int(a) if a else 1
        end = (int(b) if b else len(lines)) if sep else start
        picked += lines[start - 1:end]
    return picked


def _pyobject(source: str, name: str) -> str | None:
    """Código de una clase o función (`Clase.metodo` también) dentro de un archivo Python."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    nodes = tree.body
    found = None
    for part in name.split("."):
        found = next((n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                      and n.name == part), None)
        if found is None:
            return None
        nodes = found.body
    start = min([found.lineno] + [d.lineno for d in found.decorator_list])
    # Un método sale con la indentación de la clase; se desindenta para que quede como código independiente.
    return textwrap.dedent("\n".join(source.split("\n")[start - 1:found.end_lineno]))


def _lang(opts: dict[str, str], rel: str) -> str:
    return opts.get("language") or EXT_LANG.get(Path(rel).suffix.lower(), "")


def _fence(content: str) -> str:
    longest = max((len(m) for m in re.findall(r"`{3,}", content)), default=2)
    return "`" * max(3, longest + 1)


# rst ------------------------------------------------------------------------------------------------

RST_INCLUDE = re.compile(r"^(\s*)\.\.\s+(literalinclude|include)::\s*(\S+)\s*$")
RST_OPTION = re.compile(r"^\s+:([\w-]+):\s*(.*)$")
RST_LITERAL = re.compile(r"^(?!\s*\.\.\s).*::\s*$")


def _rst(lines: list[str], page: str, corpus: Corpus, res: Resolved, depth: int) -> list[str]:
    out, i = [], 0
    while i < len(lines):
        m = RST_INCLUDE.match(lines[i])
        if not m:
            out.append(lines[i])
            if RST_LITERAL.match(lines[i]):  # bloque literal: puede mostrar `.. include::` como ejemplo
                base = len(lines[i]) - len(lines[i].lstrip())
                j = i + 1
                while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > base):
                    out.append(lines[j])
                    j += 1
                i = j
            else:
                i += 1
            continue
        indent, kind, target = m.groups()
        opts, j = {}, i + 1
        while j < len(lines) and (o := RST_OPTION.match(lines[j])):
            opts[o.group(1)] = o.group(2).strip()
            j += 1
        i = j
        content, rel = _load(target, page, corpus, res)
        if content is None:
            continue
        content = _select(content, opts)
        if kind == "literalinclude" or "literal" in opts or not rel.endswith((".rst", ".txt", ".inc")):
            out += [f"{indent}.. code-block:: {_lang(opts, rel)}".rstrip(), "",
                    *(f"{indent}   {l}" if l.strip() else "" for l in content.split("\n")), ""]
        else:
            content = _nested(content, "rst", rel, corpus, res, depth)
            out += ["", *(f"{indent}{l}" if l.strip() else "" for l in content.split("\n")), ""]
    return out


# Markdown -------------------------------------------------------------------------------------------

MD_FENCE = re.compile(r"^(\s*)(`{3,}|~{3,}|:{3,})(.*)$")
MYST_INCLUDE = re.compile(r"^\{(literalinclude|include)\}\s*(\S+)")
FASTAPI = re.compile(r"^(\s*)\{\*\s*(\S+)(.*?)\*\}\s*$")
SNIPPET = re.compile(r'^(\s*)-{2}8<-{2}\s*"([^"]+)"\s*$')


def _md(lines: list[str], page: str, corpus: Corpus, res: Resolved, depth: int) -> list[str]:
    out, i, fence = [], 0, None  # fence: bloque de código abierto (los snippets también van adentro)
    while i < len(lines):
        line = lines[i]
        f = MD_FENCE.match(line)
        if f and fence is None and (inc := MYST_INCLUDE.match(f.group(3).strip())):
            # ```{literalinclude} ruta  /  :::{include} ruta  (las opciones van adentro, hasta el cierre)
            indent, marks, (kind, target) = f.group(1), f.group(2), inc.groups()
            close = re.compile(rf"^\s*{re.escape(marks[0])}{{{len(marks)},}}\s*$")
            j, opts = i + 1, {}
            while j < len(lines) and not close.match(lines[j]):
                if (o := re.match(r"^\s*:([\w-]+):\s*(.*)$", lines[j])):
                    opts[o.group(1)] = o.group(2).strip()
                j += 1
            i = j + 1
            content, rel = _load(target, page, corpus, res)
            if content is None:
                continue
            content = _select(content, opts)
            if kind == "include" and rel.endswith((".md", ".markdown")):
                out += ["", *_nested(content, "md", rel, corpus, res, depth).split("\n"), ""]
            else:
                ticks = _fence(content)
                out += [f"{indent}{ticks}{_lang(opts, rel)}", *content.split("\n"), f"{indent}{ticks}"]
            continue
        if f and f.group(2)[0] != ":":
            if fence is None:
                fence = f.group(2)
            elif f.group(2).startswith(fence) and not f.group(3).strip():
                fence = None
            out.append(line)
            i += 1
            continue

        m = FASTAPI.match(line) if fence is None else None
        if m:  # {* ../../docs_src/x.py hl[3] ln[1:7] *}
            indent, target, rest = m.groups()
            content, rel = _load(target, page, corpus, res)
            if content is not None:
                ln = re.search(r"\bln\[([^\]]*)\]", rest)
                if ln:
                    content = "\n".join(_pick_lines(content.split("\n"), ln.group(1)))
                content = content.strip("\n")
                ticks = _fence(content)
                out += [f"{indent}{ticks}{_lang({}, rel)}", *content.split("\n"), f"{indent}{ticks}"]
            i += 1
            continue

        m = GRADIO_CODE.match(line) if fence is None else None
        if m:  # gradio: $code_nombre → el código de demo/nombre/run.py
            content, rel = _load(f"demo/{m.group(1)}/run.py", page, corpus, res)
            if content is not None:
                content = content.strip("\n")
                ticks = _fence(content)
                out += [f"{ticks}python", *content.split("\n"), ticks]
            i += 1
            continue
        if fence is None and GRADIO_DEMO.match(line):  # app embebida: el lector ve la app, no texto
            i += 1
            continue

        m = SNIPPET.match(line)
        if m:  # --8<-- "archivo.py" o "archivo.py:5:10"; dentro de un bloque de código se pega tal cual
            indent, target = m.groups()
            content, rel = _load(target, page, corpus, res)
            if content is not None:
                rng = re.search(r":(\d+)(?::(\d+))?$", target)
                if rng:
                    content = "\n".join(_pick_lines(content.split("\n"), f"{rng.group(1)}-{rng.group(2) or ''}"))
                content = content.strip("\n")
                if fence is not None:
                    out += [f"{indent}{l}" if l.strip() else "" for l in content.split("\n")]
                elif rel.endswith((".md", ".markdown")):
                    out += ["", *_nested(content, "md", rel, corpus, res, depth).split("\n"), ""]
                else:
                    ticks = _fence(content)
                    out += [f"{indent}{ticks}{_lang({}, rel)}", *content.split("\n"), f"{indent}{ticks}"]
            i += 1
            continue

        out.append(line)
        i += 1
    return out
