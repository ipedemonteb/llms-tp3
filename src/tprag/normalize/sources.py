"""Formatos que no son documentos de texto: se arman como rst o Markdown y siguen el camino de pandoc.

- Notebooks (`.ipynb`): celdas Markdown → texto; celdas de código → bloques Python. Las salidas ya se sacaron en
  la descarga. Las celdas `raw` (directivas para Sphinx) no las ve el lector como texto y se eliminan.
- Galerías de ejemplos (`.py` de sphinx-gallery: matplotlib, scikit-learn, scikit-image, networkx; y scripts
  sueltos: kubernetes, rich, xlsxwriter): el docstring inicial y los bloques de comentarios que siguen a un
  separador (`# %%` o una línea de `#####`) son texto en rst; el resto es código.
- Docstrings (`kind: docstrings`, rapidfuzz): la documentación publicada sale de los docstrings (numpydoc). Cada
  función o clase pública con docstring pasa a ser una sección con su firma.
"""

import ast
import inspect
import json
import re

from .codeblocks import ConsoleMode
from .pandoc_tree import Normalized, normalize_md, normalize_rst

# --- Notebooks ----------------------------------------------------------------------------------------


def notebook_to_md(raw: str) -> str:
    nb = json.loads(raw)
    lang = (nb.get("metadata", {}).get("kernelspec", {}).get("language") or "python").lower()
    parts = []
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else cell.get("source", "")
        if not src.strip():
            continue
        if cell.get("cell_type") == "markdown":
            parts.append(src.strip("\n"))
        elif cell.get("cell_type") == "code":
            fence = "~~~~" if "```" in src else "```"
            parts.append(f"{fence}{lang}\n{src.strip(chr(10))}\n{fence}")
    return "\n\n".join(parts) + "\n"


def normalize_notebook(raw: str, console_mode: ConsoleMode = "outputs_as_comments") -> Normalized:
    return normalize_md(notebook_to_md(raw), console_mode)


# --- Galerías de ejemplos -------------------------------------------------------------------------------

SEPARATOR = re.compile(r"^#\s*%%(.*)$|^#{20,}\s*$")  # `# %% Título` o una línea de #####
GALLERY_CONFIG = re.compile(r"^#\s*sphinx_gallery_\w+")  # `# sphinx_gallery_thumbnail_number = 2`
LICENSE = re.compile(r"^#\s*(SPDX-License-Identifier|Copyright|Licensed under|License:|Authors?:)", re.I)


def _code_block(lines: list[str]) -> list[str]:
    lines = [l for l in lines if not LICENSE.match(l)]
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    if all(not l.strip() or l.lstrip().startswith("#") for l in lines):
        return []  # vacío o solo comentarios: no es un ejemplo de código
    return ["", ".. code-block:: python", "", *("   " + l if l.strip() else "" for l in lines), ""]


def _comment_text(lines: list[str]) -> list[str]:
    """Bloque de comentarios como texto rst (`# texto` → `texto`)."""
    out = [re.sub(r"^#\s?", "", l) for l in lines if not LICENSE.match(l)]
    return ["", *out, ""]


def gallery_to_rst(src: str) -> str:
    lines = [l for l in src.split("\n") if not GALLERY_CONFIG.match(l) and not l.startswith("#!")]
    out: list[str] = []
    # Docstring inicial (título y explicación del ejemplo, en rst).
    try:
        tree = ast.parse(src)
        first = tree.body[0] if tree.body else None
        if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)):
            out += [inspect.cleandoc(first.value.value), ""]
            # Se recortan las líneas del docstring (contando en el original, antes de sacar la configuración).
            lines = [l for l in src.split("\n")[first.end_lineno:] if not GALLERY_CONFIG.match(l)]
    except SyntaxError:
        pass

    code: list[str] = []
    i = 0
    while i < len(lines):
        m = SEPARATOR.match(lines[i])
        if not m:
            code.append(lines[i])
            i += 1
            continue
        out += _code_block(code)
        code = []
        title = (m.group(1) or "").strip()
        if title:
            out += ["", title, "-" * len(title), ""]
        i += 1
        comments = []
        while i < len(lines) and lines[i].startswith("#") and not SEPARATOR.match(lines[i]):
            comments.append(lines[i])
            i += 1
        out += _comment_text(comments)
    out += _code_block(code)
    return "\n".join(out)


def normalize_gallery(src: str, console_mode: ConsoleMode = "outputs_as_comments") -> Normalized:
    return normalize_rst(gallery_to_rst(src), console_mode)


# --- Docstrings -----------------------------------------------------------------------------------

def _module_name(path: str) -> str:
    """`src/rapidfuzz/distance/Levenshtein_py.py` → `rapidfuzz.distance.Levenshtein`."""
    parts = path.removesuffix(".py").split("/")
    if "src" in parts:
        parts = parts[parts.index("src") + 1:]
    parts[-1] = parts[-1].removesuffix("_py")
    return ".".join(p for p in parts if p != "__init__")


def docstrings_to_rst(src: str, path: str) -> str:
    module = _module_name(path)
    tree = ast.parse(src)
    # Niveles de título en orden de aparición: módulo (=), función/clase (~); numpydoc usa (-) adentro.
    out = [module, "=" * len(module), ""]
    if (doc := ast.get_docstring(tree)):
        out += [doc, ""]

    def add(node, prefix: str) -> None:
        doc = ast.get_docstring(node)
        if not doc or node.name.startswith("_"):
            return
        name = f"{prefix}{node.name}"
        signature = name
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            signature += f"({ast.unparse(node.args)})"
        title = f"``{signature}``"  # como código: `*` y `**kwargs` no se leen como énfasis
        out.extend([title, "~" * len(title), "", doc, ""])
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(child, f"{name}.")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(node, f"{module}.")
    return "\n".join(out)


def normalize_docstrings(src: str, path: str, console_mode: ConsoleMode = "outputs_as_comments") -> Normalized:
    return normalize_rst(docstrings_to_rst(src, path), console_mode)
