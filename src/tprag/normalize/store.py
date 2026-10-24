"""Normalización de una librería completa, escritura en data/normalized y carga (loaders.md §3.1).

Salida (un modo de `console_examples` por carpeta, §3.2):

    data/normalized/<modo>/
      index.jsonl                 una fila por archivo fuente (metadata; nada de metadata dentro del .md)
      <paquete>/<ruta original>.md   (README.md → README.md.md, docs/x.rst → docs/x.rst.md)

Todo el resto del pipeline obtiene los documentos solo con `load_normalized()`.
"""

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.documents import Document

from .codeblocks import ConsoleMode
from .includes import Corpus, included_paths, resolve_includes
from .pandoc_tree import Normalized, normalize_md, normalize_rst
from .prepass import clean_readme
from .sources import normalize_docstrings, normalize_gallery, normalize_notebook

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "data" / "raw"
NORMALIZED_DIR = ROOT / "data" / "normalized"
INCLUDE_ONLY_DIR = "docs_src"


@dataclass
class Entry:
    id: str
    package: str
    category: str
    source_path: str
    kind: str
    format: str
    md_path: str  # relativa a la carpeta del modo
    console_mode: str
    status: str  # ok | fallback (pandoc no pudo: se guarda el texto crudo) | empty
    n_blocks: int = 0
    block_langs: list[str] = field(default_factory=list)  # lenguaje de cada bloque, en orden
    dropped: dict[str, int] = field(default_factory=dict)
    included: list[str] = field(default_factory=list)  # archivos pegados por inclusiones
    unresolved: list[str] = field(default_factory=list)
    error: str = ""


def doc_id(package: str, source_path: str) -> str:
    return hashlib.sha1(f"{package}/{source_path}".encode()).hexdigest()[:16]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def normalize_package(pkg_dir: Path, console_mode: ConsoleMode = "outputs_as_comments"
                      ) -> list[tuple[Entry, str]]:
    """(entrada del índice, Markdown) de cada archivo de la librería, salvo los que otra página incluye."""
    manifest = json.loads(_read(pkg_dir / "manifest.json"))
    files = manifest["files"]
    corpus = Corpus(paths={f["path"] for f in files}, read=lambda p: _read(pkg_dir / p),
                    doc_dirs=manifest.get("docs_dirs", []) + manifest.get("examples_dirs", []))
    pages = {f["path"]: _read(pkg_dir / f["path"]) for f in files if f["format"] in ("md", "rst")}
    excluded = included_paths(pages, corpus)  # su contenido ya aparece en la página que los incluye (3b)

    out = []
    for f in files:
        path, fmt, kind = f["path"], f["format"], f["kind"]
        if kind == "included" or path in excluded:
            continue
        if INCLUDE_ONLY_DIR in path.split("/")[:-1]:
            # docs_src/ (fastapi, typer, sqlmodel) solo existe para que las páginas lo incluyan. Lo que ninguna
            # página nombra es la otra variante de un ejemplo incluido (p. ej. la versión sin `Annotated`, que el
            # sitio muestra como pestaña): como documento propio sería código sin explicación y duplicado.
            continue
        entry = Entry(id=doc_id(manifest["package"], path), package=manifest["package"],
                      category=manifest.get("category", ""), source_path=path, kind=kind, format=fmt,
                      md_path=f"{manifest['package']}/{path}.md", console_mode=console_mode, status="ok")
        raw = pages.get(path) or _read(pkg_dir / path)
        try:
            norm = _normalize(raw, path, fmt, kind, corpus, console_mode, entry)
            markdown = norm.markdown
            entry.n_blocks, entry.block_langs, entry.dropped = len(norm.block_langs), norm.block_langs, dict(norm.dropped)
        except Exception as e:  # noqa: BLE001  (pandoc o un archivo mal formado: se conserva el texto)
            markdown, entry.status, entry.error = raw, "fallback", f"{type(e).__name__}: {str(e)[:300]}"
        if not markdown.strip():
            entry.status = "empty"
        out.append((entry, markdown))
    return out


def _normalize(raw: str, path: str, fmt: str, kind: str, corpus: Corpus, mode: ConsoleMode,
               entry: Entry) -> Normalized:
    if fmt == "ipynb":
        return normalize_notebook(raw, mode)
    if fmt == "py":
        return normalize_docstrings(raw, path, mode) if kind == "docstrings" else normalize_gallery(raw, mode)
    if kind == "readme":
        raw = clean_readme(raw, fmt)
    res = resolve_includes(raw, fmt, path, corpus)
    entry.included, entry.unresolved = sorted(res.used), res.unresolved
    return (normalize_rst if fmt == "rst" else normalize_md)(res.text, mode)


# --- Escritura y carga --------------------------------------------------------------------------------

def mode_dir(console_mode: str, out_dir: Path = NORMALIZED_DIR) -> Path:
    out_dir = out_dir.resolve()
    if out_dir == RAW_DIR or RAW_DIR in out_dir.parents:
        raise ValueError(f"la salida no puede estar dentro de {RAW_DIR}")  # el crudo nunca se toca
    return out_dir / console_mode


def write_package(results: list[tuple[Entry, str]], package: str, base: Path) -> None:
    """Reemplaza los .md de la librería en `base` (carpeta del modo)."""
    pkg_out = base / package
    if pkg_out.exists():
        for p in sorted(pkg_out.rglob("*.md"), reverse=True):
            p.unlink()
    for entry, markdown in results:
        if entry.status == "empty":
            continue
        dst = base / entry.md_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(markdown, encoding="utf-8", newline="\n")


def write_index(entries: list[Entry], base: Path, packages: set[str]) -> None:
    """Actualiza index.jsonl: reemplaza las filas de `packages` y conserva las demás."""
    index = base / "index.jsonl"
    keep = []
    if index.exists():
        keep = [l for l in index.read_text(encoding="utf-8").splitlines()
                if l.strip() and json.loads(l)["package"] not in packages]
    rows = keep + [json.dumps(asdict(e), ensure_ascii=False) for e in entries]
    rows.sort(key=lambda r: (json.loads(r)["package"], json.loads(r)["source_path"]))
    index.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")


def load_normalized(console_mode: str = "outputs_as_comments", packages: list[str] | None = None,
                    out_dir: Path = NORMALIZED_DIR) -> list[Document]:
    """Documentos normalizados con su metadata. Es la única forma en que el resto del pipeline los lee."""
    base = out_dir / console_mode
    docs = []
    for line in (base / "index.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e["status"] == "empty" or (packages and e["package"] not in packages):
            continue
        text = (base / e["md_path"]).read_text(encoding="utf-8")
        meta = {k: e[k] for k in ("id", "package", "category", "source_path", "kind", "format", "md_path",
                                  "console_mode", "status", "n_blocks")}
        docs.append(Document(page_content=text, metadata=meta, id=e["id"]))
    return docs


def summarize(entries: list[Entry]) -> dict:
    """Totales para el reporte de la corrida."""
    langs, dropped = Counter(), Counter()
    for e in entries:
        langs.update(e.block_langs)
        dropped.update(e.dropped)
    return {
        "files": len(entries), "status": dict(Counter(e.status for e in entries)),
        "blocks": sum(e.n_blocks for e in entries), "block_langs": dict(langs.most_common()),
        "dropped": dict(dropped.most_common()),
        "fallback": [f"{e.package}/{e.source_path}: {e.error}" for e in entries if e.status == "fallback"],
        "unresolved_includes": [u for e in entries for u in e.unresolved],
    }
