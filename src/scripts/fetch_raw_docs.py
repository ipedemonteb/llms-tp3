"""Descarga el dataset crudo: README de PyPI + docs del repo de cada librería, y mide su calidad.

Uso:
    uv run python src/scripts/fetch_raw_docs.py                         # todo el pool de candidatas
    uv run python src/scripts/fetch_raw_docs.py --packages requests pypdf

Salida: data/raw/<paquete>/{manifest.json, README.*, <ruta original de la doc>}
        data/raw/index.json (estado y métricas por librería)
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from libraries import CANDIDATES, IMPORT_NAMES, REPO_OVERRIDES, SOURCES

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
INDEX_PATH = RAW_DIR / "index.json"

DOC_DIRS = ["docs", "doc", "documentation"]
# .txt: django y lxml escriben reST así; .mdx: docusaurus (spacy, hydra, mlflow).
DOC_EXTS = {".md": "md", ".markdown": "md", ".mdx": "md", ".rst": "rst", ".txt": "rst", ".ipynb": "ipynb"}
RST_HEADING = re.compile(r"^([=\-~^\"'`#*+])\1{3,}\s*$")

# Se guarda todo lo que hay en la carpeta de docs salvo lo excluido. Las palabras clave solo
# asignan el kind (orden = prioridad si coincide con varios; por defecto "guide").
KIND_RULES = [
    ("migration", ["migration", "migrating", "upgrading", "upgrade", "porting", "transition", "deprecat"]),
    ("quickstart", ["quickstart", "quick_start", "quick-start", "getting_started", "getting-started",
                    "gettingstarted", "first_steps", "first-steps"]),
    ("tutorial", ["tutorial"]),
    ("examples", ["examples", "example", "cookbook", "recipes", "gallery"]),
    ("guide", ["user_guide", "userguide", "guide", "usage", "howto", "how-to", "how_to"]),
]
# Subcadenas de la ruta que la descartan.
EXCLUDE = ["changelog", "change_log", "change-log", "changes", "history", "release", "whatsnew", "whats_new",
           "whats-new", "news", "timeline", "contributing", "contributors", "development", "internals", "_static",
           "_templates", "_build", "license", "authors", "governance", "code_of_conduct", "code-of-conduct",
           "sponsor", "funding", "roadmap", "install", "used-by", "used_by", "_links"]
# Carpetas (componentes de la ruta) que la descartan.
EXCLUDE_DIRS = {"api", "apidoc", "reference", "references", "generated", "autosummary", "dev",
                "developer", "developers", "community", "internal", "_ext", "sphinxext", "tests", "test",
                "devel", "api-reference", "api_reference", "package_reference", "versions", "blog",
                # Propuestas de diseño (NumPy Enhancement Proposals, scikit-image SKIPs, design docs).
                "neps", "skips", "design"}
# Nombres de archivo (sin extensión) que la descartan.
EXCLUDE_STEMS = {"api", "api_reference", "api-reference", "genindex", "glossary", "credits", "thanks", "conf", "menu",
                 "__init__", "setup"}
# Traducciones: se queda solo la doc en inglés.
LANG_DIRS = {"ar", "az", "bg", "bn", "cn", "cs", "de", "el", "em", "es", "fa", "fr", "he", "hi", "hu", "id",
             "it", "ja", "jp", "ka", "ko", "ms", "nl", "pl", "pt", "pt-br", "ro", "ru", "sv", "ta", "te", "th",
             "tm", "tr", "uk", "ur", "vi", "yo", "zh", "zh-cn", "zh-hans", "zh-hant", "zh-tw"}
AUTODOC = re.compile(r"^\s*\.\.\s+auto\w+::|^:::\s|^\[\[autodoc\]\]", re.M)  # Sphinx, mkdocstrings, HF doc-builder
MAX_FILE_BYTES = 2_000_000  # archivos más grandes suelen ser datos o docs generadas

REPO_RE = re.compile(r"https?://(?:www\.)?(github\.com|gitlab\.com)/([\w.-]+)/([\w.-]+)", re.I)
NOT_OWNERS = {"sponsors", "orgs", "users", "apps", "marketplace", "features"}


def pypi_info(package: str) -> dict:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json", timeout=30) as resp:
        return json.load(resp)["info"]


def find_repo(package: str, info: dict) -> str | None:
    """Devuelve 'host/owner/repo' a partir de los links de PyPI."""
    if package in REPO_OVERRIDES:
        return REPO_OVERRIDES[package]
    urls = list((info.get("project_urls") or {}).items())
    if info.get("home_page"):
        urls.append(("homepage", info["home_page"]))
    # Primero los links que suelen ser el código fuente.
    urls.sort(key=lambda kv: 0 if re.search(r"source|repo|code|github|gitlab", kv[0], re.I) else 1)
    for _, url in urls:
        m = REPO_RE.match(url or "")
        if m and m.group(2).lower() not in NOT_OWNERS:
            repo = m.group(3).removesuffix(".git")
            return f"{m.group(1).lower()}/{m.group(2)}/{repo}"
    return None


def run(cmd: list[str], cwd: Path | None = None) -> str:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True, timeout=900).stdout.strip()


def classify(rel_path: str) -> str | None:
    """kind del archivo según su ruta relativa a la carpeta de docs, o None si no se guarda."""
    p = rel_path.lower()
    parts = p.split("/")
    stem = parts[-1].rsplit(".", 1)[0]
    dirs = set(parts[:-1])
    if any(x in p for x in EXCLUDE) or (EXCLUDE_DIRS | LANG_DIRS) & dirs or stem in EXCLUDE_STEMS:
        return None
    for kind, words in KIND_RULES:
        if any(w in p for w in words):
            return kind
    return "guide"


# ---------------------------------------------------------------------------------------------
# Medición: bloques de código Python y casos de uso (secciones con explicación + código).

MD_FENCE_PY = re.compile(r"^\s*(`{3,}|~{3,})\s*(\{(code-block|code|sourcecode)\}\s*)?\{?\s*"
                         r"(python3?|py3?|pycon|ipython3?)\b|^\s*(`{3,}|~{3,})\s*\{(ipython|doctest|testcode|"
                         r"jupyter-execute)\}", re.I)
MD_ANY_FENCE = re.compile(r"^\s*(`{3,}|~{3,})")
MD_HEADING = re.compile(r"^#{1,6}\s+\S")
REPL_PY = re.compile(r"^\s*>{1,3} +(import |from [\w.]+ import |[\w.]+\s*=[^=]|[\w.]+\()")  # ">>> x" o "> x"
# Sin lenguaje, Sphinx resalta como Python por defecto.
RST_DIRECTIVE = re.compile(r"^\s*\.\.\s+(code-block|code|sourcecode)::\s*(python3?|py3?|pycon|ipython3?)?\s*$"
                           r"|^\s*\.\.\s+(ipython|doctest|testcode|jupyter-execute|plot)::", re.I)
DOXYGEN_PY = re.compile(r"^\s*@code\{\.py\}")  # tutoriales de opencv
DOCTEST = re.compile(r"^\s*>>> ")
RST_LITERAL = re.compile(r"^(?!\s*\.\.\s).*::\s*$")  # párrafo que termina en "::" (no directiva)
PY_HINT = re.compile(r"^\s*(>>> |import \w|from [\w.]+ import |def \w|class \w|@\w|[\w.\[\]'\"]+\s*=[^=]|[\w.]+\()",
                     re.M)
MIN_PROSE_CHARS = 80  # explicación mínima para que una sección cuente como caso de uso


def _indented_block_end(lines: list[str], i: int) -> int:
    """Índice siguiente al bloque indentado que empieza debajo de lines[i]."""
    base = len(lines[i]) - len(lines[i].lstrip())
    j = i + 1
    while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > base):
        j += 1
    return j


def code_spans(lines: list[str], fmt: str) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """(bloques Python, todos los bloques de código) como rangos [inicio, fin) de líneas."""
    py, every, i = [], [], 0
    while i < len(lines):
        line, start = lines[i], i
        if fmt == "md" and MD_ANY_FENCE.match(line):
            fence = MD_ANY_FENCE.match(line).group(1)
            i += 1
            while i < len(lines) and not lines[i].strip().startswith(fence):
                i += 1
            i += 1
            every.append((start, i))
            # Sesiones de REPL en fences "console"/"text" también son uso de Python.
            if MD_FENCE_PY.match(line) or any(REPL_PY.match(l) for l in lines[start + 1:i - 1]):
                py.append((start, i))
        elif DOXYGEN_PY.match(line):
            i += 1
            while i < len(lines) and "@endcode" not in lines[i]:
                i += 1
            i += 1
            py.append((start, i))
            every.append((start, i))
        # Los .md con MyST pueden embeber reST ({eval-rst}), así que ahí también se buscan directivas.
        elif RST_DIRECTIVE.match(line):
            i = _indented_block_end(lines, i)
            py.append((start, i))
            every.append((start, i))
        elif fmt == "rst" and RST_LITERAL.match(line):
            i = _indented_block_end(lines, i)
            every.append((start + 1, i))
            if PY_HINT.search("\n".join(lines[start + 1:i])):
                py.append((start + 1, i))
        elif DOCTEST.match(line):
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i][0] in " \t>."):
                i += 1
            py.append((start, i))
            every.append((start, i))
        else:
            i += 1
    return py, every


def section_starts(lines: list[str], fmt: str, code: list[tuple[int, int]]) -> list[int]:
    in_code = set()
    for s, e in code:
        in_code.update(range(s, e))
    starts = [0]
    for i, line in enumerate(lines):
        if i in in_code:
            continue
        if fmt == "md" and MD_HEADING.match(line):
            starts.append(i)
        elif (fmt == "rst" and i + 1 < len(lines) and line.strip() and not RST_HEADING.match(line)
              and RST_HEADING.match(lines[i + 1]) and len(lines[i + 1].strip()) >= len(line.strip()) - 2):
            starts.append(i)
    return sorted(set(starts))


def notebook_as_md(raw: str) -> str:
    """Notebook → Markdown equivalente (celdas de texto + fences de código) para medirlo igual."""
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    parts = []
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown":
            parts.append(src)
        elif cell.get("cell_type") == "code" and src.strip():
            parts.append(f"```python\n{src}\n```")
    return "\n\n".join(parts)


def measure(text: str, fmt: str, names: list[str]) -> dict:
    """Bloques Python, casos de uso y bloques que usan la librería en un archivo."""
    if fmt == "py":
        # Un ejemplo .py es un caso de uso entero (la explicación está en la doc o en sus comentarios).
        code = text
        uses = bool(_uses_lib(code, names))
        return {"py_blocks": 1, "use_cases": 1, "lib_blocks": int(uses)}
    if fmt == "ipynb":
        text, fmt = notebook_as_md(text), "md"
    lines = text.splitlines()
    py, every = code_spans(lines, fmt)
    starts = section_starts(lines, fmt, every) + [len(lines)]
    code_lines = set()
    for s, e in every:
        code_lines.update(range(s, e))

    use_cases = 0
    for s, e in zip(starts, starts[1:]):
        has_py = any(s <= ps < e for ps, _ in py)
        prose = sum(len(lines[i].strip()) for i in range(s, e) if i not in code_lines)
        if has_py and prose >= MIN_PROSE_CHARS:
            use_cases += 1
    lib_blocks = sum(1 for s, e in py if _uses_lib("\n".join(lines[s:e]), names))
    return {"py_blocks": len(py), "use_cases": use_cases, "lib_blocks": lib_blocks}


def _uses_lib(code: str, names: list[str]) -> bool:
    return any(re.search(rf"(?<![\w.]){re.escape(n)}\b", code) for n in names)


def count_py_blocks(text: str, fmt: str) -> int:
    return measure(text, fmt, [])["py_blocks"]


def import_names(package: str) -> list[str]:
    if package in IMPORT_NAMES:
        return IMPORT_NAMES[package]
    base = package.lower().removeprefix("python-").removeprefix("py-")
    return sorted({base.replace("-", "_"), base.split("-")[0], package.replace("-", "_")})


# ---------------------------------------------------------------------------------------------
# Descarga.

HEPTAPOD = "foss.heptapod.net/"


class _Done(Exception):
    """Corta el camino de git cuando el repo se bajó por otra vía."""


def _urlopen_retry(url: str, timeout: int, attempts: int = 4):
    """Heptapod devuelve 500 intermitentes; se reintenta con espera creciente."""
    for i in range(attempts):
        try:
            return urllib.request.urlopen(url, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code < 500 or i == attempts - 1:
                raise
            time.sleep(2 ** i)


def heptapod_download(repo: str, doc_paths: list[str], dest: Path) -> str:
    """Repos Mercurial en Heptapod (no se pueden clonar con git): baja un tar de las carpetas de
    docs en el último commit de la rama default, vía la API de GitLab. Devuelve el commit."""
    project = repo.removeprefix(HEPTAPOD)
    api = f"https://{HEPTAPOD}api/v4/projects/{urllib.parse.quote(project, safe='')}"
    with _urlopen_retry(f"{api}/repository/branches/branch%2Fdefault", timeout=30) as resp:
        commit = json.load(resp)["commit"]["id"]
    for d in doc_paths:
        name = project.rsplit("/", 1)[-1]
        url = f"https://{repo}/-/archive/{commit}/{name}-{commit}.tar.gz?path={urllib.parse.quote(d)}"
        try:
            resp = _urlopen_retry(url, timeout=300, attempts=2)
        except urllib.error.HTTPError:
            continue  # Heptapod responde 500 (no 404) si la carpeta no existe
        with resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                # Se descarta la carpeta raíz del tar (<proyecto>-<sha>-<path>/).
                parts = member.name.split("/", 1)
                if len(parts) < 2 or not member.isfile():
                    continue
                member.name = parts[1]
                tar.extract(member, dest, filter="data")
    return commit


def _has_conf(clone: Path, doc_dir: str) -> bool:
    """conf.py de Sphinx dentro de la carpeta de docs o en alguna carpeta padre (p. ej. pyarrow)."""
    d = clone / doc_dir
    if next(d.rglob("conf.py"), None):
        return True
    return any((parent / "conf.py").exists() for parent in d.parents if parent != clone and clone in parent.parents)


def readme_format(info: dict) -> str | None:
    text = info.get("description") or ""
    if len(text.strip()) < 50 or text.strip() == "UNKNOWN":
        return None
    ctype = (info.get("description_content_type") or "").lower()
    if "markdown" in ctype:
        return "md"
    if "rst" in ctype:
        return "rst"
    # Sin tipo declarado: PyPI lo interpreta como reST, pero muchos son Markdown en realidad.
    if re.search(r"^```|^#{1,3} ", text, re.M):
        return "md"
    return "rst"


def strip_outputs(raw: str) -> str | None:
    """Notebook sin outputs (pueden traer imágenes en base64 de varios MB). None si no es JSON."""
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError:
        return None
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"], cell["execution_count"] = [], None
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def collect(clone: Path, base: str, exts: dict[str, str], out: Path, files: list[dict],
            exclude: list[str]) -> None:
    """Copia a out los archivos útiles de clone/base (con su ruta original) y los agrega a files."""
    for src in sorted((clone / base).rglob("*")):
        ext = src.suffix.lower()
        if not src.is_file() or ext not in exts or src.stat().st_size > MAX_FILE_BYTES:
            continue
        rel = src.relative_to(clone).as_posix()
        if any(f["path"] == rel for f in files):
            continue  # ya lo trajo otra carpeta (p. ej. docs/ y docs/examples/)
        kind = classify(src.relative_to(clone / base).as_posix())
        if kind is None or any(x in rel for x in exclude):
            continue
        fmt = exts[ext]
        text = src.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            continue  # vacío (p. ej. snippets placeholder)
        if ext == ".txt" and not any(RST_HEADING.match(l) for l in text.splitlines()):
            continue  # .txt que no es reST (requirements, notas sueltas)
        if fmt == "ipynb":
            text = strip_outputs(text)
            if text is None:
                continue
        if fmt in {"md", "rst"} and AUTODOC.search(text) and count_py_blocks(text, fmt) == 0:
            continue  # referencia generada desde docstrings, sin ejemplos
        if fmt == "py":
            kind = "examples"
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
        files.append({"path": rel, "kind": kind, "format": fmt})


def fetch(package: str, category: str) -> dict:
    entry = {"package": package, "category": category}
    out = RAW_DIR / package
    try:
        info = pypi_info(package)
    except Exception as e:
        return entry | {"status": "error", "reason": f"pypi: {e}"}

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    files = []
    fmt = readme_format(info)
    if fmt:
        name = f"README.{fmt}"
        (out / name).write_text(info["description"], encoding="utf-8")
        files.append({"path": name, "kind": "readme", "format": fmt})

    repo = find_repo(package, info)
    src_cfg = SOURCES.get(package, {})
    doc_paths = src_cfg.get("docs", DOC_DIRS)
    example_paths = src_cfg.get("examples", [])
    docstring_globs = src_cfg.get("docstrings", [])
    docstring_dirs = sorted({g.rsplit("/", 1)[0] for g in docstring_globs})
    manifest = {
        "package": package,
        "category": category,
        "repo": repo,
        "commit": None,
        "pypi_version": info.get("version"),
        "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "docs_tool": "none",
        "docs_dirs": [],
        "examples_dirs": [],
        "files": files,
    }
    status, reason = "ok", None

    if not repo:
        status, reason = "sin_repo", "PyPI no declara un repo de GitHub/GitLab"
    else:
        with tempfile.TemporaryDirectory() as tmp:
            clone = Path(tmp) / "repo"
            try:
                if repo.startswith(HEPTAPOD):
                    manifest["commit"] = heptapod_download(repo, doc_paths + example_paths, clone)
                    raise _Done
                run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", "--quiet",
                     f"https://{repo}", str(clone)])
                run(["git", "sparse-checkout", "set", "--no-cone",
                     *[f"/{d}/" for d in doc_paths + example_paths + docstring_dirs],
                     "/mkdocs.yml", "/mkdocs.yaml", "conf.py"], cwd=clone)
                manifest["commit"] = run(["git", "rev-parse", "HEAD"], cwd=clone)
            except _Done:
                pass
            except OSError as e:
                status, reason = "error", f"descarga: {e}"
            except subprocess.CalledProcessError as e:
                status, reason = "error", f"git: {e.stderr.strip()[-300:]}"
            except subprocess.TimeoutExpired:
                status, reason = "error", "git: timeout"

            if status == "ok":
                doc_dirs = [d for d in doc_paths if (clone / d).is_dir()]
                ex_dirs = [d for d in example_paths if (clone / d).is_dir()]
                manifest["docs_dirs"], manifest["examples_dirs"] = doc_dirs, ex_dirs
                if (clone / "mkdocs.yml").exists() or (clone / "mkdocs.yaml").exists():
                    manifest["docs_tool"] = "mkdocs"
                elif any(_has_conf(clone, d) for d in doc_dirs):
                    manifest["docs_tool"] = "sphinx"

                exclude = src_cfg.get("exclude", [])
                for d in doc_dirs:
                    collect(clone, d, DOC_EXTS, out, files, exclude)
                for d in ex_dirs:
                    collect(clone, d, {".py": "py"}, out, files, exclude)
                # Librerías cuya doc es autodoc: la fuente real son los docstrings de estos módulos.
                for pattern in docstring_globs:
                    for src in sorted(clone.glob(pattern)):
                        rel = src.relative_to(clone).as_posix()
                        dst = out / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(src, dst)
                        files.append({"path": rel, "kind": "docstrings", "format": "py"})

                if not doc_dirs and not ex_dirs and not docstring_globs:
                    status, reason = "sin_docs", f"el repo no tiene {'/'.join(doc_paths)}"
                elif len(files) <= 1:
                    status, reason = "sin_docs", "ningún archivo de docs pasó las reglas de selección"

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return entry | {"status": status, "reason": reason, "repo": repo, "docs_tool": manifest["docs_tool"],
                    **metrics(package, out, files)}


def metrics(package: str, out: Path, files: list[dict]) -> dict:
    names = import_names(package)
    m = {"n_files": len(files), "doc_kb": 0, "kinds": {}, "formats": {},
         "py_blocks": {"readme": 0, "docs": 0}, "use_cases": {"readme": 0, "docs": 0}, "lib_blocks": 0}
    for f in files:
        path = out / f["path"]
        text = path.read_text(encoding="utf-8", errors="replace")
        part = "readme" if f["kind"] == "readme" else "docs"
        # Los docstrings (numpydoc/reST) se miden como reST: secciones "Examples" con doctests.
        r = measure(text, "rst" if f["kind"] == "docstrings" else f["format"], names)
        m["py_blocks"][part] += r["py_blocks"]
        m["use_cases"][part] += r["use_cases"]
        m["lib_blocks"] += r["lib_blocks"]
        m["kinds"][f["kind"]] = m["kinds"].get(f["kind"], 0) + 1
        m["formats"][f["format"]] = m["formats"].get(f["format"], 0) + 1
        if part == "docs":
            m["doc_kb"] += path.stat().st_size
    m["doc_kb"] = round(m["doc_kb"] / 1024)
    total = sum(m["py_blocks"].values())
    m["lib_ratio"] = round(m["lib_blocks"] / total, 2) if total else 0.0
    return m


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", nargs="*", help="solo estos paquetes")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    by_pkg = {p: cat for cat, ps in CANDIDATES.items() for p in ps}
    targets = {p: by_pkg.get(p, "sin_categoria") for p in (args.packages or by_pkg)}

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    index = json.loads(INDEX_PATH.read_text()) if INDEX_PATH.exists() else {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, p, c): p for p, c in targets.items()}
        for fut in as_completed(futures):
            e = fut.result()
            index[e["package"]] = e
            uc = e.get("use_cases", {})
            print(f"{e['status']:9} {e['package']:24} files={e.get('n_files', 0):4} kb={e.get('doc_kb', 0):6} "
                  f"use_cases={uc.get('docs', 0)}+{uc.get('readme', 0)} lib={e.get('lib_ratio', 0)}  "
                  f"{e.get('reason') or ''}", flush=True)

    INDEX_PATH.write_text(json.dumps(dict(sorted(index.items())), indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
