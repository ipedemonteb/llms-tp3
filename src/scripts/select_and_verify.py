"""Elige las 100 librerías finales del dataset crudo y verifica su integridad (plan §6, pasos 4 y 5).

Uso (después de fetch_raw_docs.py):
    uv run python src/scripts/select_and_verify.py

1. Umbral de calidad sobre la documentación propia (sin README): tamaño, bloques de código y casos de uso.
2. Exclusiones manuales (librerías que pasan el umbral pero no sirven para el RAG) y excepciones
   justificadas (reemplazos de §5.3 sin alternativa que pase el umbral).
3. Elección por rondas entre categorías: en cada ronda entra la más popular (descargas de PyPI) de cada
   categoría, hasta llegar a 100.
4. Borra de data/raw/ las descartadas y verifica manifests, archivos y cobertura (sale con 1 si algo falla).

Además de las 100 se conservan los EXTRAS (subconjunto de la stdlib): no compiten por un lugar ni pasan por
el umbral, pero se verifican igual.
"""

import json
import re
import shutil
import sys
from pathlib import Path

from fetch_raw_docs import INDEX_PATH, RAW_DIR, ROOT, import_names, metrics
from libraries import CANDIDATES, STDLIB

TARGET = 100
# Fuera de las 100 y del umbral: único reemplazo posible de deprecadas sin alternativa en PyPI.
EXTRAS = {STDLIB["package"]: "subconjunto de la stdlib (zoneinfo, unittest.mock, tomllib): reemplazo de pytz, "
                             "mock y toml"}
TOP_PYPI = ROOT / "data" / "top_pypi.json"  # https://hugovk.github.io/top-pypi-packages/

# Umbral de calidad (solo docs, el README no cuenta).
MIN_DOC_KB = 40
MIN_PY_BLOCKS = 40
MIN_USE_CASES = 25
MIN_PY_BLOCKS_DOCTEST = 100  # o muchos bloques: docs hechas de doctests con pocas secciones (dateutil)

# Pasan el umbral pero no sirven para corregir imports en código de usuario.
MANUAL_EXCLUDE = {
    "httpcore": "dependencia transitiva de httpx; nadie la importa directamente",
    "kombu": "capa de transporte interna de celery",
    "narwhals": "capa de compatibilidad para autores de librerías, no para usuarios",
    "soupsieve": "dependencia de beautifulsoup4; su doc es referencia de selectores CSS",
    "pyinstaller": "herramienta de empaquetado por línea de comandos (no se importa)",
    "gunicorn": "servidor por línea de comandos; su doc es de configuración",
    "nox": "runner de tareas por línea de comandos",
}

# No pasan el umbral pero se incluyen: son el único reemplazo con doc real de un par de §5.3.
EXCEPTIONS = {
    "tenacity": "reemplazo de retrying; doc completa en un solo archivo (21 KB, 40 bloques)",
    "rapidfuzz": "reemplazo de fuzzywuzzy; su doc publicada es autodoc: se guardan los docstrings (24 ejemplos) "
                 "y el README con la API compatible",
}

# Deprecado → reemplazos indexados que lo cubren (alcanza con uno).
PAIRS = {
    "PyPDF2/PyPDF4": {"pypdf"},
    "fuzzywuzzy": {"rapidfuzz"},
    "pycrypto": {"pycryptodome", "cryptography"},
    "nose": {"pytest"},
    "retrying": {"tenacity"},
    "xlrd": {"openpyxl"},
    "python-jose": {"authlib", "pyjwt"},
    "passlib": {"cryptography"},  # hashing de contraseñas con Argon2id/Scrypt (bcrypt no tiene docs)
    "sklearn": {"scikit-learn"},
    "oauth2client": {"google-api-python-client"},  # su doc de auth usa google.oauth2 (google-auth no tiene docs)
    "pytz": {"python-stdlib"},  # zoneinfo
    "mock": {"python-stdlib"},  # unittest.mock
    "toml": {"python-stdlib"},  # tomllib
}
# Deprecados: nunca se indexan.
DEPRECATED = {"pypdf2", "pypdf4", "fuzzywuzzy", "pycrypto", "nose", "retrying", "xlrd", "python-jose",
              "passlib", "sklearn", "oauth2client", "toml", "mock", "pytz"}

KINDS = {"readme", "quickstart", "tutorial", "guide", "examples", "migration", "docstrings"}
FORMATS = {"md", "rst", "ipynb", "py"}
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name.lower())


def passes(e: dict) -> bool:
    blocks, uses = e.get("py_blocks", {}).get("docs", 0), e.get("use_cases", {}).get("docs", 0)
    return (e["status"] == "ok" and e.get("doc_kb", 0) >= MIN_DOC_KB and blocks >= MIN_PY_BLOCKS
            and (uses >= MIN_USE_CASES or blocks >= MIN_PY_BLOCKS_DOCTEST))


def fail_reason(e: dict) -> str:
    if e["status"] != "ok":
        return f"{e['status']}: {e.get('reason')}"
    return (f"doc insuficiente: {e.get('doc_kb', 0)} KB, {e.get('py_blocks', {}).get('docs', 0)} bloques, "
            f"{e.get('use_cases', {}).get('docs', 0)} casos de uso (sin README)")


def select(index: dict) -> list[str]:
    ranks = {norm(r["project"]): i + 1 for i, r in enumerate(json.loads(TOP_PYPI.read_text())["rows"])}
    for e in index.values():
        e["pypi_rank"] = ranks.get(norm(e["package"]))
        e["selected"], e["selection_reason"] = False, None
        if e["package"] in EXTRAS:
            e["selected"], e["selection_reason"] = True, f"extra fuera de las {TARGET}: {EXTRAS[e['package']]}"
        elif e["package"] in MANUAL_EXCLUDE:
            e["selection_reason"] = f"excluida a mano: {MANUAL_EXCLUDE[e['package']]}"
        elif e["package"] in EXCEPTIONS:
            e["selection_reason"] = f"excepción §5.3: {EXCEPTIONS[e['package']]}"
        elif not passes(e):
            e["selection_reason"] = fail_reason(e)

    chosen = [p for p in EXCEPTIONS if p in index]
    queues = {
        cat: sorted((p for p in pkgs if p in index and passes(index[p]) and p not in MANUAL_EXCLUDE
                     and p not in chosen), key=lambda p: index[p]["pypi_rank"] or 10**9)
        for cat, pkgs in CANDIDATES.items()
    }
    while len(chosen) < TARGET and any(queues.values()):
        for cat in CANDIDATES:
            if queues[cat] and len(chosen) < TARGET:
                chosen.append(queues[cat].pop(0))

    for p in chosen:
        e = index[p]
        e["selected"] = True
        if p not in EXCEPTIONS:
            e["selection_reason"] = f"pasa el umbral; puesto {e['pypi_rank']} en descargas de PyPI"
    for q in queues.values():
        for p in q:
            index[p]["selection_reason"] = "pasa el umbral, pero su categoría ya tenía librerías más populares"
    return chosen


def verify(index: dict, chosen: list[str]) -> list[str]:
    errors = []
    sel = set(chosen)
    if len(sel) != TARGET:
        errors.append(f"se eligieron {len(sel)} librerías, no {TARGET}")
    cats = {index[p]["category"] for p in sel}
    errors += [f"categoría sin librerías: {c}" for c in CANDIDATES if c not in cats]
    keep = sel | set(EXTRAS)
    errors += [f"par {dep} sin reemplazo indexado" for dep, repl in PAIRS.items() if not repl & keep]
    errors += [f"librería deprecada en el dataset: {p}" for p in sel if norm(p) in DEPRECATED]
    errors += [f"extra sin entrada en el índice: {p}" for p in EXTRAS if p not in index]

    on_disk = {d.name for d in RAW_DIR.iterdir() if d.is_dir()}
    errors += [f"falta la carpeta de {p}" for p in keep - on_disk]
    errors += [f"carpeta de una librería no elegida: {p}" for p in on_disk - keep]

    for p in sorted(keep & on_disk & index.keys()):
        out, e = RAW_DIR / p, index[p]
        try:
            m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as err:
            errors.append(f"{p}: manifest ilegible ({err})")
            continue
        if m["package"] != p or m["category"] != e["category"]:
            errors.append(f"{p}: el manifest no coincide con el índice")
        if p in EXTRAS:
            if not m.get("ref"):
                errors.append(f"{p}: falta el tag (ref) de la doc")
        elif not m.get("pypi_version"):
            errors.append(f"{p}: falta pypi_version")
        if not SHA_RE.match(m.get("commit") or ""):
            errors.append(f"{p}: commit inválido ({m.get('commit')})")

        listed = set()
        for f in m["files"]:
            path = out / f["path"]
            listed.add(f["path"])
            if f["kind"] not in KINDS or f["format"] not in FORMATS:
                errors.append(f"{p}: kind/format inválido en {f['path']}")
            if not path.is_file() or path.stat().st_size == 0:
                errors.append(f"{p}: archivo faltante o vacío {f['path']}")
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                errors.append(f"{p}: no es UTF-8 {f['path']}")
                continue
            if f["format"] == "ipynb":
                try:
                    json.loads(text)
                except json.JSONDecodeError:
                    errors.append(f"{p}: notebook inválido {f['path']}")
        extra = {f.relative_to(out).as_posix() for f in out.rglob("*") if f.is_file()} - listed - {"manifest.json"}
        errors += [f"{p}: archivo fuera del manifest {x}" for x in sorted(extra)]

        # Las métricas del índice tienen que salir de los archivos que hay en disco.
        fresh = metrics(p, out, m["files"])
        if any(fresh[k] != e.get(k) for k in ("py_blocks", "use_cases", "lib_blocks", "doc_kb")):
            errors.append(f"{p}: las métricas no coinciden con el índice")
        if e.get("status") != "ok":
            errors.append(f"{p}: estado {e.get('status')} ({e.get('reason')})")
        if p not in EXCEPTIONS and p not in EXTRAS and not passes(e):
            errors.append(f"{p}: elegida sin pasar el umbral")
        if fresh["lib_ratio"] < 0.1:
            errors.append(f"{p}: casi ningún bloque usa la librería ({import_names(p)}, {fresh['lib_ratio']})")
    return errors


def main() -> int:
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    chosen = select(index)

    for d in RAW_DIR.iterdir():
        if d.is_dir() and d.name not in chosen and d.name not in EXTRAS:
            shutil.rmtree(d)
    INDEX_PATH.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Elegidas: {len(chosen)}")
    for cat in CANDIDATES:
        ps = [p for p in chosen if index[p]["category"] == cat]
        print(f"  {cat:24} {len(ps):2}  {', '.join(ps)}")
    print(f"Extras: {', '.join(EXTRAS)}")

    errors = verify(index, chosen)
    print(f"\nVerificación: {'OK' if not errors else f'{len(errors)} errores'}")
    for err in errors:
        print(f"  ✗ {err}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
