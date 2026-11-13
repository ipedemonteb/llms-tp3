"""Incorpora los sintéticos de test escritos por los agentes (plan_cierre.md §2.2-2.4). NO USA la API de OpenAI: solo
consulta PyPI (gratis, con caché en data/validate/) para confirmar que la librería inventada no existe.

Uso:
    uv run python src/scripts/add_synthetic_test.py --outputs <carpeta con 000.py ...> --dry-run   # solo controles
    uv run python src/scripts/add_synthetic_test.py --outputs <carpeta>                            # agrega a synthetic.jsonl

Cada archivo `<INDEX>.py` corresponde al bloque INDEX de data/eval/synthetic_test_sample.json y empieza con
`# TASK: ...` y `# INVENTED: <módulo>`. Controles por ítem:
- parsea con `ast`, no nombra la librería real y no usa ningún nombre prohibido (`forbidden` de la muestra);
- comparte menos de 3 identificadores de la API con el gold (`shared_names`);
- el módulo inventado da `not_found` en el validador; si existe en PyPI se renombra con el sufijo `qx`.
Los ítems que fallan algún control no se agregan (se listan para regenerarlos). Los que pasan entran con
`split: test`, sin tocar los de dev. NO correr split_eval.py después (plan_experimentos_indice.md §2, aviso).
"""

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

from libraries import IMPORT_NAMES
from tprag.eval.synthetic import shared_names
from tprag.eval.verify import EVAL_DIR
from tprag.validate.pypi import CachedPyPI
from tprag.validate.validate import validate

AUTHOR, PROMPT_ID, DATE = "claude-haiku-5-5", "synthetic_v2", "2026-11-13"
MAX_SHARED = 2


def _real_modules(package: str) -> list[str]:
    """Paquete y módulo de import. Los alias de IMPORT_NAMES (`app`, `page`, `Image`) son demasiado genéricos; los
    que usa el bloque ya están en `forbidden`."""
    module = IMPORT_NAMES.get(package, [package.replace("-", "_")])[0]
    return sorted({package, module})


def _modules(code: str, module: str, fetch) -> dict[str, str]:
    """Estado de cada import del módulo inventado. El validador reporta cada submódulo aparte (`mod.io`, `mod.core`),
    y como en dev (syn-django-03) cada uno es un target."""
    root = module.split(".")[0]
    return {r.module: r.status for r in validate(code, fetch) if r.module.split(".")[0] == root}


def _status(code: str, module: str, fetch) -> str:
    statuses = set(_modules(code, module, fetch).values())
    return statuses.pop() if len(statuses) == 1 else ("missing" if not statuses else "mixto: " + str(statuses))


def check(index: int, block: dict, path: Path, fetch) -> tuple[dict | None, list[str], dict]:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("# SKIP"):
        return None, ["SKIP"], {}
    lines = text.split("\n")
    task = re.match(r"#\s*TASK:\s*(.+)", lines[0])
    invented = re.match(r"#\s*INVENTED:\s*([\w.]+)", lines[1]) if len(lines) > 1 else None
    if not task or not invented:
        return None, ["cabecera TASK/INVENTED"], {}
    code, module = "\n".join(lines[2:]).lstrip("\n"), invented.group(1)
    errors, info = [], {}
    try:
        ast.parse(code)
    except SyntaxError as e:
        return None, [f"no parsea: {e.msg} (línea {e.lineno})"], {}
    # en imports o identificadores es error; en comentarios o strings solo aviso (requests, click, rich, sh, invoke
    # son también palabras comunes y hay que mirarlo a mano)
    tree = ast.parse(code)
    code_idents = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    code_idents |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    code_idents |= {p for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                    for a in n.names for p in a.name.split(".")}
    code_idents |= {p for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
                    for p in n.module.split(".")}
    for real in _real_modules(block["package"]):
        if real.split(".")[0] in code_idents:
            errors.append(f"usa la librería real: {real}")
        elif re.search(rf"\b{re.escape(real.lower())}\b", code.lower()):
            info.setdefault("mentions", []).append(real)
    # los imports de la stdlib no cuentan (`from urllib.parse import ...` no copia `feedparser.parse`)
    body = "\n".join(line for line in code.split("\n")
                     if not ((m := re.match(r"\s*(?:from|import)\s+(\w+)", line))
                             and m.group(1) in sys.stdlib_module_names))
    idents = set(re.findall(r"[A-Za-z_]\w*", body))
    used = [n for n in block["forbidden"] if (n in idents if re.fullmatch(r"[A-Za-z_]\w*", n) else n in body)]
    if used:
        errors.append(f"usa nombres prohibidos: {used}")
    shared = shared_names(code, block["code"], module)
    info["shared"] = shared
    if len(shared) > MAX_SHARED:
        errors.append(f"comparte {len(shared)} nombres con el gold: {shared}")
    status = _status(code, module, fetch)
    if status == "ok":  # el nombre inventado existe en PyPI: se renombra
        root = module.split(".")[0]
        info["renamed_from"] = root
        code = re.sub(rf"\b{re.escape(root)}\b", root + "qx", code)
        module = root + "qx" + module[len(root):]
        status = _status(code, module, fetch)
    if status != "not_found":
        errors.append(f"el validador da {status} para {module}")
    n_lines = len(code.strip().split("\n"))
    info["n_lines"] = n_lines
    gold = [{k: block[k] for k in ("doc_id", "package", "source_path", "block", "code")}]
    item = {
        "subset": "synthetic", "task": task.group(1).strip(), "code": code,
        "targets": [{"module": m, "expected_status": "not_found", "acceptable_packages": [block["package"]],
                     "written_for": None, "gold": gold} for m in sorted(_modules(code, module, fetch))],
        "provenance": {"author": AUTHOR, "prompt_id": PROMPT_ID, "date": DATE, "reviewed_by": None,
                       "sample_file": "synthetic_test_sample.json", "sample_index": index,
                       **({"renamed_from": info["renamed_from"]} if "renamed_from" in info else {})},
        "split": "test",
    }
    return item, errors, info


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    blocks = json.loads((EVAL_DIR / "synthetic_test_sample.json").read_text(encoding="utf-8"))["blocks"]
    synthetic = EVAL_DIR / "synthetic.jsonl"
    existing = [json.loads(line) for line in synthetic.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(it.get("split") == "test" for it in existing):
        print("synthetic.jsonl ya tiene sintéticos de test: no se agrega nada")
        return 1
    fetch = CachedPyPI()

    ok, failed, missing = [], {}, []
    shared_hist = Counter()
    for i, block in enumerate(blocks):
        path = args.outputs / f"{i:03d}.py"
        if not path.exists():
            missing.append(i)
            continue
        item, errors, info = check(i, block, path, fetch)
        if errors:
            failed[i] = errors
            continue
        shared_hist[min(len(info["shared"]), 3)] += 1
        if info.get("mentions"):
            print(f"  aviso {i:03d} {block['package']}: nombra {info['mentions']} en un comentario o string")
        ok.append(item)

    per_package = Counter()
    for item in ok:
        package = item["targets"][0]["acceptable_packages"][0]
        per_package[package] += 1
        item["id"] = f"syn-{package}-{per_package[package]:02d}"
    ok = [{"id": it.pop("id"), **it} for it in ok]

    print(f"{len(ok)} ok, {len(failed)} con errores, {len(missing)} sin archivo {missing or ''}")
    print(f"nombres compartidos con el gold (0/1/2): {[shared_hist[k] for k in range(3)]}; "
          f"renombrados (qx): {sum('renamed_from' in it['provenance'] for it in ok)}")
    for i, errors in failed.items():
        print(f"  {i:03d} {blocks[i]['package']}: " + "; ".join(errors))
    if args.dry_run:
        return 0
    ids = {it["id"] for it in existing}
    clash = [it["id"] for it in ok if it["id"] in ids]
    if clash:
        print(f"ids repetidos con el dataset: {clash}")
        return 1
    with synthetic.open("a", encoding="utf-8") as f:
        for item in ok:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"-> {len(ok)} ítems agregados a {synthetic} con split test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
