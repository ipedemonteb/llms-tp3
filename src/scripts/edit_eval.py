"""Editar los ítems del dataset de evaluación sin tocar el JSONL a mano (docs/agents/dataset_eval.md §3.2).

Uso:
    uv run python src/scripts/edit_eval.py export   # data/eval/*.jsonl → data/eval/_edit/<archivo>/<id>.py + <id>.json
    # editar o agregar los .py / .json (el gold es paquete + archivo + número de bloque)
    uv run python src/scripts/edit_eval.py import   # de vuelta a los JSONL (recalcula el texto de cada gold)
    uv run python src/scripts/verify_eval.py        # y verificar; split_eval.py si se agregaron ítems

`data/eval/_edit/` está en .gitignore: la fuente versionada son los JSONL.
"""

import argparse
import json
import shutil

from tprag.eval.edit import export_items, import_items
from tprag.eval.verify import EVAL_DIR, EVAL_FILES
from tprag.normalize.store import load_normalized

EDIT_DIR = EVAL_DIR / "_edit"


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["export", "import"])
    args = parser.parse_args()

    if args.action == "export":
        if EDIT_DIR.exists():
            shutil.rmtree(EDIT_DIR)
        for name in EVAL_FILES:
            if (EVAL_DIR / name).exists():
                items = read_jsonl(EVAL_DIR / name)
                export_items(items, EDIT_DIR / name.removesuffix(".jsonl"))
                print(f"{name}: {len(items)} ítems → {EDIT_DIR / name.removesuffix('.jsonl')}")
        return 0

    docs = {(d.metadata["package"], d.metadata["source_path"]): (d.id, d.page_content) for d in load_normalized()}
    for name in EVAL_FILES:
        src = EDIT_DIR / name.removesuffix(".jsonl")
        if not src.exists():
            continue
        out = EVAL_DIR / name
        order = [it["id"] for it in read_jsonl(out)] if out.exists() else []
        items = import_items(src, docs, order)
        out.write_text("\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n", encoding="utf-8")
        print(f"{name}: {len(items)} ítems")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
