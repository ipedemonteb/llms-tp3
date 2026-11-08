"""Asigna el campo `split` (dev/test) a los ítems de data/eval/ (docs/agents/dataset_eval.md §2.4, §3.7).

Uso:
    uv run python src/scripts/split_eval.py            # reescribe el campo `split` de cada JSONL
    uv run python src/scripts/split_eval.py --dry-run  # solo muestra el resumen

Semilla fija (tprag/eval/split.py): con los mismos ítems da el mismo split. Correrlo después de agregar ítems.
"""

import argparse
import json
from collections import Counter

from tprag.eval.split import assign_splits, group_key
from tprag.eval.verify import EVAL_DIR, EVAL_FILES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = {name: [json.loads(l) for l in (EVAL_DIR / name).read_text(encoding="utf-8").splitlines() if l.strip()]
             for name in EVAL_FILES if (EVAL_DIR / name).exists()}
    items = [it for its in files.values() for it in its]
    splits = assign_splits(items)

    summary = Counter((it["subset"], it.get("negative_kind", ""), splits[it["id"]]) for it in items)
    for subset, kind in sorted({(s, k) for s, k, _ in summary}):
        label = f"{subset}" + (f" ({kind})" if kind else "")
        print(f"{label:<45} dev {summary[(subset, kind, 'dev')]:>3}   test {summary[(subset, kind, 'test')]:>3}")
    groups = Counter((group_key(it), splits[it["id"]]) for it in items if not group_key(it).startswith("item:"))
    print("\ngrupos por reemplazo:")
    for (g, s), n in sorted(groups.items()):
        print(f"  {g:<45} {s:<5} {n}")

    if not args.dry_run:
        for name, its in files.items():
            lines = [json.dumps({**it, "split": splits[it["id"]]}, ensure_ascii=False) for it in its]
            (EVAL_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nsplit escrito en {', '.join(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
