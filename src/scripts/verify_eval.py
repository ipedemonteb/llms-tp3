"""Verifica el dataset de evaluación de data/eval/ (docs/agents/dataset_eval.md §3.2). No usa la API de OpenAI.

Uso:
    uv run python src/scripts/verify_eval.py             # consulta PyPI (gratis) con caché en data/validate/
    uv run python src/scripts/verify_eval.py --offline   # solo el caché de PyPI

Chequea los gold contra la doc normalizada, el validador y las pistas del reemplazo (tprag/eval/verify.py). Solo
informa; sale con código 1 si algún ítem falla. Correrlo al agregar ítems y después de cada re-normalización.
"""

import argparse

from tprag.eval.verify import load_items, verify_item
from tprag.normalize.store import load_normalized
from tprag.validate.pypi import CachedPyPI
from tprag.validate.validate import validate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="no consulta PyPI; lo que no está en caché: unknown")
    args = parser.parse_args()

    items = load_items()
    packages = {g["package"] for it in items for t in it["targets"] for g in t["gold"]}
    docs = {d.id: d.page_content for d in load_normalized(packages=sorted(packages))}
    fetch = CachedPyPI(offline=args.offline)

    failed = 0
    for item in items:
        try:
            results = validate(item["code"], fetch)
        except SyntaxError:
            results = None
        report = verify_item(item, docs, results)
        failed += bool(report.errors)
        print(f"{'FALLA' if report.errors else 'ok':<6} {report.id}")
        for msg in report.errors + report.notes:
            print(f"       - {msg}")
    print(f"\n{len(items) - failed}/{len(items)} ítems ok")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
