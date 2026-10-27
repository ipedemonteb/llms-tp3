"""Valida los imports de un script de Python (paso 5, docs/agents/validador.md). No usa la API de OpenAI.

Uso:
    uv run python src/scripts/validate_deps.py script.py
    uv run python src/scripts/validate_deps.py script.py --offline   # solo el caché de PyPI
    uv run python src/scripts/validate_deps.py script.py --years 5   # umbral de la advertencia de antigüedad

Consulta la API JSON de PyPI (gratis) con caché en data/validate/pypi_cache/ e imprime un JSON con un resultado por
import. Sale con código 1 si el script no parsea.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tprag.validate.pypi import CachedPyPI
from tprag.validate.validate import DEFAULT_YEARS, validate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    parser.add_argument("--offline", action="store_true", help="no consulta PyPI; lo que no está en caché: unknown")
    parser.add_argument("--years", type=int, default=DEFAULT_YEARS, help="advertencia si no hay releases hace más")
    args = parser.parse_args()

    source = args.script.read_text(encoding="utf-8")
    try:
        results = validate(source, CachedPyPI(offline=args.offline), years=args.years)
    except SyntaxError as e:
        print(json.dumps({"file": str(args.script), "syntax_error": f"línea {e.lineno}: {e.msg}"}, ensure_ascii=False))
        return 1
    report = {"file": str(args.script),
              "to_rag": [r.module for r in results if r.to_rag],
              "imports": [asdict(r) for r in results]}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
