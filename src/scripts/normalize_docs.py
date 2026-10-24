"""Normaliza la documentación cruda (data/raw) a Markdown en data/normalized/<modo>/ (docs/agents/loaders.md).

Uso:
    uv run python src/scripts/normalize_docs.py                                  # todas las librerías
    uv run python src/scripts/normalize_docs.py --packages requests pypdf
    uv run python src/scripts/normalize_docs.py --console-examples code_only     # otro modo, otra carpeta

Solo lee data/raw. Escribe los .md, index.jsonl (una fila por archivo) y report.json (totales de la corrida:
lenguajes de los bloques, lo eliminado, archivos que pandoc no pudo leer e inclusiones sin resolver).
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tprag.normalize.codeblocks import CONSOLE_MODES
from tprag.normalize.store import (NORMALIZED_DIR, RAW_DIR, mode_dir, normalize_package, summarize, write_index,
                                   write_package)


def _normalize(package: str, console_mode: str):
    return package, normalize_package(RAW_DIR / package, console_mode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packages", nargs="*", help="solo estas librerías")
    parser.add_argument("--console-examples", choices=CONSOLE_MODES, default="outputs_as_comments")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--out", type=Path, default=NORMALIZED_DIR)
    args = parser.parse_args()

    base = mode_dir(args.console_examples, args.out)  # falla si apunta dentro de data/raw
    base.mkdir(parents=True, exist_ok=True)
    packages = args.packages or sorted(d.name for d in RAW_DIR.iterdir() if (d / "manifest.json").exists())

    start, entries = time.time(), []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_normalize, p, args.console_examples) for p in packages]
        for fut in as_completed(futures):
            package, results = fut.result()
            write_package(results, package, base)
            pkg_entries = [e for e, _ in results]
            entries += pkg_entries
            fallback = sum(e.status == "fallback" for e in pkg_entries)
            unresolved = sum(len(e.unresolved) for e in pkg_entries)
            print(f"{package:24} archivos={len(pkg_entries):4} bloques={sum(e.n_blocks for e in pkg_entries):6} "
                  f"fallback={fallback:3} inclusiones_sin_resolver={unresolved:3}", flush=True)

    write_index(entries, base, set(packages))
    report = {"console_mode": args.console_examples, "packages": len(packages),
              "seconds": round(time.time() - start, 1), **summarize(entries)}
    (base / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n{report['files']} archivos, {report['blocks']} bloques, estados {report['status']}, "
          f"{report['seconds']} s -> {base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
