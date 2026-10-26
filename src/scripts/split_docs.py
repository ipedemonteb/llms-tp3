"""Corta la documentación normalizada en chunks con S0-S3 (docs/agents/splitters.md).

Uso:
    uv run python src/scripts/split_docs.py                                   # S0-S3, todas las librerías
    uv run python src/scripts/split_docs.py --splitters s3 --packages httpx requests
    uv run python src/scripts/split_docs.py --splitters s3 --max-tokens 512   # fase 2

Solo lee data/normalized/<modo>/ y no usa la API. Escribe data/processed/<modo>/<splitter>-<máx>/chunks.jsonl con
todos los chunks (el filtro y el tope se aplican al armar el índice, §3.7) y config.json con la configuración y los
totales de la corrida. Con --packages, el archivo tiene solo esas librerías.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict

from tprag.normalize.store import load_normalized
from tprag.split.splitters import SPLITTERS, SplitConfig, split_document
from tprag.split.store import PROCESSED_DIR, config_name, write_chunks
from tprag.split.tokens import DEFAULT_TOKENIZER


def _split_package(package: str, console_mode: str, splitters: list[str], cfg: SplitConfig):
    docs = load_normalized(console_mode, packages=[package])
    return package, {s: [c for d in docs for c in split_document(d, s, cfg)] for s in splitters}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splitters", nargs="*", choices=sorted(SPLITTERS), default=sorted(SPLITTERS))
    parser.add_argument("--packages", nargs="*", help="solo estas librerías")
    parser.add_argument("--max-tokens", type=int, default=SplitConfig.max_tokens)
    parser.add_argument("--min-tokens", type=int, default=SplitConfig.min_tokens, help="solo S3; 0 = sin mínimo")
    parser.add_argument("--overlap", type=int, default=SplitConfig.overlap, help="solo S0 y S1")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="cl100k_base o hf:<modelo>")
    parser.add_argument("--console-examples", default="outputs_as_comments", help="carpeta de data/normalized")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    args = parser.parse_args()

    cfg = SplitConfig(max_tokens=args.max_tokens, min_tokens=args.min_tokens, overlap=args.overlap,
                      tokenizer=args.tokenizer)
    packages = args.packages or sorted({d.metadata["package"] for d in load_normalized(args.console_examples)})

    start, results = time.time(), {s: [] for s in args.splitters}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_split_package, p, args.console_examples, args.splitters, cfg) for p in packages]
        for fut in as_completed(futures):
            package, by_splitter = fut.result()
            for s, chunks in by_splitter.items():
                results[s] += chunks
            print(f"{package:24} " + " ".join(f"{s}={len(c):5}" for s, c in by_splitter.items()), flush=True)

    seconds = round(time.time() - start, 1)
    base = PROCESSED_DIR / args.console_examples
    for s, chunks in results.items():
        out = base / config_name(s, cfg)
        write_chunks(chunks, out / "chunks.jsonl")
        config = {"splitter": s, **asdict(cfg), "console_mode": args.console_examples,
                  "packages": len(packages), "only_packages": args.packages or None, "chunks": len(chunks),
                  "use_cases": sum(c.is_use_case for c in chunks), "seconds_all_splitters": seconds}
        (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{s}: {len(chunks)} chunks, {config['use_cases']} casos de uso -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
