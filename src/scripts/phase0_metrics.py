"""Métricas de la fase 0 de los splitters (docs/agents/splitters.md §3.10). No usa la API.

Uso:
    uv run python src/scripts/phase0_metrics.py                       # todas las configs de data/processed/<modo>/
    uv run python src/scripts/phase0_metrics.py --configs s3-1024

Lee data/processed/<modo>/<config>/chunks.jsonl y los documentos normalizados (para los bloques cortados). Escribe en
results/phase0/: <config>.json por config, summary.md (tabla comparativa) y <config>_bordes.md (muestra de bordes de
corte para revisar a mano; solo S3).
"""

import argparse
import json
from pathlib import Path

from tprag.normalize.store import load_normalized
from tprag.split import metrics
from tprag.split.markdown import parse
from tprag.split.select import DEFAULT_CAP
from tprag.split.store import PROCESSED_DIR, ROOT, load_chunks
from tprag.split.tokens import token_counter

RESULTS_DIR = ROOT / "results" / "phase0"


def compute(name: str, base: Path, cap: int, sample: int, seed: int) -> dict:
    config = json.loads((base / name / "config.json").read_text(encoding="utf-8"))
    chunks = load_chunks(base / name / "chunks.jsonl")
    packages = sorted({c.package for c in chunks})
    parsed = {d.metadata["id"]: parse(d.page_content, d.metadata["source_path"])
              for d in load_normalized(config["console_mode"], packages=packages)}
    result = {
        "config": config,
        "sizes": metrics.size_stats(chunks),
        "code_and_prose": metrics.code_and_prose(chunks),
        "cut_code_blocks": metrics.cut_code_blocks(chunks, parsed),
        "truncated": metrics.truncated(chunks, token_counter(config["tokenizer"])),
        "use_cases": metrics.use_cases(chunks, cap),
        "oversized": metrics.oversized(chunks, config["max_tokens"]),
    }
    if config["splitter"] == "s3":
        (RESULTS_DIR / f"{name}_bordes.md").write_text(metrics.border_sample(chunks, sample, seed) + "\n",
                                                       encoding="utf-8")
    return result


def summary(results: dict[str, dict]) -> str:
    rows = ["| Config | Chunks | Mediana | p90 | Máx. | Código + explicación | Bloques cortados | > máximo "
            "| Casos de uso (sin tope) | Casos de uso (con tope) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in sorted(results.items()):
        s, u = r["sizes"], r["use_cases"]
        rows.append(f"| {name} | {s['chunks']} | {s.get('median', '-')} | {s.get('p90', '-')} | {s.get('max', '-')} "
                    f"| {r['code_and_prose']['pct']}% | {r['cut_code_blocks']['pct']}% "
                    f"({r['cut_code_blocks']['cut']}) | {r['oversized']['chunks']} | {u['total_no_cap']} "
                    f"| {u['total_cap']} ({u['packages_over_cap']} libs recortadas) |")
    return "# Fase 0: métricas de los splitters\n\nTamaños en tokens (`cl100k_base` salvo que la config diga otro).\n\n" \
           + "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", help="p. ej. s3-1024 (default: todas las de la carpeta)")
    parser.add_argument("--console-examples", default="outputs_as_comments")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--sample", type=int, default=30, help="bordes de corte a revisar a mano")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    base = PROCESSED_DIR / args.console_examples
    names = args.configs or sorted(d.name for d in base.iterdir() if (d / "chunks.jsonl").exists())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in names:
        results[name] = compute(name, base, args.cap, args.sample, args.seed)
        (RESULTS_DIR / f"{name}.json").write_text(json.dumps(results[name], indent=2, ensure_ascii=False) + "\n",
                                                  encoding="utf-8")
        print(f"{name}: listo")
    # La tabla incluye todas las configs medidas hasta ahora, no solo las de esta corrida
    every = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in RESULTS_DIR.glob("*.json")}
    (RESULTS_DIR / "summary.md").write_text(summary(every), encoding="utf-8")
    print(f"-> {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
