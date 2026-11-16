"""Gráficos del estudio del generador (docs/agents/metricas.md §3-§4). No usa API ni modelos: lee los análisis de las
corridas permisivas de dev (analyze_generation.py), su detalle (para reconstruir el nivel 1 anterior), el acuerdo de
la revisión manual (review_generation.py agreement) y la comparación de jueces (review_generation.py judge-vs-opus).

Uso:
    uv run --group analysis python src/scripts/plots_generation.py

Escribe results/study/figures/generation/*.png (detalle de cada figura en tprag.eval.plots_generation).
"""

import json
import sys

from tprag.eval.noise import summarize as noise_summary
from tprag.eval.noise import target_groups
from tprag.eval.plots_generation import (generation_figures, negative_rejection_curve, previous_success,
                                         relevance_by_success)
from tprag.eval.threshold import counts, targets
from tprag.split.store import ROOT

from analyze_generation import load_items  # mismo directorio de scripts

GEN_DIR, STUDY_DIR = ROOT / "results" / "eval" / "gen", ROOT / "results" / "study"
RUNS = {"base": "s3-1024__e4__r1__cap150__qA__gpt-4o-mini__dev",
        "rerank": "s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4o-mini__dev"}
OUT_DIR = STUDY_DIR / "figures" / "generation"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def level1_counts(records: list[dict], originals: dict[str, str]) -> dict[str, dict]:
    """Resultados con todo pasando al LLM, con el nivel 1 anterior (sin el chequeo de nombres) y el corregido."""
    previous = [r | {"targets": [t | {"success": previous_success(r, t)} for t in r["targets"]]} for r in records]
    return {"anterior": counts(targets(previous, originals), float("-inf")),
            "corregido": counts(targets(records, originals), float("-inf"))}


def main() -> int:
    analyses = {v: json.loads((STUDY_DIR / f"generation_{name}_analysis.json").read_text(encoding="utf-8"))
                for v, name in RUNS.items()}
    items = load_items()
    originals = {i: it["code"] for i, it in items.items()}
    records = {v: read_jsonl(GEN_DIR / f"{name}.items.jsonl") for v, name in RUNS.items()}
    level1 = {v: level1_counts(records[v], originals) for v in RUNS}
    agreement = json.loads((STUDY_DIR / "review_agreement.json").read_text(encoding="utf-8"))
    judges = json.loads((STUDY_DIR / "judge_vs_opus.json").read_text(encoding="utf-8"))
    extra = {v: {"negatives": negative_rejection_curve(targets(records[v], originals)),
                 "noise": noise_summary(target_groups(records[v], items, float("-inf"))),
                 "relevance": relevance_by_success(records[v], read_jsonl(GEN_DIR / f"{name}.judge.jsonl"))}
             for v, name in RUNS.items()}
    for path in generation_figures(analyses, level1, agreement, judges, extra, OUT_DIR):
        print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
