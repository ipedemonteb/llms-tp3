"""Figuras 10a a 18 de las diapositivas (generador, iteración y test). No usa API ni modelos: lee los resultados ya
guardados. Las figuras 1 a 9 las arma plots_generation.py; el detalle de cada figura está en
tprag.eval.plots_generation.

- 10a: Éxito automático y Faithfulness contra Opus 5.5 con rúbrica (reranker sin iterar, dev).
- 10b: veredictos de gpt-4o-mini y de Opus 5.5 con rúbrica (judge_rubric.json).
- 11, 14 y 18: diagnóstico de los imports no exitosos (sin iterar en dev, Variante 7 en dev y en test; 14 usa la
  escala de 11).
- 12 y 15: Éxito automático, Precisión y Negative Rejection por paso, en dev y en test.
- 13 y 16: Éxito automático y Éxito confirmado por Opus, y los dos jueces, por paso, en dev y en test.
- 17: el retrieval con reranker en dev y en test.

Todo con el umbral elegido en dev (en test no se elige).

Uso:
    uv run --group analysis python src/scripts/plots_slides.py [--out <carpeta>] [--force]

Escribe en results/study/figures/generation/ (o en --out); no pisa una figura existente salvo con --force.
"""

import argparse
import json
import sys
from pathlib import Path

from tprag.eval.outputs import refuse_overwrite
from tprag.eval.plots_generation import (fig_confirmed, fig_dev_vs_test, fig_diagnosis, fig_steps, fig_validation_judge,
                                         fig_validation_opus, success_vs_opus)
from tprag.eval.threshold import counts, targets
from tprag.generate.schema import REPLACED
from tprag.split.store import ROOT

from analyze_generation import load_items  # mismo directorio de scripts
from confirmed_imports import STEPS as CONFIRMED_STEPS

GEN_DIR, STUDY_DIR = ROOT / "results" / "eval" / "gen", ROOT / "results" / "study"
REVIEW_DIR = STUDY_DIR / "review"
OUT_DIR = STUDY_DIR / "figures" / "generation"
BASE = "s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4o-mini"  # reranker sin iterar
ITER7 = "s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4.1-mini"  # Variante 7, el sistema final
STEPS = {  # split → paso → corrida
    "dev": {"Sin iterar": f"{BASE}__dev", "Variante 3": f"{BASE}__dev__iter3", "Variante 6": f"{BASE}__dev__iter6",
            "Variante 7": f"{ITER7}__dev__iter7"},
    "test": {"Sin iterar": f"{BASE}__test", "Variante 7": f"{ITER7}__test__iter7"},
}
NAMES = ("10a_validacion_exito_faithfulness", "10b_validacion_juez", "11_diagnostico", "12_resultados_por_paso",
         "13_exito_confirmado", "14_diagnostico_variante7", "15_resultados_test", "16_validacion_test",
         "17_retrieval_dev_vs_test", "18_diagnostico_test")


def read_json(name: str) -> dict:
    return json.loads((STUDY_DIR / name).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def records(run: str) -> list[dict]:
    return read_jsonl(GEN_DIR / f"{run}.items.jsonl")


def judges(split: str) -> dict[str, dict]:
    """Paso → preservación media de gpt-4o-mini (sin rúbrica) y de Opus 5.5 con rúbrica, sobre los mismos scripts."""
    def mean(d: dict) -> dict:
        return {"gpt-4o-mini": d["mean"]["gpt-4o-mini"], "opus": d["mean"]["opus"]}
    if split == "dev":
        runs = read_json("judge_rubric.json")["runs"]
        return {"Sin iterar": mean(runs["rerank"]["judges"]["sin rúbrica"]["total"]),
                "Variante 6": mean(runs["iter6"]["judges"]["sin rúbrica"]["total"]),
                "Variante 7": mean(read_json("judge_rubric_iter7.json")["gpt_vs_opus"]["total"])}
    return {"Sin iterar": mean(read_json("judge_rubric_sin_iterar_test.json")["gpt_vs_opus"]["total"]),
            "Variante 7": mean(read_json("judge_rubric_iter7_test.json")["gpt_vs_opus"]["total"])}


def confirmed(split: str, threshold: float) -> tuple[dict[str, dict], int]:
    """Paso → veredictos de Opus por import (`success_vs_opus`), y los positivos del split."""
    out, positives = {}, None
    for name, (run, answers_file, tag) in CONFIRMED_STEPS[split].items():
        recs = records(run)
        positives = sum(t["expected"] == REPLACED for r in recs for t in r["targets"])
        out[name] = success_vs_opus(recs, read_jsonl(REVIEW_DIR / answers_file), tag, threshold)
    return out, positives


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="carpeta de salida")
    parser.add_argument("--force", action="store_true", help="pisa las figuras existentes")
    args = parser.parse_args()
    if refusal := refuse_overwrite([args.out / f"{n}.png" for n in NAMES], args.force):
        print(refusal)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    threshold = read_json(f"generation_{STEPS['dev']['Sin iterar']}_analysis.json")["chosen"]["threshold"]
    originals = {i: it["code"] for i, it in load_items().items()}
    step_counts = {split: {name: counts(targets(records(run), originals), threshold) for name, run in steps.items()}
                   for split, steps in STEPS.items()}
    diagnosis = {split: read_json(f"diagnosis_{run}.json") for split, run in
                 (("dev", STEPS["dev"]["Sin iterar"]), ("dev7", STEPS["dev"]["Variante 7"]),
                  ("test7", STEPS["test"]["Variante 7"]))}
    conf_dev, pos_dev = confirmed("dev", threshold)
    conf_test, pos_test = confirmed("test", threshold)

    paths = [
        fig_validation_opus(conf_dev["Sin iterar"], read_json("faith_vs_opus.json"), args.out),
        fig_validation_judge(read_json("judge_rubric.json"), args.out),
        fig_diagnosis(diagnosis["dev"], args.out),
        fig_steps(step_counts["dev"], args.out),
        fig_confirmed(conf_dev, pos_dev, judges("dev"), args.out),
        fig_diagnosis(diagnosis["dev7"], args.out, "14_diagnostico_variante7",
                      xmax=max(diagnosis["dev"]["causes"].values())),
        fig_steps(step_counts["test"], args.out, "15_resultados_test"),
        fig_confirmed(conf_test, pos_test, judges("test"), args.out, "16_validacion_test"),
        fig_dev_vs_test(read_json("rerank_full_dev_analysis.json"), read_json("rerank_full_test_iter7check_analysis.json"),
                        {split: step_counts[split]["Sin iterar"]["passed_pos"] for split in ("dev", "test")}, args.out),
        fig_diagnosis(diagnosis["test7"], args.out, "18_diagnostico_test"),
    ]
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
