"""Éxito confirmado por import (reranker, umbral elegido en dev): los éxitos automáticos en scripts que Opus 5.5 con
rúbrica marca `si`. Misma unidad que el Éxito automático (imports sobre los positivos del split), para sin iterar y
las variantes revisadas con rúbrica (dev: `__iter6`, `__iter7`; test: `__iter7`). No usa API: cruza el detalle de cada
corrida con las respuestas de Opus ya guardadas (`tprag.eval.plots_generation.success_vs_opus`).

Uso:
    uv run --group analysis python src/scripts/confirmed_imports.py [--split test]

Escribe results/study/exito_confirmado_imports{,_test}.{json,md}; no pisa una salida existente.
"""

import argparse
import json
import sys

from tprag.eval.outputs import refuse_overwrite
from tprag.eval.plots_generation import success_vs_opus
from tprag.generate.schema import REPLACED
from tprag.split.store import ROOT

GEN_DIR, STUDY_DIR = ROOT / "results" / "eval" / "gen", ROOT / "results" / "study"
REVIEW_DIR = STUDY_DIR / "review"
BASE = "s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4o-mini__dev"
ITER7 = "s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4.1-mini"
STEPS = {  # split → paso → (corrida, respuestas de Opus con rúbrica, nombre de la corrida en las respuestas)
    "dev": {
        "Sin iterar": (BASE, "opus_rubric_answers.jsonl", "rerank"),
        "Variante 6": (f"{BASE}__iter6", "opus_rubric_answers.jsonl", "iter6"),
        "Variante 7": (f"{ITER7}__dev__iter7", "opus_rubric_iter7_answers.jsonl", "iter7"),
    },
    "test": {
        "Sin iterar": (BASE.replace("__dev", "__test"), "opus_rubric_sin_iterar_test_answers.jsonl", "sin_iterar_test"),
        "Variante 7": (f"{ITER7}__test__iter7", "opus_rubric_iter7_test_answers.jsonl", "iter7_test"),
    },
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mixed_scripts(records: list[dict], judged: set[str], threshold: float) -> list[str]:
    """Scripts juzgados con imports exitosos y no exitosos a la vez (el veredicto de Opus es del script entero)."""
    out = []
    for r in records:
        ok = {t["success"] for t in r["targets"] if t["expected"] == REPLACED and t["retrieval_score"] is not None
              and t["retrieval_score"] >= threshold}
        if r["id"] in judged and len(ok) > 1:
            out.append(r["id"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=list(STEPS), default="dev")
    args = parser.parse_args()
    suffix = "" if args.split == "dev" else f"_{args.split}"
    OUT = [STUDY_DIR / f"exito_confirmado_imports{suffix}.json", STUDY_DIR / f"exito_confirmado_imports{suffix}.md"]
    if refusal := refuse_overwrite(OUT, False):
        print(refusal)
        return 1
    # el umbral es siempre el elegido en dev (en test no se elige)
    threshold = json.loads((STUDY_DIR / f"generation_{BASE}_analysis.json").read_text(encoding="utf-8"))["chosen"][
        "threshold"]
    steps = {}
    for name, (run, answers_file, tag) in STEPS[args.split].items():
        records = read_jsonl(GEN_DIR / f"{run}.items.jsonl")
        answers = read_jsonl(REVIEW_DIR / answers_file)
        positives = sum(t["expected"] == REPLACED for r in records for t in r["targets"])
        by = success_vs_opus(records, answers, tag, threshold)
        automatic = sum(by[True].values())
        steps[name] = {
            "run": run, "positives": positives, "automatic": automatic, "confirmed": by[True]["si"],
            "automatic_rate": automatic / positives, "confirmed_rate": by[True]["si"] / positives,
            "opus_on_successful": by[True], "opus_on_unsuccessful": by[False],
            "mixed_scripts": mixed_scripts(records, {a["id"] for a in answers if a["run"] == tag}, threshold),
        }
    OUT[0].write_text(json.dumps({"threshold": threshold, "steps": steps}, ensure_ascii=False, indent=1) + "\n",
                      encoding="utf-8")
    lines = [f"# Éxito confirmado por import ({args.split}, reranker)", "",
             f"Umbral elegido: {threshold}. Confirmado = éxito automático en un script que Opus 5.5 con rúbrica marca "
             "`si` (el veredicto es del script entero y vale para sus imports). Imports sobre los positivos del split.", "",
             "| Paso | Éxito automático | Éxito confirmado | Opus en los exitosos (sí / parcial / no) | Scripts mixtos |",
             "|---|---|---|---|---|"]
    for name, s in steps.items():
        o = s["opus_on_successful"]
        lines.append(f"| {name} | {s['automatic_rate']:.3f} ({s['automatic']}/{s['positives']}) | "
                     f"{s['confirmed_rate']:.3f} ({s['confirmed']}/{s['positives']}) | "
                     f"{o['si']} / {o['parcial']} / {o['no']} | {', '.join(s['mixed_scripts']) or '—'} |")
    lines += ["", "Scripts mixtos: juzgados con imports exitosos y no exitosos a la vez; el import exitoso cuenta como "
              "confirmado solo si Opus marcó `si` el script.", ""]
    OUT[1].write_text("\n".join(lines), encoding="utf-8")
    for p in OUT:
        print(p.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
