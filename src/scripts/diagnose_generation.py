"""Diagnóstico de una corrida del generador (iteracion_ej4.md §1, §8.5, §9). No usa API, modelos ni Chroma: lee el detalle
que dejó eval_generation.py (results/eval/gen/<corrida>.items.jsonl) y el dataset.

Uso:
    uv run python src/scripts/diagnose_generation.py <corrida>
    uv run python src/scripts/diagnose_generation.py <corrida> --vs <otra> [<otra> ...] [--print-only]

Con el umbral elegido de cada corrida (la regla de retrieval al 95%, como analyze_generation.py):
- por qué falla cada import positivo no exitoso, con una causa por import (tprag.eval.diagnosis.CAUSES);
- los negativos que terminan cambiados (cambios indebidos);
- cuánto reescribe el script completo (variantes uniformes: líneas del original cambiadas);
- con --vs, la diferencia pareada de éxito, precisión y Negative Rejection contra cada corrida, con IC 95% (bootstrap
  por script, 10.000 remuestreos, semilla 2026).

Escribe results/study/diagnosis_<corrida>.json y .md (con --print-only, solo imprime). Si ya existen no los pisa
(tprag.eval.outputs): --tag <nombre> escribe la variante al lado y --force la reemplaza.
"""

import argparse
import json

from tprag.eval.diagnosis import CAUSES, LABELS, REWRITE_LINES, compare_runs, diagnose
from tprag.eval.outputs import refuse_overwrite, tag_arg, tagged
from tprag.eval.threshold import RETRIEVAL, choose, describe, sweep, targets

from analyze_generation import STUDY_DIR, fmt, load_items, run_paths  # mismo directorio de scripts

METRICS = {"success_pos": "Éxito en positivos", "precision": "Precisión", "negative_rejection": "Negative Rejection"}


def load_run(run: str) -> tuple[str, list[dict]]:
    _, items_path = run_paths(run)
    records = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return items_path.name.removesuffix(".items.jsonl"), records


def chosen_threshold(records: list[dict], items: dict[str, dict]) -> float:
    originals = {i: it["code"] for i, it in items.items()}
    return choose(sweep(targets(records, originals)), criterion=RETRIEVAL)["threshold"]


def table(name: str, d: dict, comparisons: dict[str, dict]) -> str:
    lines = [f"# Diagnóstico del generador: `{name}`", "",
             f"Umbral elegido ({describe(RETRIEVAL)}): {fmt(d['threshold'], 4)}. Positivos: {d['positives']}; "
             f"exitosos: {d['good']}; no exitosos: {d['not_successful']}.", "",
             "## Por qué fallan los imports no exitosos (una causa por import: la primera en este orden)", "",
             "| Causa | Imports |", "|---|---|"]
    lines += [f"| {LABELS[c]} | {d['causes'][c]} |" for c in CAUSES if d["causes"][c]]
    lines += [f"| **Total** | **{sum(d['causes'].values())}** |", "",
              "## Negativos cambiados (cambios indebidos)", ""]
    lines += ([f"- `{w['item']}` ({w['module']} → {w['replacement_package']})" for w in d["wrong_changes"]]
              or ["Ninguno."])
    if d["rewrite"]:
        r = d["rewrite"]
        lines += ["", "## Cuánto reescribe (scripts entregados cambiados; líneas del original cambiadas)", "",
                  f"{r['scripts']} scripts: mediana {r['median']:g}, máximo {r['max']}, "
                  f"más de {REWRITE_LINES} líneas: {r[f'over_{REWRITE_LINES}']}."]
    for other, c in comparisons.items():
        lines += ["", f"## Comparación pareada: esta corrida (A) vs `{other}` (B)", "",
                  "IC 95% bootstrap por script (10.000 remuestreos, semilla 2026); empate si el intervalo de la "
                  "diferencia incluye el cero.", "",
                  "| Métrica | A | B | A − B | IC 95% | Veredicto |", "|---|---|---|---|---|---|"]
        lines += [f"| {label} | {fmt(c[m]['a'])} | {fmt(c[m]['b'])} | {c[m]['diff']:+.3f} | "
                  f"[{c[m]['ci'][0]:+.3f}; {c[m]['ci'][1]:+.3f}] | {c[m]['verdict']} |" if c[m] else
                  f"| {label} | - | - | - | - | sin casos |" for m, label in METRICS.items()]
        lines += ["", f"Positivos que A corrige y B no: {c['positives_won']}; que B corrige y A no: "
                      f"{c['positives_lost']}."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="nombre de la corrida en results/eval/gen/ (sin extensión) o su ruta")
    parser.add_argument("--vs", nargs="*", default=[], help="corridas contra las que comparar (pareado)")
    parser.add_argument("--print-only", action="store_true", help="no escribe archivos")
    parser.add_argument("--tag", type=tag_arg, help="sufijo __<tag> en las salidas: una variante al lado")
    parser.add_argument("--force", action="store_true", help="reemplaza las salidas si ya existen")
    args = parser.parse_args()

    name, records = load_run(args.run)
    out = STUDY_DIR / tagged(f"diagnosis_{name}", args.tag)
    outputs = [out.with_name(out.name + ".json"), out.with_name(out.name + ".md")]
    if not args.print_only and (refusal := refuse_overwrite(outputs, args.force)):
        print(refusal)
        return 1
    items = load_items()
    threshold = chosen_threshold(records, items)
    d = diagnose(records, items, threshold)
    comparisons = {}
    for other in args.vs:
        other_name, other_records = load_run(other)
        comparisons[other_name] = compare_runs(records, other_records, items, threshold,
                                               chosen_threshold(other_records, items))
    text = table(name, d, comparisons)
    print(text)
    if not args.print_only:
        STUDY_DIR.mkdir(parents=True, exist_ok=True)
        outputs[0].write_text(json.dumps({"run": name, **d, "comparisons": comparisons}, ensure_ascii=False, indent=1),
                              encoding="utf-8")
        outputs[1].write_text(text, encoding="utf-8")
        print(f"escrito {outputs[0].name} y {outputs[1].name} en {STUDY_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
