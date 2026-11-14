"""Análisis de una corrida del generador (docs/agents/metricas.md §3-§4). No usa API, modelos ni Chroma: lee el detalle
que dejó eval_generation.py (results/eval/gen/<corrida>.items.jsonl y su resumen .json) y el dataset.

Uso:
    uv run python src/scripts/analyze_generation.py s3-1024__e4__r1__cap150__qA__gpt-4o-mini__dev
    uv run python src/scripts/analyze_generation.py results/eval/gen/<corrida> --print-only

Calcula, sin volver a llamar al LLM:
- Barrido del umbral (tprag.eval.threshold): por cada umbral evaluable, correcciones buenas / malas / perdidas,
  cambios indebidos, abstenciones correctas, reporte engañoso, éxito en positivos, Negative Rejection, llamadas al LLM
  y el puntaje con λ = 1, 2, 3. Elige el umbral con λ (2 por defecto; empate → el más alto) y da la sensibilidad.
- Faithfulness (tprag.eval.faithfulness) por script cambiado; en cada umbral, el promedio sobre los scripts que lo
  pasan. El texto de los chunks sale del detalle o, en corridas viejas que no lo guardan, de data/processed/.
- Answer Relevance (a) y (b), si ya se juzgó la corrida (judge_generation.py deja <corrida>.judge.jsonl): en cada
  umbral, el promedio sobre los scripts cambiados que lo pasan. Si no hay juicio, esas columnas quedan vacías.
- Noise Robustness (tprag.eval.noise) con el umbral elegido.
- Por subconjunto (manual / sintético / negativos) con el umbral elegido.

Para elegir el umbral, la corrida tiene que ser permisiva (eval_generation.py --threshold por debajo de todo
puntaje): con una corrida no permisiva solo se pueden evaluar umbrales iguales o mayores que el que usó.

Si el dataset cambió desde la corrida (sha256 distinto), se niega a seguir salvo con --allow-dataset-mismatch
(los ítems regenerados se compararían contra su versión nueva).

Escribe results/study/generation_<corrida>_analysis.json y _table.md (con --print-only, solo imprime).
"""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from tprag.eval.faithfulness import faithfulness, load_chunk_texts, record_context
from tprag.eval.noise import summarize as noise_summary
from tprag.eval.noise import target_groups
from tprag.eval.retrieval import dataset_info
from tprag.eval.threshold import (BAD, CORRECT_ABSTENTION, GOOD, LAMBDA, MISSED, SENSITIVITY, WRONG_CHANGE, by_subset,
                                  choose, sensitivity, sweep, targets, utility)
from tprag.eval.verify import EVAL_DIR, EVAL_FILES
from tprag.split.store import ROOT

GEN_DIR, STUDY_DIR = ROOT / "results" / "eval" / "gen", ROOT / "results" / "study"


def run_paths(run: str) -> tuple[Path, Path]:
    base = Path(run) if "/" in run else GEN_DIR / run
    base = base.with_name(base.name.removesuffix(".items.jsonl").removesuffix(".json"))
    return base.with_name(base.name + ".json"), base.with_name(base.name + ".items.jsonl")


def load_items() -> dict[str, dict]:
    return {it["id"]: it for name in EVAL_FILES if (EVAL_DIR / name).exists()
            for it in (json.loads(line) for line in (EVAL_DIR / name).read_text(encoding="utf-8").splitlines()
                       if line.strip())}


def chunk_texts(summary: dict, records: list[dict]) -> dict[str, str]:
    """Texto de los chunks que no vienen en el detalle (corridas anteriores al cambio de eval_generation.py)."""
    missing = {ch["id"] for r in records for c in r.get("contexts", []) for ch in c["chunks"] if "text" not in ch}
    if not missing:
        return {}
    cfg = summary["config"]
    path = ROOT / "data" / "processed" / cfg["console_mode"] / cfg["chunks"] / "chunks.jsonl"
    texts = load_chunk_texts(path, missing)
    if len(texts) < len(missing):
        raise SystemExit(f"faltan {len(missing) - len(texts)} chunks en {path} (¿chunks regenerados?)")
    return texts


def faithfulness_by_item(records: list[dict], originals: dict[str, str], texts: dict[str, str]) -> dict[str, dict]:
    return {r["id"]: faithfulness(originals[r["id"]], r["code"], record_context(r, texts))
            for r in records if r["code"] != originals[r["id"]]}


def mean_passing(records: list[dict], scores: dict[str, float | None], threshold: float) -> tuple[float | None, int]:
    """Promedio de un puntaje por script (Faithfulness, Answer Relevance) sobre los scripts que lo tienen (los
    cambiados) y con al menos un target que pasa el umbral."""
    vals = [scores[r["id"]] for r in records if scores.get(r["id"]) is not None
            and any(t["retrieval_score"] is not None and t["retrieval_score"] >= threshold for t in r["targets"])]
    return (mean(vals) if vals else None), len(vals)


def load_judge(items_path: Path) -> dict[str, dict]:
    """id → {"a": puntaje, "b": puntaje} del juicio de la corrida, si existe."""
    path = items_path.with_name(items_path.name.removesuffix(".items.jsonl") + ".judge.jsonl")
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["id"]: {j: r[j]["score"] for j in ("a", "b") if j in r} for r in rows}


def jsonable(x):
    """JSON no admite infinitos: ±inf → "±inf"; las claves λ → texto."""
    if isinstance(x, float) and math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [jsonable(v) for v in x]
    return x


def fmt(x, digits=3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return ("+∞" if x > 0 else "−∞") if math.isinf(x) else f"{x:.{digits}f}"
    return str(x)


def table(a: dict) -> str:
    run, ch = a["run"], a["chosen"]
    lines = [
        f"# Análisis del generador: `{run['name']}`",
        "",
        f"Colección `{run['collection']}`, reranker {run['reranker'] or 'no'}, {run['llm']}, split {run['split']}, "
        f"umbral de la corrida {fmt(run['threshold'], 4)} ({'permisiva' if run['permissive'] else 'NO permisiva'}). "
        f"Targets: {ch['positives']} positivos, {ch['negatives']} negativos.",
        "",
        f"## Umbral elegido (λ = {a['lambda']}): {fmt(ch['threshold'], 4)}",
        "",
        "| Buenas | Malas | Perdidas | Cambios indebidos | Abst. correctas | Reporte engañoso | Éxito positivos | "
        "Negative Rejection | Faithfulness | AR (b) | AR (a) | Llamadas LLM | Puntaje |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        f"| {ch[GOOD]} | {ch[BAD]} | {ch[MISSED]} | {ch[WRONG_CHANGE]} | {ch[CORRECT_ABSTENTION]} | "
        f"{ch['misleading_report']} | {fmt(ch['success_pos'])} | {fmt(ch['negative_rejection'])} | "
        f"{fmt(ch['faithfulness'])} (n={ch['faithfulness_n']}) | {fmt(ch['relevance_b'])} (n={ch['relevance_b_n']}) | "
        f"{fmt(ch['relevance_a'])} (n={ch['relevance_a_n']}) | {ch['llm_calls']} | {fmt(utility(ch, a['lambda']), 1)} |",
        "",
        "AR = Answer Relevance: (b) juez directo (principal), (a) técnica de clase. "
        + ("Sin juicio todavía (judge_generation.py)." if not a["run"]["judged"] else
           f"Juzgados {a['run']['judged']} scripts."),
        "",
        "Sensibilidad (umbral elegido con cada λ): "
        + ", ".join(f"λ = {lam:g} → {fmt(th, 4)}" for lam, th in a["sensitivity"].items()),
        "",
        "## Por subconjunto (umbral elegido)",
        "",
        "| Subconjunto | Positivos | Negativos | Éxito positivos | Malas | Negative Rejection | Cambios indebidos |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [f"| {s} | {r['positives']} | {r['negatives']} | {fmt(r['success_pos'])} | {r[BAD]} | "
              f"{fmt(r['negative_rejection'])} | {r[WRONG_CHANGE]} |" for s, r in a["by_subset"].items()]
    n = a["noise"]
    lines += [
        "",
        "## Noise Robustness (umbral elegido; positivos con un paquete aceptable en el contexto)",
        "",
        "| Grupo | Targets | Éxito |",
        "|---|---|---|",
    ] + [f"| {g} | {n[g]['targets']} | {fmt(n[g]['success'])} |"
         for g in ("limpio", "ruido_aceptable_primero", "ruido_aceptable_despues", "noisy",
                   "sin_aceptable_en_contexto")] + [
        "",
        f"Noise Robustness (éxito con ruido / éxito limpio): {fmt(n['noise_robustness'])}",
        "",
        "## Barrido del umbral",
        "",
        "| Umbral | Buenas | Malas | Perdidas | Indebidos | Abst. ok | Engañoso | Éxito + | Neg. Rej. | Faithf. | "
        "AR (b) | AR (a) | Llamadas | " + " | ".join(f"U(λ={lam:g})" for lam in SENSITIVITY) + " |",
        "|" + "---|" * (13 + len(SENSITIVITY)),
    ]
    rows = a["sweep"]
    step = max(1, len(rows) // 25)
    keep = {0, len(rows) - 1, *range(0, len(rows), step)} | {i for i, r in enumerate(rows)
                                                              if r["threshold"] == ch["threshold"]}
    for i in sorted(keep):
        r = rows[i]
        mark = " ⭐" if r["threshold"] == ch["threshold"] else ""
        lines.append(f"| {fmt(r['threshold'], 4)}{mark} | {r[GOOD]} | {r[BAD]} | {r[MISSED]} | {r[WRONG_CHANGE]} | "
                     f"{r[CORRECT_ABSTENTION]} | {r['misleading_report']} | {fmt(r['success_pos'])} | "
                     f"{fmt(r['negative_rejection'])} | {fmt(r['faithfulness'])} | {fmt(r['relevance_b'])} | "
                     f"{fmt(r['relevance_a'])} | {r['llm_calls']} | "
                     + " | ".join(fmt(r["utility"][lam], 1) for lam in SENSITIVITY) + " |")
    if len(keep) < len(rows):
        lines += ["", f"(Se muestran {len(keep)} de {len(rows)} umbrales; todos están en el JSON.)"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", help="nombre de la corrida en results/eval/gen/ (sin extensión) o su ruta")
    parser.add_argument("--lambda", dest="lam", type=float, default=LAMBDA, help="penalización de los errores")
    parser.add_argument("--allow-dataset-mismatch", action="store_true",
                        help="analizar aunque el dataset haya cambiado desde la corrida (solo para probar)")
    parser.add_argument("--print-only", action="store_true", help="no escribe archivos")
    args = parser.parse_args()

    summary_path, items_path = run_paths(args.run)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    current = dataset_info()["sha256"]
    mismatch = {k for k, v in summary.get("dataset", {}).get("sha256", {}).items() if current.get(k) != v}
    if mismatch and not args.allow_dataset_mismatch:
        print(f"el dataset cambió desde la corrida ({', '.join(sorted(mismatch))}): correrla de nuevo o usar "
              "--allow-dataset-mismatch (solo para probar)")
        return 1

    items = load_items()
    originals = {i: it["code"] for i, it in items.items()}
    ts = targets(records, originals)
    rows = sweep(ts)
    faith = faithfulness_by_item(records, originals, chunk_texts(summary, records))
    judge = load_judge(items_path)
    per_item = {"faithfulness": {i: f["score"] for i, f in faith.items()},
                "relevance_a": {i: j.get("a") for i, j in judge.items()},
                "relevance_b": {i: j.get("b") for i, j in judge.items()}}
    for r in rows:
        for metric, scores in per_item.items():
            r[metric], r[metric + "_n"] = mean_passing(records, scores, r["threshold"])
    chosen = choose(rows, args.lam)
    noise = noise_summary(target_groups(records, items, chosen["threshold"], strict=not mismatch))

    name = items_path.name.removesuffix(".items.jsonl")
    analysis = {
        "run": {"name": name, "collection": summary["collection"], "reranker": summary.get("reranker"),
                "llm": summary["llm"], "split": summary["split"], "threshold": summary["threshold"],
                "permissive": all(t.passed_run for t in ts if t.score is not None), "judged": len(judge),
                "dataset_mismatch": sorted(mismatch)},
        "lambda": args.lam,
        "chosen": chosen,
        "sensitivity": sensitivity(rows),
        "by_subset": by_subset(ts, chosen["threshold"]),
        "noise": noise,
        "faithfulness_by_item": faith,
        "judge_by_item": judge,
        "sweep": rows,
    }
    text = table(analysis)
    print(text)
    if not args.print_only:
        STUDY_DIR.mkdir(parents=True, exist_ok=True)
        out = STUDY_DIR / f"generation_{name}"
        out.with_name(out.name + "_analysis.json").write_text(
            json.dumps(jsonable(analysis), ensure_ascii=False, indent=1), encoding="utf-8")
        out.with_name(out.name + "_table.md").write_text(text, encoding="utf-8")
        print(f"escrito {out.name}_analysis.json y _table.md en {STUDY_DIR.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
