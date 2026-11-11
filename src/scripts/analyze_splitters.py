"""Análisis de la etapa 1 (docs/agents/plan_experimentos_indice.md): compara los splitters con lo que dejó
run_splitter_study.py. No usa API, embeddings ni Chroma: lee results/eval/ (el detalle con los 50 chunks de cada
target), results/index/ y results/study/.

Uso:
    uv run python src/scripts/analyze_splitters.py                 # tabla + JSON (+ gráficos si hay matplotlib)
    uv run python src/scripts/analyze_splitters.py --no-plots

Métricas (evaluacion.md): Package Hit@1, Span Hit con presupuesto de tokens (2000 es el default del proyecto; se
recalculan 1000 y 4000 como sensibilidad), AUROC. Los intervalos son bootstrap sobre los ítems de dev y las
comparaciones se hacen por pares, contra el mejor de cada métrica. Si el intervalo de la diferencia incluye el cero es
"empate". No elige el splitter ganador por sí solo: la regla (principal Span Hit@2000, desempate Package Hit@1, luego
el más simple o barato) se aplica mirando la tabla de empates y de costo.

Escribe results/study/splitters_analysis.json, splitters_table.md y, con matplotlib, figures/*.png.
"""

import argparse
import json
from collections import defaultdict

import numpy as np

from tprag.eval.analysis import auroc, compare, interval, rate, resample_weights
from tprag.eval.metrics import CONTEXT_BUDGET, budget_hit
from tprag.eval.retrieval import gold_spans, select_items
from tprag.eval.verify import EVAL_DIR, load_items
from tprag.index.build import IndexConfig
from tprag.normalize.store import load_normalized
from tprag.split.store import ROOT

SPLITTERS = ("s0", "s1", "s2", "s3", "char", "tok")
SIZE = 1024
BUDGETS = (1000, CONTEXT_BUDGET, 4000)
STUDY_DIR, EVAL_RESULTS, INDEX_DIR = ROOT / "results" / "study", ROOT / "results" / "eval", ROOT / "results" / "index"
SEED, N_BOOT = 2026, 10_000


def read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="e4")
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--packages", nargs="*", help="default: las de data/eval/corpus_subset.json")
    parser.add_argument("--only", nargs="*", choices=SPLITTERS, help="default: los que tengan resultados")
    parser.add_argument("--bootstrap", type=int, default=N_BOOT)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    packages = args.packages or json.loads((EVAL_DIR / "corpus_subset.json").read_text(encoding="utf-8"))["packages"]
    names = {s: IndexConfig(chunks=f"{s}-{SIZE}", model=args.model, packages=packages).collection_name()
             for s in (args.only or SPLITTERS)}
    detail = {s: EVAL_RESULTS / f"{n}__qA__{args.split}.targets.jsonl" for s, n in names.items()}
    names = {s: n for s, n in names.items() if detail[s].exists()}
    if not names:
        print("no hay resultados todavía en results/eval/")
        return 1
    print("splitters con resultados:", ", ".join(names))

    # Posiciones de los bloques gold (para recalcular el presupuesto con otros valores)
    items = select_items(load_items(), args.split, packages)
    gold_packages = sorted({g["package"] for it in items for t in it["targets"] for g in t["gold"]})
    spans = gold_spans(items, {d.id: d.page_content for d in load_normalized(packages=gold_packages)})
    item_index = {it["id"]: i for i, it in enumerate(items)}

    records = {s: {(r["id"], r["module"]): r for r in read_jsonl(detail[s])} for s in names}
    keys = sorted(next(iter(records.values())))
    for s, recs in records.items():
        if sorted(recs) != keys:
            raise ValueError(f"{s} no evaluó los mismos targets que el resto: no se pueden comparar por pares")
    positive = [k for k in keys if records[next(iter(names))][k]["acceptable_packages"]]
    negative = [k for k in keys if k not in set(positive)]
    gold = [k for k in positive if spans.get(k)]
    weights = resample_weights(len(items), args.bootstrap, SEED)
    of = lambda ks: np.array([item_index[k[0]] for k in ks])  # noqa: E731

    est: dict[str, dict[str, tuple[float, np.ndarray]]] = defaultdict(dict)  # splitter → métrica → (valor, bootstrap)
    for s, recs in records.items():
        pkg = np.array([recs[k]["metrics"]["package_rank"]["best_chunk"] == 1 for k in positive], dtype=float)
        est[s]["pkg_hit@1"] = rate(pkg, of(positive), weights)
        for b in BUDGETS:
            hits = np.array([budget_hit(recs[k]["hits"], spans[k], b) for k in gold], dtype=float)
            est[s][f"span_hit@{b}"] = rate(hits, of(gold), weights)
        pos_score = np.array([recs[k]["metrics"]["top1_score"] for k in positive])
        neg_score = np.array([recs[k]["metrics"]["top1_score"] for k in negative])
        if len(negative):
            est[s]["auroc"] = auroc(pos_score, neg_score, of(positive), of(negative), weights)

    # Por subconjunto: solo valores (los intervalos con tan pocos ítems no dicen nada)
    by_subset = {}
    for s, recs in records.items():
        out = {}
        for subset in sorted({recs[k]["subset"] for k in gold}):
            ks = [k for k in gold if recs[k]["subset"] == subset]
            out[subset] = {"targets": len(ks),
                           "pkg_hit@1": round(float(np.mean([recs[k]["metrics"]["package_rank"]["best_chunk"] == 1
                                                             for k in ks])), 4),
                           f"span_hit@{CONTEXT_BUDGET}": round(float(np.mean(
                               [budget_hit(recs[k]["hits"], spans[k], CONTEXT_BUDGET) for k in ks])), 4)}
        by_subset[s] = out

    # Contra el mejor de cada métrica
    paired = {}
    for metric in next(iter(est.values())):
        best = max(names, key=lambda s: est[s][metric][0])
        paired[metric] = {"reference": best, "vs": {
            s: compare(*est[s][metric], *est[best][metric]) for s in names if s != best}}

    # Costo y forma de los chunks
    extra = {}
    for s, n in names.items():
        index = json.loads((INDEX_DIR / f"{n}.json").read_text()) if (INDEX_DIR / f"{n}.json").exists() else {}
        stats = json.loads((STUDY_DIR / f"{n}__chunks.json").read_text()) if (STUDY_DIR / f"{n}__chunks.json").exists() else {}
        extra[s] = {
            "index_seconds": index.get("seconds"), "use_cases": index.get("use_cases"),
            "indexed_tokens": (stats.get("indexed") or {}).get("tokens_total"),
            "median_tokens": ((stats.get("indexed") or {}).get("n_tokens") or {}).get("median"),
            "python_blocks_not_intact": (stats.get("code_blocks", {}).get("python") or {}).get("not_intact_rate"),
            "truncated_by_model": (stats.get("model_tokens") or {}).get("over_max")}

    result = {
        "model": args.model, "split": args.split, "budgets": list(BUDGETS),
        "bootstrap": {"resamples": args.bootstrap, "seed": SEED, "level": 95, "unit": "ítems de dev"},
        "targets": {"positives": len(positive), "with_gold": len(gold), "negatives": len(negative), "items": len(items)},
        "splitters": {s: {**{m: {"value": round(v, 4), "ci": interval(b)} for m, (v, b) in est[s].items()},
                          "by_subset": by_subset[s], **extra[s]} for s in names},
        "paired": paired, "collections": names,
    }
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    (STUDY_DIR / "splitters_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                                       encoding="utf-8")
    table = markdown_table(result)
    (STUDY_DIR / "splitters_table.md").write_text(table, encoding="utf-8")
    print(table)
    if not args.no_plots:
        try:
            from tprag.eval.plots import splitter_figures  # noqa: PLC0415
        except ImportError as e:
            print(f"sin gráficos ({e}); instalar matplotlib")
        else:
            for path in splitter_figures(result, STUDY_DIR / "figures"):
                print("->", path)
    return 0


def markdown_table(result: dict) -> str:
    fmt = lambda m: f"{m['value']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]"  # noqa: E731
    head = (f"Etapa 1, {result['model']}, {result['split']}: {result['targets']['with_gold']} positivos, "
            f"{result['targets']['negatives']} negativos, {result['targets']['items']} ítems; "
            f"IC 95% bootstrap sobre ítems ({result['bootstrap']['resamples']} remuestreos)\n\n")
    rows = ["| Splitter | Pkg Hit@1 | Span@1000 | **Span@2000** | Span@4000 | AUROC | Chunks | Mediana tok | "
            "Bloques Py no enteros | Min indexado |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s, v in result["splitters"].items():
        pct = f"{v['python_blocks_not_intact']:.1%}" if v["python_blocks_not_intact"] is not None else "-"
        minutes = f"{v['index_seconds'] / 60:.1f}" if v["index_seconds"] else "-"
        rows.append(f"| {s} | {fmt(v['pkg_hit@1'])} | {fmt(v['span_hit@1000'])} | **{fmt(v['span_hit@2000'])}** | "
                    f"{fmt(v['span_hit@4000'])} | {fmt(v['auroc']) if 'auroc' in v else '-'} | {v['use_cases']} | "
                    f"{v['median_tokens']} | {pct} | {minutes} |")
    pairs = ["", "Diferencia por pares contra el mejor de cada métrica (empate = el IC de la diferencia incluye 0):", ""]
    for metric, p in result["paired"].items():
        pairs.append(f"- **{metric}** (mejor: {p['reference']}): " + "; ".join(
            f"{s} {d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] {d['verdict']}" for s, d in p["vs"].items()))
    return head + "\n".join(rows) + "\n" + "\n".join(pairs) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
