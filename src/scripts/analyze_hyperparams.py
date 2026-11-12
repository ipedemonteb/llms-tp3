"""Análisis de la etapa 3 (docs/agents/plan_experimentos_indice.md): compara las variantes de run_hyperparam_study.py
contra la base y entre niveles de un mismo factor. No usa API, embeddings ni Chroma: lee results/eval/ (el detalle con
los 50 chunks de cada target), results/index/ y results/study/.

Uso:
    uv run python src/scripts/analyze_hyperparams.py                # tabla + JSON + gráficos
    uv run --group analysis python src/scripts/analyze_hyperparams.py   # si falta matplotlib
    uv run python src/scripts/analyze_hyperparams.py --no-plots

Métricas (todas con IC 95% por bootstrap sobre los ítems de dev, remuestreos compartidos entre variantes, así que las
diferencias son por pares):
- Primarias: Package Hit@1 y Span Hit@2000. Span Hit se calcula también con 500, 1000, 3000 y 4000 tokens (curva).
- Apoyo: MRR de librería, nDCG@10, eficiencia de contexto@2000 (parte del contexto que es respuesta), AUROC y
  especificidad al 95% de recall (solo valor: con 9 negativos el IC no dice nada).
- Costo: chunks, tokens a embeber, mediana de tokens por chunk, bloques de Python partidos, textos truncados por el
  modelo. El tiempo de indexado depende del caché de vectores y no se compara.

Regla de decisión (fijada antes de correr): una variante reemplaza a la base solo si no es peor en Package Hit@1 y
mejora Span Hit@2000 de forma significativa (el IC de la diferencia pareada contra la base no incluye 0); si empatan,
decide el costo. Los finalistas (hasta 3) pasan a la evaluación con generación: ver tprag.eval.hyperparams.finalists.

Escribe results/study/hyperparams_analysis.json, hyperparams_table.md y, con matplotlib, figures/hyperparams/*.png.
"""

import argparse
import json

import numpy as np
from analyze_splitters import EVAL_RESULTS, INDEX_DIR, N_BOOT, SEED, STUDY_DIR, read_jsonl
from tprag.eval.analysis import auroc, compare, interval, rate, resample_weights
from tprag.eval.hyperparams import BASE, BUDGETS, FACTORS, MAIN_BUDGET, VARIANTS, finalists
from tprag.eval.metrics import budget_hit, context_efficiency, specificity_at_recall
from tprag.eval.retrieval import gold_spans, select_items
from tprag.eval.verify import EVAL_DIR, load_items
from tprag.normalize.store import load_normalized
from tprag.split.store import ROOT

PAIRED = ("pkg_hit@1", f"span_hit@{MAIN_BUDGET}", f"ctx_eff@{MAIN_BUDGET}", "ndcg@10")
GEN_DIR = EVAL_RESULTS / "gen"


def generation_summary(collection: str, split: str) -> dict | None:
    """Éxito de generación de una colección (results/eval/gen/), si ya se corrió eval_generation.py."""
    found = sorted(GEN_DIR.glob(f"{collection}__qA__*__{split}.json")) if GEN_DIR.exists() else []
    if not found:
        return None
    m = json.loads(found[-1].read_text(encoding="utf-8"))["metrics"]
    return {"positives": m["positives"]["success"], "negatives": m["negatives"]["success"]}


def generation_items(collection: str, split: str) -> dict[str, bool]:
    """id del ítem → éxito de generación (results/eval/gen/<colección>__qA__*.items.jsonl); vacío si no se corrió."""
    found = sorted(GEN_DIR.glob(f"{collection}__qA__*__{split}.items.jsonl")) if GEN_DIR.exists() else []
    return {r["id"]: bool(r["success"]) for r in read_jsonl(found[-1])} if found else {}


def cost(collection: str) -> dict:
    def load(path):
        return json.loads(path.read_text()) if path.exists() else {}

    index, stats = load(INDEX_DIR / f"{collection}.json"), load(STUDY_DIR / f"{collection}__chunks.json")
    indexed = stats.get("indexed") or {}
    return {
        "chunks": index.get("use_cases"), "tokens": indexed.get("tokens_total"),
        "median_tokens": (indexed.get("n_tokens") or {}).get("median"),
        "python_blocks_not_intact": (stats.get("code_blocks", {}).get("python") or {}).get("not_intact_rate"),
        "truncated_by_model": (stats.get("model_tokens") or {}).get("over_max")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--packages", nargs="*", help="default: las de data/eval/corpus_subset.json")
    parser.add_argument("--bootstrap", type=int, default=N_BOOT)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    packages = args.packages or json.loads((EVAL_DIR / "corpus_subset.json").read_text(encoding="utf-8"))["packages"]
    names = {v.id: v.index_config(packages).collection_name() for v in VARIANTS}
    detail = {i: EVAL_RESULTS / f"{n}__qA__{args.split}.targets.jsonl" for i, n in names.items()}
    names = {i: n for i, n in names.items() if detail[i].exists()}
    if "base" not in names:
        print("falta la base en results/eval/ (la deja la etapa 1: run_splitter_study.py)")
        return 1
    print("con resultados:", ", ".join(names))
    variants = {v.id: v for v in VARIANTS if v.id in names}

    items = select_items(load_items(), args.split, packages)
    gold_packages = sorted({g["package"] for it in items for t in it["targets"] for g in t["gold"]})
    spans = gold_spans(items, {d.id: d.page_content for d in load_normalized(packages=gold_packages)})
    item_index = {it["id"]: i for i, it in enumerate(items)}

    records = {i: {(r["id"], r["module"]): r for r in read_jsonl(detail[i])} for i in names}
    keys = sorted(records["base"])
    for i, recs in records.items():
        if sorted(recs) != keys:
            raise ValueError(f"{i} no evaluó los mismos targets que la base: no se pueden comparar por pares")
    positive = [k for k in keys if records["base"][k]["acceptable_packages"]]
    negative = [k for k in keys if k not in set(positive)]
    gold = [k for k in positive if spans.get(k)]
    weights = resample_weights(len(items), args.bootstrap, SEED)
    of = lambda ks: np.array([item_index[k[0]] for k in ks])  # noqa: E731

    est: dict[str, dict[str, tuple[float, np.ndarray]]] = {i: {} for i in names}
    point: dict[str, dict] = {i: {} for i in names}  # solo valores (sin IC)
    for i, recs in records.items():
        rank = lambda k: recs[k]["metrics"]["package_rank"]["best_chunk"]  # noqa: E731
        est[i]["pkg_hit@1"] = rate(np.array([rank(k) == 1 for k in positive], dtype=float), of(positive), weights)
        est[i]["pkg_mrr"] = rate(np.array([1 / rank(k) if rank(k) else 0.0 for k in positive]), of(positive), weights)
        est[i]["ndcg@10"] = rate(np.array([recs[k]["metrics"]["ndcg@10"] for k in gold]), of(gold), weights)
        for b in BUDGETS:
            est[i][f"span_hit@{b}"] = rate(np.array([budget_hit(recs[k]["hits"], spans[k], b) for k in gold],
                                                    dtype=float), of(gold), weights)
        est[i][f"ctx_eff@{MAIN_BUDGET}"] = rate(np.array([context_efficiency(recs[k]["hits"], spans[k], MAIN_BUDGET)
                                                          for k in gold]), of(gold), weights)
        pos_score = np.array([recs[k]["metrics"]["top1_score"] for k in positive])
        neg_score = np.array([recs[k]["metrics"]["top1_score"] for k in negative])
        if len(negative):
            est[i]["auroc"] = auroc(pos_score, neg_score, of(positive), of(negative), weights)
            point[i]["specificity@95"] = specificity_at_recall(list(pos_score), list(neg_score))

    # Contra la base, y entre niveles de un mismo factor (contra el mejor de cada métrica primaria)
    vs_base = {i: {m: compare(*est[i][m], *est["base"][m]) for m in PAIRED} for i in names if i != "base"}
    within = {}
    for factor, (title, ids, labels) in FACTORS.items():
        ids = [i for i in ids if i in names]
        if len(ids) < 2:
            continue
        within[factor] = {"title": title, "levels": ids, "labels": {i: labels[i] for i in ids}, "metrics": {}}
        for m in ("pkg_hit@1", f"span_hit@{MAIN_BUDGET}"):
            best = max(ids, key=lambda i: est[i][m][0])
            within[factor]["metrics"][m] = {"reference": best, "vs": {
                i: compare(*est[i][m], *est[best][m]) for i in ids if i != best}}

    # Generación (solo las configs que se corrieron): éxito en los ítems positivos con IC, y diferencia pareada contra la base
    gen_ok = {i: generation_items(n, args.split) for i, n in names.items()}
    pos_ids = [it["id"] for it in items if any(t["acceptable_packages"] for t in it["targets"])]
    gen_est = {i: rate(np.array([g[x] for x in pos_ids], dtype=float), np.array([item_index[x] for x in pos_ids]),
                       weights) for i, g in gen_ok.items() if g and all(x in g for x in pos_ids)}
    gen_extra = {i: {"items_success": round(gen_est[i][0], 4), "positives_ci": interval(gen_est[i][1]), "n_positives": len(pos_ids),
                     **({"vs_base": compare(*gen_est[i], *gen_est["base"])} if i != "base" and "base" in gen_est else {})}
                 for i in gen_est}

    costs = {i: cost(n) for i, n in names.items()}
    span_key = f"span_hit@{MAIN_BUDGET}"
    best_span = max(names, key=lambda i: est[i][span_key][0] if costs[i]["tokens"] else -1)
    rows = {i: {"span": est[i][span_key][0], "tokens": costs[i]["tokens"] or float("inf"),
                "pkg_verdict": "empate" if i == "base" else vs_base[i]["pkg_hit@1"]["verdict"],
                "span_vs_best": "empate" if i == best_span else compare(*est[i][span_key], *est[best_span][span_key])["verdict"]}
            for i in names}
    chosen = finalists(rows)

    result = {
        "split": args.split, "budgets": list(BUDGETS), "main_budget": MAIN_BUDGET,
        "bootstrap": {"resamples": args.bootstrap, "seed": SEED, "level": 95, "unit": "ítems de dev"},
        "targets": {"positives": len(positive), "with_gold": len(gold), "negatives": len(negative), "items": len(items)},
        "variants": {i: {"label": variants[i].label, "factor": variants[i].factor, "collection": names[i],
                         "chunks_dir": variants[i].chunks, "model": variants[i].model, "cap": variants[i].cap,
                         "representation": variants[i].representation,
                         "metrics": {m: {"value": round(v, 4), "ci": interval(b)} for m, (v, b) in est[i].items()},
                         **point[i], **costs[i], "generation": (generation_summary(names[i], args.split) or None) and {
                             **generation_summary(names[i], args.split), **gen_extra.get(i, {})}}
                     for i in names},
        "vs_base": vs_base, "within_factor": within, "finalists": chosen, "base_best_span": best_span,
    }
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    (STUDY_DIR / "hyperparams_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                                         encoding="utf-8")
    table = markdown_table(result)
    (STUDY_DIR / "hyperparams_table.md").write_text(table, encoding="utf-8")
    print(table)
    if not args.no_plots:
        try:
            from tprag.eval.plots_hyperparams import hyperparam_figures  # noqa: PLC0415
        except ImportError as e:
            print(f"sin gráficos ({e}); correr con `uv run --group analysis`")
        else:
            for path in hyperparam_figures(result, STUDY_DIR / "figures" / "hyperparams"):
                print("->", path.relative_to(ROOT))
    return 0


def markdown_table(result: dict) -> str:
    fmt = lambda m: f"{m['value']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]"  # noqa: E731
    t = result["targets"]
    head = (f"Etapa 3, hiperparámetros de s3 + e4, {result['split']}: {t['with_gold']} positivos, {t['negatives']} "
            f"negativos, {t['items']} ítems; IC 95% bootstrap sobre ítems ({result['bootstrap']['resamples']} "
            "remuestreos)\n\n")
    span = f"span_hit@{result['main_budget']}"
    rows = ["| Variante | Pkg Hit@1 | Span@1000 | **Span@2000** | Span@4000 | Ef. contexto@2000 | AUROC | Espec.@95 | "
            "Chunks | Tokens (M) | Mediana tok | Py partidos | Truncados | Gen. pos / neg |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, v in result["variants"].items():
        m, gen, spec = v["metrics"], v["generation"], v.get("specificity@95")
        cells = [
            f"{i}{' ⭐' if i in result['finalists'] else ''} · {v['label']}", fmt(m["pkg_hit@1"]), fmt(m["span_hit@1000"]),
            f"**{fmt(m[span])}**", fmt(m["span_hit@4000"]), fmt(m["ctx_eff@2000"]),
            fmt(m["auroc"]) if "auroc" in m else "-", f"{spec:.2f}" if spec is not None else "-",
            str(v["chunks"] or "-"), f"{v['tokens'] / 1e6:.2f}" if v["tokens"] else "-", str(v["median_tokens"] or "-"),
            f"{v['python_blocks_not_intact']:.1%}" if v["python_blocks_not_intact"] is not None else "-",
            str(v["truncated_by_model"]) if v["truncated_by_model"] is not None else "-",
            "{:.2f} / {:.2f}".format(gen["positives"], gen["negatives"]) if gen else "-"]
        rows.append("| " + " | ".join(cells) + " |")
    out = [head + "\n".join(rows), "", "Diferencia contra la base (pareada; empate = el IC de la diferencia incluye 0):", ""]
    for i, d in result["vs_base"].items():
        out.append(f"- **{i}**: " + "; ".join(
            f"{m} {x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] {x['verdict']}" for m, x in d.items()))
    out += ["", "Dentro de cada factor (contra el mejor nivel de ese factor):", ""]
    for f, w in result["within_factor"].items():
        for m, p in w["metrics"].items():
            out.append(f"- **{w['title']} · {m}** (mejor: {w['labels'][p['reference']]}): " + "; ".join(
                f"{w['labels'][i]} {d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] {d['verdict']}"
                for i, d in p["vs"].items()))
    out += ["", f"Finalistas para la evaluación con generación (⭐): {', '.join(result['finalists']) or 'ninguno'}. "
            "Correr: `uv run --group local python src/scripts/run_hyperparam_study.py --generation`."]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
