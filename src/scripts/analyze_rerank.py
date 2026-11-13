"""Análisis del estudio del reranker (docs/agents/plan_reranker.md). No usa modelos, API ni Chroma: lee el detalle de la
base (results/eval/, los 50 chunks de cada target), los puntajes que dejó run_rerank_study.py (results/eval/rerank/) y
su manifiesto (tiempos).

Uso:
    uv run --group analysis python src/scripts/analyze_rerank.py         # tabla + JSON + gráficos
    uv run python src/scripts/analyze_rerank.py --no-plots

Por cada variante (reranker × K × modo, grilla en tprag.eval.rerank_study) reordena los candidatos de cada target,
recalcula las métricas del paso 7 y escribe su detalle en results/eval/rerank/<colección>__qA__<variante>__<split>.
targets.jsonl. Después compara contra la base.

Métricas (IC 95% por bootstrap sobre los ítems de dev, remuestreos compartidos: las diferencias son por pares):
- Primarias: Package Hit@1 y Span Hit@2000.
- Apoyo: Span Hit con 500-4000 tokens, Span Hit@1/3/5, nDCG@10, MRR de librería, eficiencia de contexto@2000.
- Antes/después: aciertos rescatados y rotos respecto de la base, y rescatados por tipo de fallo de la base.
- Confianza: AUROC con el coseno del top-1 y con el máximo del reranker entre los K candidatos; especificidad al 95%
  de recall (solo valor: con 9 negativos el IC no dice nada).
- Costo: segundos por consulta = K × segundos por par (medidos en la corrida), pares truncados.

Regla de decisión (fijada antes de correr, tprag.eval.rerank_study.choose): una variante reemplaza a la base solo si
mejora Span Hit@2000 de forma significativa y no es peor en Package Hit@1; entre esas, la más barata.

Umbral de abstención (plan_cierre.md §3): el más alto que deja pasar al menos el 95% de los positivos, con el coseno del
top-1 (base) y con el máximo del reranker (cada variante).

Además: comparaciones pareadas entre variantes (`--pairs`, por defecto las del desvío del estudio), Pkg Hit@1 y
Span@2000 por subconjunto (manual / sintético) y, con `--compare-with`, el cambio respecto de otro análisis.

Escribe results/study/rerank_analysis.json, rerank_table.md y, con matplotlib, figures/rerank/*.png. Con
`--corpus full` (las 101 librerías) los nombres llevan el prefijo rerank_full_<split>_ y no pisan los del estudio:
    uv run --group analysis python src/scripts/analyze_rerank.py --corpus full
Con `--tag` el prefijo suma el tag y nunca pisa nada (falla si ya existe); el detalle por variante solo se escribe si
no existía o si sale idéntico (plan_cierre.md §9):
    uv run --group analysis python src/scripts/analyze_rerank.py --corpus full --tag estudio
"""

import argparse
import json
from pathlib import Path

import numpy as np
from analyze_splitters import EVAL_RESULTS, N_BOOT, SEED, STUDY_DIR, read_jsonl
from run_rerank_study import corpus_packages, manifest_path
from tprag.eval.analysis import auroc, compare, interval, rate, resample_weights
from tprag.eval.hyperparams import BUDGETS, MAIN_BUDGET
from tprag.eval.metrics import (SPAN_KS, budget_hit, calibrate_threshold, context_efficiency, specificity_at_recall,
                                target_metrics)
from tprag.eval.rerank_study import (BASE_VARIANT, DEFAULT_PAIRS, KS, MODES, POOL, VARIANTS, choose, parse_pairs,
                                     scores_path, targets_path, write_jsonl_once)
from tprag.eval.retrieval import gold_spans, select_items
from tprag.eval.verify import load_items
from tprag.index.rerank import RERANKERS
from tprag.normalize.store import load_normalized
from tprag.split.store import ROOT

SPAN, PKG = f"span_hit@{MAIN_BUDGET}", "pkg_hit@1"
PAIRED = (PKG, SPAN, f"ctx_eff@{MAIN_BUDGET}", "ndcg@10")
PAIRED_BETWEEN = (PKG, SPAN, "span_hit@1000", "ndcg@10")  # las de rerank_choice.json (plan_reranker.md §7)
FAILURES = {"over_budget": "gold fuera del presupuesto", "wrong_package": "otra librería en el top-1",
            "not_retrieved": "gold no recuperado"}

Key = tuple[str, str]


def base_failure(rec: dict, span_ok: bool) -> str | None:
    """Tipo de fallo de la base en Span Hit@2000 (el mismo orden que el análisis de errores del 2026-11-11)."""
    if span_ok:
        return None
    if rec["hits"][0]["package"] not in rec["acceptable_packages"]:
        return "wrong_package"
    return "not_retrieved" if rec["metrics"]["span_rank"] is None else "over_budget"


def load_scores(collection: str, reranker: str, split: str, base: dict[Key, dict]) -> dict[Key, list[float]] | None:
    path = scores_path(collection, reranker, split)
    if not path.exists():
        return None
    out = {}
    for row in read_jsonl(path):
        key = (row["id"], row["module"])
        if key not in base:
            raise ValueError(f"{path.name}: {key} no está en la base")
        if row["chunk_ids"] != [h["id"] for h in base[key]["hits"][:POOL]]:
            raise ValueError(f"{path.name}: los candidatos de {key} no son los de la base (¿se volvió a evaluar?)")
        out[key] = row["scores"]
    if set(out) != set(base):
        raise ValueError(f"{path.name} no puntuó los mismos targets que la base")
    return out


def estimates(recs: dict[Key, dict], positive: list[Key], negative: list[Key], gold: list[Key],
              spans: dict[Key, list], of, weights, conf: dict[Key, float] | None = None) -> tuple[dict, dict]:
    """Métricas con su distribución bootstrap (`est`) y valores sin IC (`point`). `conf`: puntaje de confianza del
    reranker por target, para el AUROC con ese puntaje."""
    est, point = {}, {}
    rank = lambda k: recs[k]["metrics"]["package_rank"]["best_chunk"]  # noqa: E731
    est[PKG] = rate(np.array([rank(k) == 1 for k in positive], dtype=float), of(positive), weights)
    est["pkg_mrr"] = rate(np.array([1 / rank(k) if rank(k) else 0.0 for k in positive]), of(positive), weights)
    est["ndcg@10"] = rate(np.array([recs[k]["metrics"]["ndcg@10"] for k in gold]), of(gold), weights)
    for n in SPAN_KS[:3]:
        sr = lambda k: recs[k]["metrics"]["span_rank"]  # noqa: E731
        est[f"span_hit@{n}"] = rate(np.array([sr(k) is not None and sr(k) <= n for k in gold], dtype=float), of(gold),
                                    weights)
    for b in BUDGETS:
        est[f"span_hit@{b}"] = rate(np.array([budget_hit(recs[k]["hits"], spans[k], b) for k in gold], dtype=float),
                                    of(gold), weights)
    est[f"ctx_eff@{MAIN_BUDGET}"] = rate(np.array([context_efficiency(recs[k]["hits"], spans[k], MAIN_BUDGET)
                                                   for k in gold]), of(gold), weights)
    if negative:
        for name, score in (("cos", lambda k: recs[k]["metrics"]["top1_score"]), ("rr", conf and conf.get)):
            if score is None:
                continue
            pos, neg = np.array([score(k) for k in positive]), np.array([score(k) for k in negative])
            est[f"auroc_{name}"] = auroc(pos, neg, of(positive), of(negative), weights)
            point[f"specificity@95_{name}"] = specificity_at_recall(list(pos), list(neg))
            point[f"threshold_{name}"] = calibrate_threshold([float(x) for x in pos], [float(x) for x in neg])
    return est, point


def transitions(base: dict[Key, dict], recs: dict[Key, dict], positive: list[Key], gold: list[Key],
                spans: dict[Key, list]) -> dict:
    """Aciertos que la variante rescata y rompe respecto de la base, en Span Hit@2000 y en Package Hit@1."""
    span_base = {k: budget_hit(base[k]["hits"], spans[k], MAIN_BUDGET) for k in gold}
    span_var = {k: budget_hit(recs[k]["hits"], spans[k], MAIN_BUDGET) for k in gold}
    by_type = {t: 0 for t in FAILURES}
    for k in gold:
        if span_var[k] and not span_base[k]:
            by_type[base_failure(base[k], False)] += 1
    pkg = lambda recs_, k: recs_[k]["metrics"]["package_rank"]["best_chunk"] == 1  # noqa: E731
    return {
        "span": {"rescued": sum(span_var[k] and not span_base[k] for k in gold),
                 "broken": sum(span_base[k] and not span_var[k] for k in gold), "rescued_by_failure": by_type},
        "pkg": {"rescued": sum(pkg(recs, k) and not pkg(base, k) for k in positive),
                "broken": sum(pkg(base, k) and not pkg(recs, k) for k in positive)},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--corpus", choices=("subset", "full"), default="subset",
                        help="subset: las 30 de corpus_subset.json (estudio); full: las 101 librerías")
    parser.add_argument("--packages", nargs="*", help="pisa --corpus: solo estas librerías")
    parser.add_argument("--tag", help="salidas con prefijo rerank_<corpus>_<split>_<tag> (y su manifiesto), sin pisar "
                                      "las de corridas anteriores")
    parser.add_argument("--pairs", nargs="*", default=list(DEFAULT_PAIRS),
                        help="comparaciones pareadas entre variantes, A:B (default: las del desvío del estudio)")
    parser.add_argument("--compare-with", type=Path,
                        help="otro *_analysis.json (p. ej. el del subconjunto) para el gráfico 4; con --corpus full y "
                             "--tag usa results/study/rerank_analysis.json")
    parser.add_argument("--bootstrap", type=int, default=N_BOOT)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    packages = corpus_packages(args.corpus, args.packages)
    collection = BASE_VARIANT.index_config(packages).collection_name()
    prefix = ("rerank" if args.corpus == "subset" and not args.tag
              else "_".join(["rerank", args.corpus, args.split] + ([args.tag] if args.tag else [])))
    if args.tag and (STUDY_DIR / f"{prefix}_analysis.json").exists():
        print(f"{prefix}_analysis.json ya existe: usar otro --tag (no se pisa)")
        return 1
    pairs = parse_pairs(args.pairs)
    compare_with = args.compare_with or (STUDY_DIR / "rerank_analysis.json"
                                         if args.corpus == "full" and args.tag else None)
    base_path = EVAL_RESULTS / f"{collection}__qA__{args.split}.targets.jsonl"
    if not base_path.exists():
        print(f"falta {base_path.name}: correr eval_retrieval.py sobre la base")
        return 1
    base = {(r["id"], r["module"]): r for r in read_jsonl(base_path)}
    scores = {r: s for r in RERANKERS if (s := load_scores(collection, r, args.split, base)) is not None}
    if not scores:
        print("no hay puntajes: correr run_rerank_study.py")
        return 1
    print("con puntajes:", ", ".join(scores))
    manifest_file = manifest_path(args.corpus, args.split, args.tag)
    if args.tag and not manifest_file.exists():  # el runner no se corrió con este tag: los tiempos del original
        manifest_file = manifest_path(args.corpus, args.split)
    manifest = (json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.exists()
                else {"rerankers": {}})

    items = select_items(load_items(), args.split, packages)
    gold_packages = sorted({g["package"] for it in items for t in it["targets"] for g in t["gold"]})
    spans = gold_spans(items, {d.id: d.page_content for d in load_normalized(packages=gold_packages)})
    item_index = {it["id"]: i for i, it in enumerate(items)}
    keys = sorted(base)
    positive = [k for k in keys if base[k]["acceptable_packages"]]
    negative = [k for k in keys if not base[k]["acceptable_packages"]]
    gold = [k for k in positive if spans.get(k)]
    weights = resample_weights(len(items), args.bootstrap, SEED)
    of = lambda ks: np.array([item_index[k[0]] for k in ks])  # noqa: E731

    # Variantes: reordenar, recalcular las métricas del paso 7 y guardar el detalle
    variants = [v for v in VARIANTS if v.reranker in scores]
    records: dict[str, dict[Key, dict]] = {"base": base}
    conf: dict[str, dict[Key, float]] = {}
    written = {"nuevo": 0, "igual": 0}
    for v in variants:
        recs = {}
        for k in keys:
            hits = v.apply(base[k]["hits"], scores[v.reranker][k])
            recs[k] = {**base[k], "hits": hits, "variant": v.id,
                       "metrics": target_metrics(hits, base[k]["acceptable_packages"], spans.get(k, []), MAIN_BUDGET)}
        records[v.id] = recs
        conf[v.id] = {k: max(scores[v.reranker][k][:v.k]) for k in keys}
        # el detalle de una variante es determinista: si ya estaba (paso 3) tiene que salir igual, y no se pisa
        written[write_jsonl_once(targets_path(collection, v.id, args.split), [recs[k] for k in keys])] += 1
    print(f"detalle por variante: {written['nuevo']} nuevos, {written['igual']} ya estaban iguales")

    est, point = {}, {}
    for i, recs in records.items():
        est[i], point[i] = estimates(recs, positive, negative, gold, spans, of, weights, conf.get(i))
    vs_base = {v.id: {m: compare(*est[v.id][m], *est["base"][m]) for m in PAIRED} for v in variants}
    ids = {v.id for v in variants}
    between = {f"{a} vs {b}": {m: compare(*est[a][m], *est[b][m]) for m in PAIRED_BETWEEN}
               for a, b in pairs if a in ids and b in ids}
    subsets = sorted({base[k]["subset"] for k in positive})
    by_subset = {}
    for i, recs in records.items():
        by_subset[i] = {}
        for s in subsets:
            pos_s, gold_s = [k for k in positive if base[k]["subset"] == s], [k for k in gold if base[k]["subset"] == s]
            e, _ = estimates(recs, pos_s, [], gold_s, spans, of, weights)
            by_subset[i][s] = {"targets": len(pos_s), **{m: {"value": round(e[m][0], 4), "ci": interval(e[m][1])}
                                                         for m in (PKG, SPAN)}}
    moves = {v.id: transitions(base, records[v.id], positive, gold, spans) for v in variants}
    base_failures = {t: sum(base_failure(base[k], budget_hit(base[k]["hits"], spans[k], MAIN_BUDGET)) == t
                            for k in gold) for t in FAILURES}

    def cost(v) -> dict:
        info = manifest["rerankers"].get(v.reranker, {})
        per_pair = (info.get("timing") or {}).get("seconds_per_pair")
        return {"seconds_per_pair": per_pair, "seconds_per_query": round(v.k * per_pair, 2) if per_pair else None,
                "truncated_rate": info.get("truncated_rate"), "peak_mps_mb": (info.get("timing") or {}).get("peak_mps_mb")}

    costs = {v.id: cost(v) for v in variants}
    rows = {v.id: {"span_verdict": vs_base[v.id][SPAN]["verdict"], "pkg_verdict": vs_base[v.id][PKG]["verdict"],
                   "seconds": costs[v.id]["seconds_per_query"] or float("inf"), "span": est[v.id][SPAN][0]}
            for v in variants}
    chosen = choose(rows)

    metrics = lambda i: {m: {"value": round(val, 4), "ci": interval(b)} for m, (val, b) in est[i].items()}  # noqa: E731
    result = {
        "split": args.split, "collection": collection, "pool": POOL, "ks": list(KS), "modes": MODES,
        "budgets": list(BUDGETS), "main_budget": MAIN_BUDGET,
        "bootstrap": {"resamples": args.bootstrap, "seed": SEED, "level": 95, "unit": "ítems de dev"},
        "targets": {"positives": len(positive), "with_gold": len(gold), "negatives": len(negative), "items": len(items)},
        "base": {"label": "sin reranker (S3 · 1024 · E4 · R1 · tope 150)", "metrics": metrics("base"), **point["base"],
                 "failures": base_failures},
        "rerankers": {r: {"name": RERANKERS[r].name, "label": RERANKERS[r].label,
                          "instruction": RERANKERS[r].instruction, **manifest["rerankers"].get(r, {})} for r in scores},
        "variants": {v.id: {"label": v.label, "reranker": v.reranker, "k": v.k, "mode": v.mode,
                            "metrics": metrics(v.id), **point[v.id], **costs[v.id], "transitions": moves[v.id]}
                     for v in variants},
        "vs_base": vs_base, "chosen": chosen, "pairs": between, "by_subset": by_subset,
    }
    if compare_with:
        other = json.loads(compare_with.read_text(encoding="utf-8"))
        result["compared_with"] = {
            "file": str(compare_with.relative_to(ROOT)) if compare_with.is_relative_to(ROOT) else str(compare_with),
            "collection": other["collection"], "targets": other["targets"],
            "metrics": {i: {m: v["metrics"][m] for m in (PKG, SPAN)}
                        for i, v in [("base", other["base"])] + list(other["variants"].items())}}
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    (STUDY_DIR / f"{prefix}_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                                    encoding="utf-8")
    table = markdown_table(result)
    (STUDY_DIR / f"{prefix}_table.md").write_text(table, encoding="utf-8")
    print(table)
    if not args.no_plots:
        try:
            from tprag.eval.plots_rerank import rerank_figures  # noqa: PLC0415
        except ImportError as e:
            print(f"sin gráficos ({e}); correr con `uv run --group analysis`")
        else:
            for path in rerank_figures(result, STUDY_DIR / "figures" / prefix):
                print("->", path.relative_to(ROOT))
    return 0


def markdown_table(result: dict) -> str:
    fmt = lambda m: f"{m['value']:.3f} [{m['ci'][0]:.3f}, {m['ci'][1]:.3f}]"  # noqa: E731
    opt = lambda m, k: fmt(m[k]) if k in m else "-"  # noqa: E731
    t = result["targets"]
    head = (f"Reranker sobre {result['collection']}, {result['split']}: {t['with_gold']} positivos, {t['negatives']} "
            f"negativos, {t['items']} ítems; {result['pool']} candidatos puntuados; IC 95% bootstrap sobre ítems "
            f"({result['bootstrap']['resamples']} remuestreos)\n\n")
    rows = ["| Variante | Pkg Hit@1 | Span@1000 | **Span@2000** | Span@4000 | nDCG@10 | Ef. contexto@2000 | "
            "AUROC coseno | AUROC reranker | s / consulta | Span rescata / rompe | Pkg rescata / rompe |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    entries = [("base", result["base"])] + list(result["variants"].items())
    for i, v in entries:
        m, tr = v["metrics"], v.get("transitions")
        star = " ⭐" if i == result["chosen"] else ""
        cells = [f"{i}{star} · {v['label']}", fmt(m[PKG]), fmt(m["span_hit@1000"]), f"**{fmt(m[SPAN])}**",
                 fmt(m["span_hit@4000"]), fmt(m["ndcg@10"]), fmt(m[f"ctx_eff@{MAIN_BUDGET}"]), opt(m, "auroc_cos"),
                 opt(m, "auroc_rr"), f"{v['seconds_per_query']:.2f}" if v.get("seconds_per_query") else "-",
                 f"+{tr['span']['rescued']} / −{tr['span']['broken']}" if tr else "-",
                 f"+{tr['pkg']['rescued']} / −{tr['pkg']['broken']}" if tr else "-"]
        rows.append("| " + " | ".join(cells) + " |")
    out = [head + "\n".join(rows), "", "Diferencia contra la base (pareada; empate = el IC de la diferencia incluye 0):", ""]
    for i, d in result["vs_base"].items():
        out.append(f"- **{i}**: " + "; ".join(
            f"{m} {x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] {x['verdict']}" for m, x in d.items()))
    fails = result["base"]["failures"]
    out += ["", "Fallos de la base en Span@2000 y cuántos rescata cada variante, por tipo ("
            + ", ".join(f"{FAILURES[f]}: {n}" for f, n in fails.items()) + "):", ""]
    for i, v in result["variants"].items():
        r = v["transitions"]["span"]["rescued_by_failure"]
        out.append(f"- **{i}**: " + ", ".join(f"{FAILURES[f]} {r[f]}/{fails[f]}" for f in FAILURES))
    out += ["", "Umbral al 95% de los positivos (coseno del top-1 en la base, máximo del reranker en las variantes):", ""]
    for i, v in entries:
        th = v.get("threshold_rr" if i != "base" else "threshold_cos")
        if th:
            out.append(f"- **{i}**: {th['value']:.4f} -> pasan {th['positives_pass']}/{th['positives']} positivos y "
                       f"{th['negatives_pass']}/{th['negatives']} negativos")
    if result.get("pairs"):
        out += ["", "Comparaciones pareadas entre variantes (A − B; empate = el IC incluye 0):", ""]
        for name, d in result["pairs"].items():
            out.append(f"- **{name}**: " + "; ".join(
                f"{m} {x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] {x['verdict']}" for m, x in d.items()))
    if result.get("by_subset"):
        subsets = list(next(iter(result["by_subset"].values())))
        out += ["", "Por subconjunto (Pkg Hit@1 / Span@2000):", "",
                "| Variante | " + " | ".join(f"{s} ({result['by_subset']['base'][s]['targets']})" for s in subsets)
                + " |", "|---|" + "---|" * len(subsets)]
        for i, d in result["by_subset"].items():
            out.append(f"| {i} | " + " | ".join(f"{d[s][PKG]['value']:.3f} / {d[s][SPAN]['value']:.3f}"
                                                 for s in subsets) + " |")
    if result.get("compared_with"):
        c = result["compared_with"]
        out += ["", f"Contra {c['file']} ({c['collection']}; Pkg Hit@1 / Span@2000, otro → este):", ""]
        for i, m in c["metrics"].items():
            here = result["base"] if i == "base" else result["variants"].get(i)
            if here:
                out.append(f"- **{i}**: {m[PKG]['value']:.3f} → {here['metrics'][PKG]['value']:.3f} / "
                           f"{m[SPAN]['value']:.3f} → {here['metrics'][SPAN]['value']:.3f}")
    out += ["", "Costo por reranker:", ""]
    for r, info in result["rerankers"].items():
        timing = info.get("timing") or {}
        out.append(f"- **{r}** {info['name']}: {timing.get('seconds_per_pair') or 0:.3f} s/par, "
                   f"{timing.get('seconds', '-')} s en total, GPU máx. {timing.get('peak_mps_mb', '-')} MB, "
                   f"truncados {info.get('truncated_rate', '-')}")
    out += ["", "Decisión (regla fijada antes de correr): "
            + (f"**{result['chosen']}** reemplaza a la base." if result["chosen"]
               else "**ninguna variante** mejora Span@2000 de forma significativa sin empeorar Pkg Hit@1: "
                    "el reranker no se integra.")]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
