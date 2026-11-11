"""Evalúa el retrieval de una colección de Chroma con el dataset de data/eval/ (paso 7, docs/agents/evaluacion.md).
USA LA API (solo para embeber las consultas que no están en el caché).

Uso:
    uv run python src/scripts/eval_retrieval.py --dry-run                     # solo estima, sin API
    uv run python src/scripts/eval_retrieval.py                               # s3-1024, consulta A, dev
    uv run python src/scripts/eval_retrieval.py --chunks s0-1024              # otro índice, mismas consultas (0 USD)
    uv run python src/scripts/eval_retrieval.py --split test                  # test: solo al final
    uv run python src/scripts/eval_retrieval.py --budget 4000                 # Span Hit con otro presupuesto (default 2000)

La colección se elige con los mismos flags que build_index.py y tiene que estar armada. Necesita la doc normalizada
(data/normalized/) para ubicar los bloques gold. Escribe en results/eval/ (no versionado) el resumen
<colección>__q<consulta>__<split>.json (con otro --budget, sufijo __b<N>) y el detalle por target <...>.targets.jsonl.
"""

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime

from dotenv import load_dotenv

from tprag.eval.metrics import CONTEXT_BUDGET, summarize
from tprag.eval.query import QUERIES
from tprag.eval.retrieval import N_RESULTS, build_targets, dataset_info, gold_spans, run, select_items
from tprag.eval.verify import load_items
from tprag.index.build import REPRESENTATIONS, IndexConfig, open_collection
from tprag.index.cost import estimate
from tprag.index.embeddings import MODELS, cached
from tprag.normalize.store import load_normalized
from tprag.split.select import DEFAULT_CAP, STRATEGIES
from tprag.split.store import ROOT

RESULTS_DIR = ROOT / "results" / "eval"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="s3-1024", help="carpeta de data/processed/<modo>/")
    parser.add_argument("--console-examples", default="outputs_as_comments", help="carpeta de data/processed")
    parser.add_argument("--model", choices=sorted(MODELS), default="e1")
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="r1")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP, help="0 = sin tope")
    parser.add_argument("--strategy", choices=STRATEGIES, default="round_robin")
    parser.add_argument("--packages", nargs="*", help="colección armada solo con estas librerías")
    parser.add_argument("--split", choices=("dev", "test"), default="dev", help="test solo para el reporte final")
    parser.add_argument("--query", choices=sorted(QUERIES), default="A", help="forma de armar la consulta")
    parser.add_argument("--budget", type=int, default=CONTEXT_BUDGET,
                        help="tokens de contexto para el Span Hit con presupuesto (el generador usa el default)")
    parser.add_argument("--max-usd", type=float, default=0.05, help="aborta si lo que hay que pagar lo supera")
    parser.add_argument("--dry-run", action="store_true", help="solo cuenta tokens y estima el costo")
    args = parser.parse_args()

    cfg = IndexConfig(chunks=args.chunks, console_mode=args.console_examples, model=args.model,
                      representation=args.representation, cap=args.cap or None, strategy=args.strategy,
                      packages=args.packages)
    model = MODELS[cfg.model]
    name = cfg.collection_name()
    items = select_items(load_items(), args.split, args.packages)
    dropped = sum(len(it["targets"]) for it in select_items(load_items(), args.split)) - sum(
        len(it["targets"]) for it in items)
    records = build_targets(items, args.query)
    queries = [r["query"] for r in records]
    embeddings = cached(cfg.model)
    total, to_pay = estimate(queries, model), estimate(embeddings.missing(queries), model)
    print(f"{name} / consulta {args.query} / {args.split}: {len(items)} ítems, {len(records)} targets, "
          f"{total['tokens']} tokens" + (f" ({dropped} targets descartados: su respuesta está fuera de las librerías "
                                         "del índice)" if dropped else ""))
    print(f"a pagar (fuera del caché): {to_pay['texts']} consultas, {to_pay['tokens']} tokens, {to_pay['usd']:.6f} USD "
          f"({model.name}, {model.usd_per_m_tokens} USD/M)")
    if args.dry_run:
        return 0
    if to_pay["usd"] > args.max_usd:
        print(f"abortado: supera --max-usd {args.max_usd}")
        return 1

    store = open_collection(name, embeddings)
    if store._collection.count() == 0:  # open_collection crea la colección vacía si no existe
        print(f"la colección {name} no existe o está vacía: armarla con build_index.py")
        return 1
    packages = sorted({g["package"] for it in items for t in it["targets"] for g in t["gold"]})
    docs = {d.id: d.page_content for d in load_normalized(cfg.console_mode, packages=packages)}
    spans = gold_spans(items, docs)

    load_dotenv(ROOT / ".env")
    start = time.time()
    run(store, records, spans, args.budget)
    seconds = round(time.time() - start, 1)

    report = {
        "collection": name, "query": args.query, "split": args.split,
        "date": datetime.now().isoformat(timespec="seconds"), "config": asdict(cfg), "dataset": dataset_info(),
        "budget": args.budget, "items": len(items), "targets_dropped_outside_packages": dropped, "n_results": N_RESULTS, "model": model.name,
        "paid": {**to_pay, "api_calls_texts": embeddings.misses}, "cache_hits": embeddings.hits, "seconds": seconds,
        "metrics": summarize(records, args.budget),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{name}__q{args.query}__{args.split}" + ("" if args.budget == CONTEXT_BUDGET else f"__b{args.budget}")
    summary = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    (RESULTS_DIR / f"{stem}.json").write_text(summary, encoding="utf-8")
    with (RESULTS_DIR / f"{stem}.targets.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    best = report["metrics"]["all"]
    print(f"listo en {seconds} s: {embeddings.misses} consultas a la API, {embeddings.hits} del caché")
    print(f"mejor chunk: Package Hit@1 {best['package']['best_chunk']['hit@1']}, "
          f"MRR {best['package']['best_chunk']['mrr']}, Span Hit@5 {best['span']['hit@5']}, "
          f"AUROC {report['metrics']['auroc']['value']}")
    if (th := report["metrics"]["threshold"]) is not None:
        print(f"umbral ({th['min_positive_pass']:.0%} de positivos): {th['value']} -> pasan {th['positives_pass']}/"
              f"{th['positives']} positivos y {th['negatives_pass']}/{th['negatives']} negativos")
    print(f"-> {RESULTS_DIR / stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
