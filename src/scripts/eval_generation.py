"""Evalúa el generador con el dataset de data/eval/ (paso 8, docs/agents/generador.md §2.6). USA LA API (el modelo de
chat, y embeddings para las consultas que no están en el caché).

Uso:
    uv run python src/scripts/eval_generation.py --dry-run          # solo estima, sin API
    uv run python src/scripts/eval_generation.py                    # s3-1024, gpt-4o-mini, dev
    uv run python src/scripts/eval_generation.py --split test       # test: solo al final
    uv run --group local python src/scripts/eval_generation.py --model e4 --rerank k1-rr30 --dry-run   # con reranker

La colección se elige con los mismos flags que build_index.py. El umbral sale de la corrida de eval_retrieval.py en dev
con esa colección (o --threshold). Las respuestas del LLM quedan en caché (data/generate/): repetir una corrida no
vuelve a pagar. Escribe en results/eval/gen/ (no versionado) el resumen y el detalle por ítem (código corregido,
reporte, reemplazos y chequeos).
"""

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime

from dotenv import load_dotenv

from tprag.eval.generation import check_item, summarize
from tprag.eval.query import QUERIES
from tprag.eval.retrieval import dataset_info, select_items
from tprag.eval.verify import load_items
from tprag.generate.context import dev_threshold, load_reranker, rerank_threshold
from tprag.generate.llm import CHAT_MODELS, CachedLLM
from tprag.generate.pipeline import finish, prepare
from tprag.index.build import REPRESENTATIONS, IndexConfig, open_collection
from tprag.index.cost import estimate
from tprag.index.embeddings import MODELS, cached
from tprag.split.select import DEFAULT_CAP, STRATEGIES
from tprag.split.store import ROOT
from tprag.validate.pypi import CachedPyPI
from tprag.validate.validate import validate

RESULTS_DIR = ROOT / "results" / "eval" / "gen"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="s3-1024", help="carpeta de data/processed/<modo>/")
    parser.add_argument("--console-examples", default="outputs_as_comments", help="carpeta de data/processed")
    parser.add_argument("--model", choices=sorted(MODELS), default="e1", help="modelo de embeddings del índice")
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="r1")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP, help="0 = sin tope")
    parser.add_argument("--strategy", choices=STRATEGIES, default="round_robin")
    parser.add_argument("--packages", nargs="*", help="colección armada solo con estas librerías")
    parser.add_argument("--llm", choices=sorted(CHAT_MODELS), default="gpt-4o-mini")
    parser.add_argument("--threshold", type=float, help="umbral de confianza (default: el calibrado en dev)")
    parser.add_argument("--budget", type=int, default=2000, help="tokens de contexto por import")
    parser.add_argument("--rerank", help="variante del reranker (p. ej. k1-rr30; necesita el grupo `local`); el umbral "
                                         "pasa a ser el de su puntaje, calibrado en dev")
    parser.add_argument("--split", choices=("dev", "test"), default="dev", help="test solo para el reporte final")
    parser.add_argument("--max-usd", type=float, default=0.10, help="aborta si lo que hay que pagar lo supera")
    parser.add_argument("--offline", action="store_true", help="no consulta PyPI; lo que no está en caché: unknown")
    parser.add_argument("--dry-run", action="store_true", help="solo cuenta tokens y estima el costo")
    args = parser.parse_args()

    cfg = IndexConfig(chunks=args.chunks, console_mode=args.console_examples, model=args.model,
                      representation=args.representation, cap=args.cap or None, strategy=args.strategy,
                      packages=args.packages)
    name = cfg.collection_name()
    reranker = load_reranker(args.rerank) if args.rerank else None
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = rerank_threshold(name, args.rerank) if args.rerank else dev_threshold(name)
    embeddings = cached(cfg.model)
    store = open_collection(name, embeddings)
    if store._collection.count() == 0:  # open_collection crea la colección vacía si no existe
        print(f"la colección {name} no existe o está vacía: armarla con build_index.py")
        return 1
    fetch = CachedPyPI(offline=args.offline)
    llm = CachedLLM(CHAT_MODELS[args.llm])
    items = select_items(load_items(), args.split, args.packages)

    # Las consultas al índice son las del paso 7: si ya se corrió eval_retrieval.py, están en el caché (0 USD).
    queries = [QUERIES["A"](it["code"], r.module) for it in items for r in validate(it["code"], fetch) if r.to_rag]
    emb = estimate(embeddings.missing(queries), MODELS[cfg.model])
    if emb["texts"]:
        print(f"consultas fuera del caché de embeddings: {emb['texts']} ({emb['usd']:.6f} USD)")
        if args.dry_run:
            print("dry-run: el contexto necesita embeberlas; correr antes eval_retrieval.py con esta colección")
            return 0
    load_dotenv(ROOT / ".env")

    if reranker:  # el dry-run también reordena: los pares que no estén en el caché se calculan con el modelo local
        print(f"reranker {reranker.variant}: puntúa {reranker.k} chunks por import (los del caché no se recalculan)")
    preps = [prepare(it["code"], store, fetch, threshold, args.budget, reranker) for it in items]
    prompts = [p.prompt for p in preps if p.prompt]
    to_pay = llm.estimate(llm.missing(prompts))
    print(f"{name} / {args.rerank or 'sin reranker'} / {args.llm} / {args.split}: {len(items)} ítems, umbral {threshold}, "
          f"{len(prompts)} llamadas al LLM ({len(items) - len(prompts)} sin LLM: ningún import pasó el umbral)")
    print(f"a pagar (fuera del caché): {to_pay['calls']} llamadas, {to_pay['input_tokens']} tokens de entrada + "
          f"~{to_pay['output_tokens_estimate']} de salida (estimada) = ~{to_pay['usd']:.4f} USD")
    if args.dry_run:
        return 0
    if to_pay["usd"] > args.max_usd:
        print(f"abortado: supera --max-usd {args.max_usd}")
        return 1

    start = time.time()
    records = []
    for it, prep in zip(items, preps):
        fix = finish(prep, llm(*prep.prompt) if prep.prompt else None)
        record = check_item(it, prep, fix, fetch)
        record |= {"code": fix.code, "report": fix.report, "edits": fix.edits,
                   "contexts": [{"module": c.module, "score": c.score, "passed": c.passed,
                                 "chunks": [{k: ch[k] for k in ("header", "package", "id", "n_tokens")}
                                            for ch in c.chunks]} for c in prep.contexts]}
        records.append(record)
        print(f"{'ok' if record['success'] else 'FALLA':<6} {it['id']}")
    seconds = round(time.time() - start, 1)

    report = {
        "collection": name, "reranker": args.rerank, "llm": llm.model.name, "split": args.split, "threshold": threshold,
        "budget": args.budget, "date": datetime.now().isoformat(timespec="seconds"), "config": asdict(cfg),
        "dataset": dataset_info(), "paid": {"calls": llm.misses, "input_tokens": llm.input_tokens,
                                            "output_tokens": llm.output_tokens, "usd": llm.usd},
        "cache_hits": llm.hits, "seconds": seconds, "metrics": summarize(records),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{name}__qA__{args.rerank + '__' if args.rerank else ''}{llm.model.name}__{args.split}"
    summary = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    (RESULTS_DIR / f"{stem}.json").write_text(summary, encoding="utf-8")
    with (RESULTS_DIR / f"{stem}.items.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    m = report["metrics"]
    print(f"listo en {seconds} s: {llm.misses} llamadas pagas ({llm.input_tokens} + {llm.output_tokens} tokens, "
          f"{llm.usd:.4f} USD), {llm.hits} del caché")
    print(f"positivos: éxito {m['positives']['success']}; negativos: éxito {m['negatives']['success']} "
          f"({m['negatives']['rejected_by_llm']}/{m['negatives']['passed_threshold']} rechazados por el LLM); "
          f"imports alucinados: {m['hallucinated_imports']}")
    print(f"-> {RESULTS_DIR / stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
