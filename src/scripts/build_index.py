"""Arma una colección de Chroma con los casos de uso de un splitter (paso 4, docs/agents/estado.md). USA LA API.

Uso:
    uv run python src/scripts/build_index.py --packages requests httpx pypdf --dry-run   # solo estima, sin API
    uv run python src/scripts/build_index.py --packages requests httpx pypdf             # piloto
    uv run python src/scripts/build_index.py --chunks s3-1024 --model e1 --cap 150       # corpus completo

Con un modelo local, `--warm-batch N` calcula los vectores de a N textos y los guarda en el caché después de cada lote
(`--pause` segundos entre lotes): si algo se corta, al relanzar solo se rehace el último lote. Es lo que usa
run_splitter_study.py.

Lee data/processed/<modo>/<chunks>/chunks.jsonl, aplica filtro + tope y embebe prefijo + chunk (R1). Antes de llamar
a la API cuenta los tokens de lo que no está en el caché y aborta si el costo supera --max-usd. Escribe la colección
en data/index/chroma/, los vectores en data/index/cache/ y el reporte (tokens, USD, tiempo) en
results/index/<colección>.json (versionado).
"""

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime

from dotenv import load_dotenv

from tprag.index.build import REPRESENTATIONS, IndexConfig, build, select_chunks, to_document
from tprag.index.cost import estimate
from tprag.index.embeddings import MODELS, cached
from tprag.split.select import DEFAULT_CAP, STRATEGIES
from tprag.split.store import ROOT

RESULTS_DIR = ROOT / "results" / "index"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="s3-1024", help="carpeta de data/processed/<modo>/")
    parser.add_argument("--console-examples", default="outputs_as_comments", help="carpeta de data/processed")
    parser.add_argument("--model", choices=sorted(MODELS), default="e1")
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="r1")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP, help="0 = sin tope")
    parser.add_argument("--strategy", choices=STRATEGIES, default="round_robin")
    parser.add_argument("--packages", nargs="*", help="solo estas librerías")
    parser.add_argument("--max-usd", type=float, default=0.50, help="aborta si lo que hay que pagar lo supera")
    parser.add_argument("--warm-batch", type=int, default=0,
                        help="solo modelos locales: calcula y guarda los vectores de a N textos (0 = todo junto)")
    parser.add_argument("--pause", type=float, default=0, help="segundos de pausa entre lotes de --warm-batch")
    parser.add_argument("--dry-run", action="store_true", help="solo cuenta tokens y estima el costo")
    args = parser.parse_args()

    cfg = IndexConfig(chunks=args.chunks, console_mode=args.console_examples, model=args.model,
                      representation=args.representation, cap=args.cap or None, strategy=args.strategy,
                      packages=args.packages)
    model = MODELS[cfg.model]
    name = cfg.collection_name()
    docs = [to_document(c, cfg.representation) for c in select_chunks(cfg)]
    texts = [d.page_content for d in docs]
    embeddings = cached(cfg.model)
    total, to_pay = estimate(texts, model), estimate(embeddings.missing(texts), model)
    print(f"{name}: {total['texts']} casos de uso, {total['tokens']} tokens ({total['usd']:.4f} USD sin caché)")
    print(f"a pagar (fuera del caché): {to_pay['texts']} textos, {to_pay['tokens']} tokens, {to_pay['usd']:.4f} USD "
          f"({model.name}, {model.usd_per_m_tokens} USD/M)")
    if args.dry_run:
        return 0
    if to_pay["usd"] > args.max_usd:
        print(f"abortado: supera --max-usd {args.max_usd}")
        return 1

    load_dotenv(ROOT / ".env")
    start = time.time()
    warmed = bool(args.warm_batch) and model.provider == "local"
    if warmed:
        todo = embeddings.missing(texts)
        for i in range(0, len(todo), args.warm_batch):
            embeddings.embed_documents(todo[i:i + args.warm_batch])
            done = min(i + args.warm_batch, len(todo))
            print(f"  vectores {done}/{len(todo)} ({round(time.time() - start)} s)", flush=True)
            if args.pause and done < len(todo):
                time.sleep(args.pause)
    build(name, docs, embeddings)  # con el caché ya lleno, solo arma la colección
    seconds = round(time.time() - start, 1)
    computed = embeddings.misses
    cache_hits = len(texts) - computed if warmed else embeddings.hits

    packages = sorted({d.metadata["package"] for d in docs})
    report = {
        "collection": name, "date": datetime.now().isoformat(timespec="seconds"), "config": asdict(cfg),
        "model": model.name, "api_model": model.api_name, "usd_per_m_tokens": model.usd_per_m_tokens,
        "model_config": asdict(model),
        "packages": len(packages),
        "package_list": packages if cfg.packages else None, "use_cases": len(docs), "tokens_total": total["tokens"],
        "paid": {**to_pay, "api_calls_texts": computed}, "cache_hits": cache_hits,
        "seconds": seconds,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"listo en {seconds} s: {computed} textos calculados (API o modelo local), {cache_hits} del caché -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
