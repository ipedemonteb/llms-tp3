"""Estudio del reranker (docs/agents/plan_reranker.md): puntúa con cada reranker los 50 chunks que el paso 7 recuperó
para cada target de la base (S3 · 1024 · E4 · R1 · tope 150). No usa API ni Chroma: lee el `.targets.jsonl` de la base
y el texto de los chunks de data/processed/.

Uso (hace falta el grupo de dependencias `local`):
    uv run --group local python src/scripts/run_rerank_study.py --dry-run          # qué pares faltan puntuar
    uv run --group local python src/scripts/run_rerank_study.py --limit 5          # prueba de velocidad (5 targets)
    uv run --group local python src/scripts/run_rerank_study.py                    # K2 y K1 sobre todo dev
    uv run --group local python src/scripts/run_rerank_study.py --rerankers k2     # uno solo
    uv run --group analysis python src/scripts/analyze_rerank.py                   # variantes, tabla y gráficos
    uv run --group local python src/scripts/run_rerank_study.py --corpus full --rerankers k1   # corpus completo
    uv run --group local python src/scripts/run_rerank_study.py --corpus full --rerankers k2 --tag estudio
        # estudio completo sobre las 101 (plan_cierre.md §9): manifiesto propio, K1 ya está del paso 3

Los puntajes van a un caché (data/index/cache/rerank.sqlite): la prueba con --limit no se pierde y una corrida cortada
se retoma. Con --limit solo se mide la velocidad (no escribe puntajes para el análisis).

Salidas: results/eval/rerank/<colección>__qA__<reranker>__<split>.scores.jsonl (un registro por target, con los 50
puntajes en el orden denso) y el manifiesto results/study/rerank_manifest.json (tiempos, memoria, truncado, entorno). Con `--corpus full` (las 101
librerías, plan_cierre.md §3) el manifiesto es rerank_full_<split>_manifest.json y no pisa el del estudio.
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

from analyze_splitters import EVAL_RESULTS, read_jsonl
from run_splitter_study import STUDY_DIR, environment, git_dirty, load_manifest, save_manifest
from tprag.eval.rerank_study import BASE_VARIANT, POOL, scores_path, write_jsonl_once
from tprag.eval.retrieval import dataset_info
from tprag.eval.verify import EVAL_DIR
from tprag.index.rerank import RERANKERS, cached_reranker
from tprag.split.store import PROCESSED_DIR, load_chunks

CONSOLE_MODE = "outputs_as_comments"
MANIFEST = STUDY_DIR / "rerank_manifest.json"
TEMPLATE_TOKENS = 64  # margen por par para el formato del modelo (instrucción y plantilla de Qwen, separadores de BERT)


def load_pairs(records: list[dict], texts: dict[str, str]) -> list[tuple[str, str]]:
    missing = {h["id"] for r in records for h in r["hits"][:POOL]} - set(texts)
    if missing:
        raise ValueError(f"{len(missing)} chunks recuperados no están en chunks.jsonl (¿se rehízo el split?)")
    return [(r["query"], texts[h["id"]]) for r in records for h in r["hits"][:POOL]]


def truncated(scorer, pairs: list[tuple[str, str]], max_length: int) -> int:
    """Pares que pasan `max_length` tokens (aproximado: consulta + chunk + margen de plantilla)."""
    queries, docs = list(dict.fromkeys(q for q, _ in pairs)), list(dict.fromkeys(d for _, d in pairs))
    nq = dict(zip(queries, scorer.n_tokens(queries)))
    nd = dict(zip(docs, scorer.n_tokens(docs)))
    return sum(nq[q] + nd[d] + TEMPLATE_TOKENS > max_length for q, d in pairs)


def corpus_packages(corpus: str, packages: list[str] | None) -> list[str] | None:
    """Librerías del índice: las pedidas, las del subconjunto del estudio o None (corpus completo, sin sufijo)."""
    if packages:
        return packages
    if corpus == "full":
        return None
    return json.loads((EVAL_DIR / "corpus_subset.json").read_text(encoding="utf-8"))["packages"]


def manifest_path(corpus: str, split: str, tag: str | None = None):
    if corpus == "subset" and not tag:
        return MANIFEST
    return STUDY_DIR / "_".join(["rerank", corpus, split] + ([tag] if tag else []) + ["manifest.json"])


def start_manifest(corpus: str, split: str, tag: str | None) -> dict:
    """Manifiesto de la corrida. Con `tag` es un archivo propio que, la primera vez, arranca como copia del manifiesto
    sin tag (así los tiempos de un reranker que ya se midió no se pierden), sin tocar el original."""
    path = manifest_path(corpus, split, tag)
    if tag and not path.exists():
        return load_manifest(manifest_path(corpus, split))
    return load_manifest(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerankers", nargs="*", choices=sorted(RERANKERS), default=["k2", "k1"],
                        help="en este orden (default: primero el rápido)")
    parser.add_argument("--corpus", choices=("subset", "full"), default="subset",
                        help="subset: las 30 de corpus_subset.json (estudio); full: las 101 librerías")
    parser.add_argument("--packages", nargs="*", help="pisa --corpus: solo estas librerías")
    parser.add_argument("--tag", help="manifiesto propio (rerank_<corpus>_<split>_<tag>_manifest.json), sin pisar "
                                      "el de corridas anteriores")
    parser.add_argument("--split", choices=("dev", "test"), default="dev", help="test solo para el reporte final")
    parser.add_argument("--limit", type=int, help="solo los primeros N targets: prueba de velocidad, sin salidas")
    # 1 es lo más rápido en el M3 (float16, 32 pares): K2 0,21 s/par con 1 y 0,29 con 8; K1 0,84 con 1 y 1,20 con 8.
    # Con lotes los pares se rellenan hasta el más largo (mediana 355 tokens, máximo 3.067) y la GPU compartida no rinde
    parser.add_argument("--batch-size", type=int, default=1, help="pares por lote del modelo")
    parser.add_argument("--dry-run", action="store_true", help="solo cuenta los pares que faltan (no carga modelos)")
    args = parser.parse_args()

    packages = corpus_packages(args.corpus, args.packages)
    collection = BASE_VARIANT.index_config(packages).collection_name()
    base = EVAL_RESULTS / f"{collection}__qA__{args.split}.targets.jsonl"
    if not base.exists():
        print(f"falta {base.name}: correr eval_retrieval.py sobre la base (o el estudio de hiperparámetros)")
        return 1
    records = read_jsonl(base)[:args.limit] if args.limit else read_jsonl(base)
    chunks = load_chunks(PROCESSED_DIR / CONSOLE_MODE / BASE_VARIANT.chunks / "chunks.jsonl")
    texts = {c.id: c.embed_text(BASE_VARIANT.representation) for c in chunks}
    pairs = load_pairs(records, texts)
    print(f"reranker sobre {collection} / {args.split}: {len(records)} targets × {POOL} = {len(pairs)} pares"
          + (f" (--limit {args.limit}: solo velocidad)" if args.limit else ""))

    scorers = {r: cached_reranker(r, batch_size=args.batch_size) for r in args.rerankers}
    for r, sc in scorers.items():
        print(f"  {r} {RERANKERS[r].name}: faltan {len(sc.missing(pairs))} pares distintos")
    if args.dry_run:
        return 0

    if sys.platform == "darwin" and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])  # sin dormir mientras corre este proceso
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    manifest_file = manifest_path(args.corpus, args.split, args.tag)
    manifest = start_manifest(args.corpus, args.split, args.tag)
    manifest.update({"study": "rerank", "split": args.split, "collection": collection, "pool": POOL,
                     "dataset": dataset_info(), "git_dirty": git_dirty(), "environment": environment()})
    manifest.setdefault("rerankers", {})
    for r, sc in scorers.items():
        model = RERANKERS[r]
        print(f"[{r}] {model.name} ...", flush=True)
        last = [time.time()]

        def progress(done, total, last=last):
            if time.time() - last[0] > 30 or done == total:
                print(f"  {done}/{total} pares", flush=True)
                last[0] = time.time()

        load_seconds = None
        if sc.missing(pairs):  # la carga del modelo se mide aparte: no es costo por consulta
            start = time.time()
            sc.inner._load()
            load_seconds = round(time.time() - start, 1)
        start = time.time()
        scores = sc.score(pairs, progress)
        seconds = round(time.time() - start, 1)
        n_trunc = truncated(sc.inner, pairs, model.max_length)
        per_pair = seconds / sc.misses if sc.misses else None
        print(f"  listo en {seconds} s: {sc.misses} pares calculados, {sc.hits} del caché"
              + (f" ({per_pair:.3f} s/par; dev completo ≈ {per_pair * len(read_jsonl(base)) * POOL / 60:.0f} min)"
                 if per_pair else "") + f"; truncados {n_trunc}/{len(pairs)}", flush=True)

        entry = manifest["rerankers"].get(r, {})
        if args.limit:
            entry["smoke"] = {"targets": len(records), "pairs": len(pairs), "computed": sc.misses, "seconds": seconds,
                              "load_seconds": load_seconds, "seconds_per_pair": per_pair,
                              "finished": datetime.now().isoformat(timespec="seconds")}
        else:
            if sc.misses:  # con todo en el caché el tiempo no dice nada: se conserva el de la corrida que calculó
                entry["timing"] = {"computed": sc.misses, "seconds": seconds, "load_seconds": load_seconds,
                                   "seconds_per_pair": per_pair, "batch_size": args.batch_size,
                                   "dtype": model.dtype, "peak_mps_mb": round(sc.inner.peak_mb)}
            entry.update({"pairs": len(pairs), "truncated_pairs": n_trunc,
                          "truncated_rate": round(n_trunc / len(pairs), 4),
                          "finished": datetime.now().isoformat(timespec="seconds")})
            out = scores_path(collection, r, args.split)
            rows, i = [], 0
            for rec in records:
                n = len(rec["hits"][:POOL])
                rows.append({"id": rec["id"], "module": rec["module"],
                             "chunk_ids": [h["id"] for h in rec["hits"][:POOL]], "scores": scores[i:i + n]})
                i += n
            # no pisa: si ya estaba (p. ej. K1 del paso 3) tiene que coincidir, si no falla y lo deja igual
            print(f"  -> {out} ({write_jsonl_once(out, rows)})")
        entry.update({"name": model.name, "label": model.label, "cache_key": model.cache_key,
                      "instruction": model.instruction, "max_length": model.max_length})
        manifest["rerankers"][r] = entry
        save_manifest(manifest, manifest_file)
        sc._inner = None  # libera el modelo antes del siguiente
        gc.collect()
    if not args.limit:
        print(f"siguiente: uv run --group analysis python src/scripts/analyze_rerank.py --corpus {args.corpus} "
              f"--split {args.split}" + (f" --tag {args.tag}" if args.tag else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
