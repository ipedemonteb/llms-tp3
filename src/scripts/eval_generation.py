"""Evalúa el generador con el dataset de data/eval/ (paso 8, docs/agents/generador.md §2.6). USA LA API (el modelo de
chat, y embeddings para las consultas que no están en el caché).

Uso:
    uv run python src/scripts/eval_generation.py --dry-run          # solo estima, sin API
    uv run python src/scripts/eval_generation.py                    # s3-1024, gpt-4o-mini, dev
    uv run python src/scripts/eval_generation.py --split test       # test: solo al final
    uv run --group local python src/scripts/eval_generation.py --model e4 --rerank k1-rr30 --dry-run   # con reranker

La colección se elige con los mismos flags que build_index.py. El umbral sale de la corrida de eval_retrieval.py en dev
con esa colección (o --threshold). Las respuestas del LLM quedan en caché (data/generate/): repetir una corrida no
vuelve a pagar. Escribe en results/eval/gen/ el resumen y el detalle por ítem (código corregido, reporte, reemplazos y
chequeos). Si esas salidas ya existen no las pisa (tprag.eval.outputs): --tag <nombre> escribe la variante al lado
(<corrida>__<nombre>) y --force la reemplaza.

--retry-format edits|script: la iteración (iteracion_ej4.md §3.2), apagada por defecto. Si la corrección tiene problemas
que se ven sin el dataset (reemplazo que no se aplica, código que no parsea, nombres nuevos sin definir), un reintento con
esos problemas, pidiendo reemplazos (edits, la corrida `__iter`) o el script completo (script, `__iter2`); si siguen, se
abstiene. La primera llamada es la misma que sin la opción (sale del caché); el dry-run cuenta los reintentos con las
primeras respuestas del caché. Exige --tag, para no escribir nunca con el nombre de la corrida sin iteración.

--first-format script: la variante uniforme (`__iter3`): el primer intento también pide el script completo (sin
iteración, el primer intento es siempre "edits", la lista de reemplazos). Exige --tag. Como esas primeras respuestas
no están en el caché, el dry-run estima los reintentos en el peor caso (que se reintenten todas).

--check-context: sobre la variante uniforme (`__iter4`), también va al reintento un script que usa nombres de la API
nueva que no aparecen en los fragmentos (la heurística de Faithfulness). Exige --first-format script y --tag.

--first-prompt v2: sobre la variante uniforme, sin --check-context (`__iter5`), el primer prompt v2 (prompt.PROMPTS):
antes de elegir, el modelo lista todos los usos del import y qué cubre cada paquete de los fragmentos. Las respuestas
usan los esquemas `*V2`, así que ninguna sale del caché de las otras corridas. Exige --tag. Sin la opción: "v1", el de
siempre. --first-prompt v3 (`__iter6`): v2 con otra decisión (reemplaza si el paquete cubre el uso principal, no todos).
"""

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime

from dotenv import load_dotenv

from tprag.eval.generation import check_item, summarize
from tprag.eval.outputs import refuse_overwrite, tag_arg, tagged
from tprag.eval.query import QUERIES
from tprag.eval.retrieval import dataset_info, select_items
from tprag.eval.verify import load_items
from tprag.generate.context import dev_threshold, load_reranker, rerank_threshold
from tprag.generate.llm import CHAT_MODELS, CachedLLM, TruncatedOutput
from tprag.generate.pipeline import answer, finish, first_prompt, prepare, retry_prompt
from tprag.split.tokens import token_counter
from tprag.generate.schema import ScriptOutput, ScriptOutputV2
from tprag.index.build import REPRESENTATIONS, IndexConfig, open_collection
from tprag.index.cost import estimate
from tprag.index.embeddings import MODELS, cached
from tprag.split.select import DEFAULT_CAP, STRATEGIES
from tprag.split.store import ROOT
from tprag.validate.pypi import CachedPyPI
from tprag.validate.validate import validate

RESULTS_DIR = ROOT / "results" / "eval" / "gen"


def retry_prompts(preps: list, llm: CachedLLM, fmt: str, first: str = "edits", context_check: bool = False,
                  version: str = "v1") -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Prompts de los reintentos que haría --retry-format `fmt`, armados con las primeras respuestas del caché (sin
    API), y los primeros prompts que faltan en el caché (sus reintentos no se pueden saber antes de pagarlos)."""
    prompts, unknown = [], []
    for prep in preps:
        if (prompt := first_prompt(prep, first, version)) is None:
            continue
        try:
            output = llm.peek(*prompt)
        except TruncatedOutput:
            continue  # respuesta cortada: el script queda igual, no hay nada que reintentar
        if output is None:
            unknown.append(prompt)
        elif retry := retry_prompt(prep, finish(prep, output), fmt, first == "script", context_check, version):
            prompts.append(retry)
    return prompts, unknown


RETRY_EXTRA_TOKENS = 850  # peor caso de un reintento sin la primera respuesta: el script o los reemplazos anteriores
#                           (~700, la salida estimada) + problemas y líneas del import (~150), sobre el mismo mensaje


def worst_case_retries(prompts: list[tuple[str, str]], llm: CachedLLM) -> dict:
    """Costo de reintentar todas estas llamadas cuando las primeras respuestas todavía no existen (no se puede saber
    cuáles tendrían problemas): el mismo mensaje + `RETRY_EXTRA_TOKENS` de entrada y la salida estimada de `llm`."""
    count = token_counter(llm.model.tokenizer)
    input_tokens = sum(count(system) + count(user) + RETRY_EXTRA_TOKENS for system, user in prompts)
    output_tokens = llm.estimate(prompts)["output_tokens_estimate"]
    return {"calls": len(prompts), "input_tokens": input_tokens, "output_tokens_estimate": output_tokens,
            "usd": llm.model.usd(input_tokens, output_tokens)}


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
    parser.add_argument("--retry-format", choices=("edits", "script"),
                        help="iteración: un reintento con los problemas de la corrección (pidiendo reemplazos o el script "
                             "completo) y, si siguen, abstención; exige --tag")
    parser.add_argument("--first-format", choices=("edits", "script"), default="edits",
                        help="formato del primer intento: edits (el de siempre) o script (variante uniforme; exige "
                             "--tag y --retry-format script)")
    parser.add_argument("--check-context", action="store_true",
                        help="sobre la variante uniforme: reintento también si usa nombres de la API nueva que no "
                             "aparecen en los fragmentos; exige --first-format script y --tag")
    parser.add_argument("--first-prompt", choices=("v1", "v2", "v3"), default="v1",
                        help="primer prompt: v1 (el de siempre) o v2/v3 (lista los usos y los paquetes candidatos antes "
                             "de elegir; exige --tag y la variante uniforme, sin --check-context)")
    parser.add_argument("--tag", type=tag_arg, help="sufijo __<tag> en las salidas: una variante al lado de la corrida")
    parser.add_argument("--force", action="store_true", help="reemplaza las salidas si ya existen")
    args = parser.parse_args()
    if (args.retry_format or args.first_format != "edits" or args.first_prompt != "v1") and not args.tag:
        parser.error("--retry-format, --first-format script y --first-prompt v2/v3 exigen --tag: la iteración no escribe "
                     "con el nombre de la corrida sin iteración")
    if args.first_prompt != "v1" and (args.first_format != "script" or args.check_context):
        parser.error("--first-prompt v2/v3 va sobre la variante uniforme sin --check-context: --first-format script "
                     "--retry-format script")
    if args.check_context and args.first_format != "script":
        parser.error("--check-context va sobre la variante uniforme: --first-format script --retry-format script")
    if args.first_format == "script" and args.retry_format != "script":
        parser.error("--first-format script va con --retry-format script (variante uniforme)")

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
    retry_llm = llm
    if args.retry_format == "script":  # el mismo caché; la clave incluye el esquema, así que no se mezcla con "edits"
        retry_llm = CachedLLM(CHAT_MODELS[args.llm])
        retry_llm.output_schema = ScriptOutputV2 if args.first_prompt != "v1" else ScriptOutput
    if args.first_format == "script":  # variante uniforme: el primer intento también devuelve `ScriptOutput`
        llm = retry_llm
    # un resultado ya corrido no se pisa sin querer (tprag.eval.outputs): se chequea antes de llamar al LLM
    stem = tagged(f"{name}__qA__{args.rerank + '__' if args.rerank else ''}{llm.model.name}__{args.split}", args.tag)
    if refusal := refuse_overwrite([RESULTS_DIR / f"{stem}.json", RESULTS_DIR / f"{stem}.items.jsonl"], args.force):
        print(refusal)
        if not args.dry_run:
            return 1
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
    prompts = [p for prep in preps if (p := first_prompt(prep, args.first_format, args.first_prompt))]
    to_pay = llm.estimate(llm.missing(prompts))
    print(f"{name} / {args.rerank or 'sin reranker'} / {args.llm} / {args.split}: {len(items)} ítems, umbral {threshold}, "
          f"{len(prompts)} llamadas al LLM ({len(items) - len(prompts)} sin LLM: ningún import pasó el umbral)")
    print(f"a pagar (fuera del caché): {to_pay['calls']} llamadas, {to_pay['input_tokens']} tokens de entrada + "
          f"~{to_pay['output_tokens_estimate']} de salida (estimada) = ~{to_pay['usd']:.4f} USD")
    if args.retry_format:
        retries, unknown = retry_prompts(preps, llm, args.retry_format, args.first_format, args.check_context,
                                         args.first_prompt)
        retry_pay = retry_llm.estimate(retry_llm.missing(retries))
        print(f"reintentos: {len(retries)} con problemas en la primera respuesta, {retry_pay['calls']} fuera del caché "
              f"= ~{retry_pay['usd']:.4f} USD")
        to_pay["usd"] += retry_pay["usd"]
        if unknown:  # primeras respuestas que todavía no existen: no se sabe cuáles se reintentarían
            worst = worst_case_retries(unknown, retry_llm)
            print(f"  + peor caso de las {worst['calls']} primeras respuestas fuera del caché (que se reintenten todas): "
                  f"{worst['input_tokens']} tokens de entrada + ~{worst['output_tokens_estimate']} de salida = "
                  f"~{worst['usd']:.4f} USD")
            to_pay["usd"] += worst["usd"]
    if args.dry_run:
        return 0
    if to_pay["usd"] > args.max_usd:
        print(f"abortado: supera --max-usd {args.max_usd}")
        return 1

    start = time.time()
    records = []
    for it, prep in zip(items, preps):
        fix = answer(prep, llm, retry=args.retry_format, retry_llm=retry_llm, first=args.first_format,
                     context_check=args.check_context, version=args.first_prompt)
        record = check_item(it, prep, fix, fetch)
        record |= {"code": fix.code, "report": fix.report, "edits": fix.edits, "llm_raw": fix.llm_raw, "retry": fix.retry,
                   "contexts": [{"module": c.module, "score": c.score, "passed": c.passed,
                                 "chunks": [{k: ch[k] for k in ("header", "package", "id", "n_tokens", "text")}
                                            for ch in c.chunks]} for c in prep.contexts]}
        if fix.changed_lines is not None:  # variante uniforme
            record["changed_lines"] = fix.changed_lines
        records.append(record)
        print(f"{'ok' if record['success'] else 'FALLA':<6} {it['id']}"
              f"{'  (respuesta del LLM cortada por el tope de salida)' if fix.llm_error else ''}")
    seconds = round(time.time() - start, 1)
    llms = [llm] + ([retry_llm] if retry_llm is not llm else [])  # con --retry-format script, dos cachés (esquemas)
    paid = {"calls": sum(x.misses for x in llms), "input_tokens": sum(x.input_tokens for x in llms),
            "output_tokens": sum(x.output_tokens for x in llms), "usd": round(sum(x.usd for x in llms), 6)}

    report = {
        "collection": name, "reranker": args.rerank, "llm": llm.model.name, "split": args.split, "threshold": threshold,
        "budget": args.budget, "date": datetime.now().isoformat(timespec="seconds"), "config": asdict(cfg),
        "dataset": dataset_info(), "paid": paid,
        "tag": args.tag, "cache_hits": sum(x.hits for x in llms), "seconds": seconds, "metrics": summarize(records),
        "first_format": args.first_format, "check_context": args.check_context, "first_prompt": args.first_prompt,
        "retry": {"format": args.retry_format, "retried": sum(r["retry"] is not None for r in records),
                  "abstained": sum(bool(r["retry"] and r["retry"]["remaining"]) for r in records)},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    (RESULTS_DIR / f"{stem}.json").write_text(summary, encoding="utf-8")
    with (RESULTS_DIR / f"{stem}.items.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    m = report["metrics"]
    print(f"listo en {seconds} s: {paid['calls']} llamadas pagas ({paid['input_tokens']} + {paid['output_tokens']} "
          f"tokens, {paid['usd']:.4f} USD), {report['cache_hits']} del caché")
    print(f"positivos: éxito {m['positives']['success']}; negativos: éxito {m['negatives']['success']} "
          f"({m['negatives']['rejected_by_llm']}/{m['negatives']['passed_threshold']} rechazados por el LLM); "
          f"imports alucinados: {m['hallucinated_imports']}; respuestas cortadas por el tope: {m['llm_errors']}")
    print(f"-> {RESULTS_DIR / stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
