"""Answer Relevance (a) y (b) sobre una corrida del generador (docs/agents/metricas.md §3.3). **USA API** salvo con
--dry-run o --export.

Uso:
    uv run python src/scripts/judge_generation.py <corrida> --dry-run            # cuenta tokens y estima el costo
    uv run python src/scripts/judge_generation.py <corrida> --export prompts.jsonl  # prompts para validar con Claude
    uv run python src/scripts/judge_generation.py <corrida> --max-usd 0.05 --confirm  # juzga (paga lo que no está en caché)

<corrida>: nombre en results/eval/gen/ (sin extensión) o su ruta. Solo se juzgan los scripts que cambiaron.
--ids a,b,c o --ids-file ids.txt: solo esos ítems (p. ej. los ~20 de la validación, metricas.md §3.8).

Protecciones: sin --confirm no llama a la API aunque el costo esté bajo --max-usd; con --confirm, aborta si lo que
falta pagar supera --max-usd (0,05 por defecto). Las respuestas quedan en data/generate/cache/judge.sqlite y los
embeddings de (a) en el caché de embeddings: juzgar de nuevo no vuelve a pagar.

Escribe results/eval/gen/<corrida>.judge.jsonl (una línea por script) y muestra el resumen.
"""

import argparse
import json
from pathlib import Path
from statistics import mean

from tprag.eval.judge import (JUDGES, PRESERVE_SCORE, JudgeLLM, describe_prompt, judged_records, preserve_prompt,
                              relevance_a)
from tprag.eval.retrieval import dataset_info
from tprag.generate.llm import CHAT_MODELS
from tprag.index.embeddings import MODELS, cached
from tprag.split.tokens import token_counter

from analyze_generation import load_items, run_paths  # mismo directorio de scripts

DESCRIPTION_TOKENS = 30  # por descripción, para estimar los embeddings de (a)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--judge", choices=("a", "b", "ab"), default="ab")
    parser.add_argument("--llm", choices=sorted(CHAT_MODELS), default="gpt-4o-mini")
    parser.add_argument("--ids", help="ids separados por coma")
    parser.add_argument("--ids-file", type=Path, help="un id por línea")
    parser.add_argument("--dry-run", action="store_true", help="solo cuenta tokens y estima el costo")
    parser.add_argument("--export", type=Path, help="escribe los prompts (JSONL) para validar con otro juez; no paga")
    parser.add_argument("--max-usd", type=float, default=0.05, help="aborta si lo que hay que pagar lo supera")
    parser.add_argument("--confirm", action="store_true", help="necesario para llamar a la API")
    args = parser.parse_args()

    summary_path, items_path = run_paths(args.run)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    current = dataset_info()["sha256"]
    if any(current.get(k) != v for k, v in summary.get("dataset", {}).get("sha256", {}).items()):
        print("el dataset cambió desde la corrida: correrla de nuevo antes de juzgarla")
        return 1
    items = load_items()
    originals = {i: it["code"] for i, it in items.items()}
    ids = None
    if args.ids or args.ids_file:
        ids = set(args.ids.split(",")) if args.ids else set(args.ids_file.read_text().split())
    selected = judged_records(records, originals, ids)
    judges = [JUDGES[j] for j in args.judge]
    print(f"{len(selected)} scripts cambiados a juzgar ({', '.join(j.name for j in judges)})")

    prompts = {}
    for r in selected:
        modules = [t["module"] for t in r["targets"]]
        prompts[r["id"]] = {"a": describe_prompt(r["code"]), "b": preserve_prompt(originals[r["id"]], r["code"], modules)}

    if args.export:
        with args.export.open("w", encoding="utf-8") as f:
            for i, p in prompts.items():
                for j in judges:
                    f.write(json.dumps({"id": i, "judge": j.name, "schema": j.schema.model_json_schema(),
                                        "system": p[j.name][0], "user": p[j.name][1]}, ensure_ascii=False) + "\n")
        print(f"escrito {args.export} (no se llamó a la API)")
        return 0

    model = CHAT_MODELS[args.llm]
    llms = {j.name: JudgeLLM(model, j) for j in judges}
    total = 0.0
    for j in judges:
        todo = llms[j.name].missing([p[j.name] for p in prompts.values()])
        est = llms[j.name].estimate(todo)
        total += est["usd"]
        print(f"juez ({j.name}): {len(prompts) - len(todo)} en caché, {est['calls']} a pagar, "
              f"{est['input_tokens']} tokens de entrada + ~{est['output_tokens_estimate']} de salida → "
              f"~{est['usd']:.4f} USD")
    if "a" in args.judge:
        e1 = MODELS["e1"]
        count = token_counter(e1.tokenizer)
        emb_tokens = sum(count(items[i]["task"]) + 3 * DESCRIPTION_TOKENS for i in prompts)
        emb_usd = emb_tokens * e1.usd_per_m_tokens / 1e6
        total += emb_usd
        print(f"embeddings de (a) con {e1.name}: hasta ~{emb_tokens} tokens → ~{emb_usd:.4f} USD (menos si están en caché)")
    print(f"total estimado: ~{total:.4f} USD")

    if args.dry_run:
        return 0
    if total > args.max_usd:
        print(f"abortado: supera --max-usd {args.max_usd}")
        return 1
    if not args.confirm:
        print("no se llamó a la API: agregar --confirm para juzgar")
        return 1

    embed = cached("e1").embed_documents
    out = []
    for r in selected:
        row = {"id": r["id"], "subset": r["subset"], "success": r["success"]}
        if "b" in llms:
            v = llms["b"](*prompts[r["id"]]["b"])
            row["b"] = {**v.model_dump(), "score": PRESERVE_SCORE[v.preserves]}
        if "a" in llms:
            d = llms["a"](*prompts[r["id"]]["a"])
            row["a"] = {"descriptions": d.descriptions,
                        "score": relevance_a(d.descriptions, items[r["id"]]["task"], embed)}
        out.append(row)
        print(f"{r['id']:<40} " + " ".join(f"{j}={row[j]['score']:.2f}" for j in ("a", "b") if j in row))

    path = items_path.with_name(items_path.name.removesuffix(".items.jsonl") + ".judge.jsonl")
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in out), encoding="utf-8")
    for j in ("a", "b"):
        scores = [row[j]["score"] for row in out if j in row]
        if scores:
            ok = [row[j]["score"] for row in out if j in row and row["success"]]
            bad = [row[j]["score"] for row in out if j in row and not row["success"]]
            print(f"({j}) media {mean(scores):.3f} | éxito del nivel 1: {mean(ok) if ok else float('nan'):.3f} "
                  f"| sin éxito: {mean(bad) if bad else float('nan'):.3f}")
    paid = sum(llm.usd for llm in llms.values())
    print(f"pagado (chat): {paid:.4f} USD; escrito {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
