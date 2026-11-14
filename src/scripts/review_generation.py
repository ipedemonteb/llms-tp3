"""Revisión manual de la generación (docs/agents/metricas.md §3.8): valida el juez de Answer Relevance (b) y la
heurística de Faithfulness contra una persona (y el juez contra Claude Opus 5.5). No usa API.

Uso:
    uv run python src/scripts/review_generation.py sheet       # elige la muestra y escribe la planilla a completar
    uv run python src/scripts/review_generation.py opus        # exporta los prompts de (b) para Claude Opus 5.5
    uv run python src/scripts/review_generation.py opus-check  # valida y une las respuestas de Opus

`sheet`: muestra estratificada de 20 scripts cambiados de las dos corridas permisivas de dev (10 de la base y 10 del
reranker), con cupos por (éxito del nivel 1 × veredicto del juez) y sin repetir un ítem entre corridas; semilla fija.
Escribe results/study/review/sample.json y planilla.md. La planilla es a ciegas: no muestra el veredicto del juez ni el
éxito del nivel 1. No pisa una planilla existente (puede estar completada) salvo con --force.

`opus`: los mismos prompts de (b) que recibió gpt-4o-mini (instrucciones + mensaje + esquema JSON), en tandas, en
results/study/review/opus/prompts_<k>.jsonl. Cada tanda la responde un agente de Claude Opus 5.5 lanzado desde Claude
Code (0 USD del presupuesto de OpenAI) en opus/answers_<k>.jsonl. `opus-check` valida cada respuesta con el mismo
esquema (`Preservation`) y las une en opus_answers.jsonl.
"""

import argparse
import difflib
import json
import random
import re
from pathlib import Path

from tprag.eval.faithfulness import claims, mentions, record_context
from tprag.eval.judge import PRESERVE_SYSTEM, Preservation, preserve_prompt
from tprag.split.store import ROOT

from analyze_generation import load_items  # mismo directorio de scripts

GEN_DIR, REVIEW_DIR = ROOT / "results" / "eval" / "gen", ROOT / "results" / "study" / "review"
RUNS = {"base": "s3-1024__e4__r1__cap150__qA__gpt-4o-mini__dev",
        "rerank": "s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4o-mini__dev"}
SEED = 2026
# Cupos por (éxito del nivel 1, veredicto del juez): proporcionales al tamaño de cada grupo, al menos 1 por grupo
# (decidido 2026-11-14 sobre los juicios de las dos corridas)
QUOTAS = {
    "base": {(True, "si"): 2, (True, "parcial"): 3, (True, "no"): 1, (False, "parcial"): 3, (False, "no"): 1},
    "rerank": {(True, "si"): 2, (True, "parcial"): 3, (True, "no"): 1, (False, "parcial"): 2, (False, "no"): 1,
               (False, "si"): 1},
}
VERDICTS = ("si", "parcial", "no")
OPUS_DIR, OPUS_MODEL, OPUS_BATCH = REVIEW_DIR / "opus", "claude-opus-5-5", 5


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sample(judged: dict[str, list[dict]], quotas: dict = QUOTAS, seed: int = SEED) -> list[dict]:
    """judged: corrida → filas de su `.judge.jsonl`. Devuelve [{run, id, group}] en orden al azar."""
    rng, taken, out = random.Random(seed), set(), []
    for run, cells in quotas.items():
        for (success, verdict), n in cells.items():
            pool = sorted(r["id"] for r in judged[run] if r["success"] == success
                          and r.get("b", {}).get("preserves") == verdict and r["id"] not in taken)
            if len(pool) < n:
                raise ValueError(f"{run}: hay {len(pool)} scripts con (éxito={success}, {verdict}) y el cupo es {n}")
            for item in rng.sample(pool, n):
                taken.add(item)
                out.append({"run": run, "id": item, "group": {"success": success, "judge_b": verdict}})
    rng.shuffle(out)
    return [s | {"review_id": f"R{i:02d}"} for i, s in enumerate(out, 1)]


def claim_rows(original: str, fixed: str, context: str) -> list[dict]:
    """Un identificador por fila (el primero de cada nombre), con lo que dice la heurística y dónde aparece."""
    seen, rows = set(), []
    for c in claims(original, fixed, context):
        if c.name in seen:
            continue
        seen.add(c.name)
        where = next((line.strip() for line in context.splitlines() if mentions(line, c.name)), None)
        rows.append({"name": c.name, "line": c.line, "kind": c.kind, "in_context": mentions(context, c.name),
                     "where": where[:140] if where else None})
    return rows


def fence(code: str, lang: str) -> str:
    return f"````{lang}\n{code.rstrip()}\n````"


def item_block(s: dict, item: dict, record: dict) -> str:
    original, fixed = item["code"], record["code"]
    diff = "".join(difflib.unified_diff(original.splitlines(keepends=True), fixed.splitlines(keepends=True),
                                        "original", "modificado", n=2))
    rows = claim_rows(original, fixed, record_context(record))
    lines = [
        f"## {s['review_id']}",
        "",
        f"- **Tarea del script:** {item['task']}",
        f"- **Import reemplazado:** {', '.join(f'`{t['module']}`' for t in record['targets'])}",
        "",
        "### Cambios",
        "",
        fence(diff, "diff"),
        "",
        "<details><summary>Script original completo</summary>",
        "",
        fence(original, "python"),
        "",
        "</details>",
        "",
        "<details><summary>Script modificado completo</summary>",
        "",
        fence(fixed, "python"),
        "",
        "</details>",
        "",
        "### Preservación de la tarea",
        "",
        "¿El script modificado hace la misma tarea que el original? `si` / `parcial` / `no`.",
        "",
        "- preserva:",
        "- que_se_perdio:",
        "",
        "### Faithfulness (heurística)",
        "",
    ]
    if rows:
        lines += [
            "Identificadores de la API nueva en las líneas cambiadas. La heurística dice que un identificador es fiel "
            "si aparece en los chunks que recibió el LLM. Marcá los números en los que **la heurística se equivoca** "
            "(p. ej. aparece en el contexto pero no como esa API, o no aparece y sí está).",
            "",
            "| # | Identificador | Línea | Tipo | ¿En el contexto? (heurística) | Dónde aparece |",
            "|---|---|---|---|---|---|",
        ] + [f"| {i} | `{r['name']}` | {r['line']} | {r['kind']} | {'sí' if r['in_context'] else 'no'} | "
             + (f"`{r['where'].replace('|', '¦').replace('`', "'")}`" if r["where"] else "—") + " |"
             for i, r in enumerate(rows, 1)] + [
            "",
            "- errores_heuristica:",
        ]
    else:
        lines += ["La heurística no encontró identificadores de una API nueva en este cambio (no hay nada que marcar).",
                  ""]
    return "\n".join(lines) + "\n"


def header(n: int) -> str:
    return "\n".join([
        "# Revisión manual de la generación",
        "",
        f"{n} scripts de dev corregidos por gpt-4o-mini (metricas.md §3.8). Para cada uno, completá después de los dos "
        "puntos (sin borrar la etiqueta):",
        "",
        "- `preserva:` `si`, `parcial` o `no`.",
        "- `que_se_perdio:` qué se perdió o cambió, con el número de línea; vacío si nada.",
        "- `errores_heuristica:` los números de la tabla en los que la heurística se equivoca, separados por coma, o "
        "`ninguno`.",
        "",
        "Las instrucciones que recibe el juez para la misma pregunta (usá el mismo criterio):",
        "",
        f"> {PRESERVE_SYSTEM}",
        "",
        "Los scripts están en orden al azar y la planilla no muestra qué corrida es cada uno ni qué dijo el juez.",
        "",
    ]) + "\n"


FIELD = re.compile(r"^- (preserva|que_se_perdio|errores_heuristica):[ \t]*(.*)$", re.M)


def parse_sheet(text: str) -> dict[str, dict]:
    """review_id → {"preserva": "si"|"parcial"|"no"|None, "que_se_perdio": str, "errores_heuristica": list[int]|None}.
    None: sin completar. Levanta ValueError si un valor no es válido (para corregirlo en la planilla)."""
    out = {}
    for block in re.split(r"^## (?=R\d+\s*$)", text, flags=re.M)[1:]:
        rid = block.split("\n", 1)[0].strip()
        fields = {k: v.strip() for k, v in FIELD.findall(block)}
        verdict = fields.get("preserva", "").strip("`").lower().replace("sí", "si") or None
        if verdict is not None and verdict not in VERDICTS:
            raise ValueError(f"{rid}: preserva = {fields['preserva']!r} (opciones: si, parcial, no)")
        errors = fields.get("errores_heuristica", "").strip("`").lower()
        if "errores_heuristica" not in fields:
            parsed = []  # el ítem no tiene identificadores para marcar
        elif not errors:
            parsed = None
        elif errors == "ninguno":
            parsed = []
        elif re.fullmatch(r"\d+(\s*,\s*\d+)*", errors):
            parsed = sorted({int(x) for x in errors.split(",")})
        else:
            raise ValueError(f"{rid}: errores_heuristica = {fields['errores_heuristica']!r} (números o 'ninguno')")
        out[rid] = {"preserva": verdict, "que_se_perdio": fields.get("que_se_perdio", ""), "errores_heuristica": parsed}
    return out


def cmd_sheet(force: bool) -> int:
    sheet_path, sample_path = REVIEW_DIR / "planilla.md", REVIEW_DIR / "sample.json"
    if sheet_path.exists() and not force:
        print(f"ya existe {sheet_path.relative_to(ROOT)} (puede estar completada): no la piso; usar --force")
        return 1
    judged = {run: read_jsonl(GEN_DIR / f"{name}.judge.jsonl") for run, name in RUNS.items()}
    records = {run: {r["id"]: r for r in read_jsonl(GEN_DIR / f"{name}.items.jsonl")} for run, name in RUNS.items()}
    items = load_items()
    chosen = sample(judged)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for s in chosen:
        s["claims"] = claim_rows(items[s["id"]]["code"], records[s["run"]][s["id"]]["code"],
                                 record_context(records[s["run"]][s["id"]]))
    sample_path.write_text(json.dumps({"seed": SEED, "runs": RUNS, "quotas": {r: {f"{k[0]}|{k[1]}": n for k, n in q.items()}
                                                                              for r, q in QUOTAS.items()},
                                       "items": chosen}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    sheet_path.write_text(header(len(chosen)) + "\n".join(item_block(s, items[s["id"]], records[s["run"]][s["id"]])
                                                          for s in chosen), encoding="utf-8")
    print(f"{len(chosen)} scripts → {sheet_path.relative_to(ROOT)} y {sample_path.relative_to(ROOT)}")
    return 0


def opus_prompts(chosen: list[dict], items: dict[str, dict], records: dict[str, dict[str, dict]]) -> list[dict]:
    """Los prompts de (b) tal como los arma judge_generation.py, con el esquema de la salida estructurada."""
    schema = Preservation.model_json_schema()
    out = []
    for s in chosen:
        r = records[s["run"]][s["id"]]
        system, user = preserve_prompt(items[s["id"]]["code"], r["code"], [t["module"] for t in r["targets"]])
        out.append({"review_id": s["review_id"], "system": system, "user": user, "schema": schema})
    return out


def check_answer(prompt_ids: set[str], row: dict) -> dict:
    """Valida una respuesta de Opus con el mismo esquema que el juez; devuelve la fila normalizada."""
    if row.get("review_id") not in prompt_ids:
        raise ValueError(f"review_id desconocido: {row.get('review_id')!r}")
    v = Preservation.model_validate(row["output"])
    return {"review_id": row["review_id"], "model": row.get("model", OPUS_MODEL), "output": v.model_dump()}


def cmd_opus(force: bool) -> int:
    sample_path = REVIEW_DIR / "sample.json"
    if not sample_path.exists():
        print("falta sample.json: correr primero `sheet`")
        return 1
    if OPUS_DIR.exists() and any(OPUS_DIR.glob("answers_*.jsonl")) and not force:
        print(f"ya hay respuestas en {OPUS_DIR.relative_to(ROOT)}: no piso los prompts; usar --force")
        return 1
    chosen = json.loads(sample_path.read_text(encoding="utf-8"))["items"]
    records = {run: {r["id"]: r for r in read_jsonl(GEN_DIR / f"{name}.items.jsonl")} for run, name in RUNS.items()}
    prompts = opus_prompts(chosen, load_items(), records)
    OPUS_DIR.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(prompts), OPUS_BATCH):
        path = OPUS_DIR / f"prompts_{k // OPUS_BATCH + 1}.jsonl"
        path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prompts[k:k + OPUS_BATCH]),
                        encoding="utf-8")
    print(f"{len(prompts)} prompts en {-(-len(prompts) // OPUS_BATCH)} tandas → {OPUS_DIR.relative_to(ROOT)}/")
    return 0


def cmd_opus_check() -> int:
    prompt_ids = {p["review_id"] for path in sorted(OPUS_DIR.glob("prompts_*.jsonl")) for p in read_jsonl(path)}
    answers = {}
    for path in sorted(OPUS_DIR.glob("answers_*.jsonl")):
        for row in read_jsonl(path):
            a = check_answer(prompt_ids, row)
            if a["review_id"] in answers:
                raise ValueError(f"{a['review_id']} respondido dos veces")
            answers[a["review_id"]] = a
    missing = sorted(prompt_ids - answers.keys())
    out = REVIEW_DIR / "opus_answers.jsonl"
    out.write_text("".join(json.dumps(answers[k], ensure_ascii=False) + "\n" for k in sorted(answers)),
                   encoding="utf-8")
    print(f"{len(answers)}/{len(prompt_ids)} respuestas válidas → {out.relative_to(ROOT)}"
          + (f"; faltan {', '.join(missing)}" if missing else ""))
    return 0 if not missing else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sheet = sub.add_parser("sheet", help="elige la muestra y escribe la planilla a completar")
    sheet.add_argument("--force", action="store_true", help="pisa la planilla aunque exista")
    opus = sub.add_parser("opus", help="exporta los prompts de (b) para Claude Opus 5.5, en tandas")
    opus.add_argument("--force", action="store_true", help="vuelve a escribir los prompts aunque haya respuestas")
    sub.add_parser("opus-check", help="valida las respuestas de Opus y las une en opus_answers.jsonl")
    args = parser.parse_args()
    if args.cmd == "sheet":
        return cmd_sheet(args.force)
    if args.cmd == "opus":
        return cmd_opus(args.force)
    if args.cmd == "opus-check":
        return cmd_opus_check()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
