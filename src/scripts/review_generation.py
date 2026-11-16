"""Revisión manual de la generación (docs/agents/metricas.md §3.8): valida el juez de Answer Relevance (b) y la
heurística de Faithfulness contra una persona (y el juez contra Claude Opus 5.5). No usa API.

Uso:
    uv run python src/scripts/review_generation.py sheet       # elige la muestra y escribe la planilla a completar
    uv run python src/scripts/review_generation.py opus        # exporta los prompts de (b) para Claude Opus 5.5
    uv run python src/scripts/review_generation.py opus-check  # valida y une las respuestas de Opus
    uv run python src/scripts/review_generation.py context R02 [--chunks]  # dónde aparece cada identificador
    uv run python src/scripts/review_generation.py agreement   # acuerdo con la planilla completa

`sheet`: muestra estratificada de 20 scripts cambiados de las dos corridas permisivas de dev (10 de la base y 10 del
reranker), con cupos por (éxito del nivel 1 × veredicto del juez) y sin repetir un ítem entre corridas; semilla fija.
Escribe results/study/review/sample.json y planilla.md. La planilla es a ciegas: no muestra el veredicto del juez ni el
éxito del nivel 1. No pisa una planilla existente (puede estar completada) salvo con --force.

`opus`: los mismos prompts de (b) que recibió gpt-4o-mini (instrucciones + mensaje + esquema JSON), en tandas, en
results/study/review/opus/prompts_<k>.jsonl. Cada tanda la responde un agente de Claude Opus 5.5 lanzado desde Claude
Code (0 USD del presupuesto de OpenAI) en opus/answers_<k>.jsonl. `opus-check` valida cada respuesta con el mismo
esquema (`Preservation`) y las une en opus_answers.jsonl.

`context`: para un ítem de la planilla, todas las líneas de los chunks en las que aparece cada identificador de su tabla
(misma numeración y mismo criterio que la heurística), enteras y marcadas como código o prosa; con --chunks, además el
texto completo de cada chunk. Solo imprime: no escribe archivos ni muestra la corrida, el éxito ni el veredicto.

`opus-all`: los mismos prompts de (b) para el resto de los scripts que juzgó gpt-4o-mini (todos los cambiados de las dos
corridas, menos los 20 de la muestra, que ya tienen respuesta de Opus), en tandas, en results/study/review/opus_all/
(clave `<versión>:<ítem>`). `opus-all-check` valida las respuestas y las une en opus_all_answers.jsonl. `judge-vs-opus`
compara gpt-4o-mini con Opus sobre todos los scripts juzgados (sin la revisión humana, que se hizo con ayuda de Claude
y no es independiente de Opus): preservación media de cada uno, diferencia pareada con IC y acuerdo. Escribe
results/study/judge_vs_opus.{json,md}.

`agreement` (con la planilla completa): une la planilla, el .judge.jsonl de cada corrida y opus_answers.jsonl (cada R
se asigna a su versión con sample.json). Preservación: acuerdo exacto, a un nivel y kappa de Cohen de gpt-4o-mini vs.
humano, gpt-4o-mini vs. Opus y Opus vs. humano, en total y por versión, con IC 95% por bootstrap sobre ítems.
Faithfulness: filas de la tabla que la persona marcó como error de la heurística (falsos sí / falsos no). Escribe
results/study/review_agreement.{json,md}.
"""

import argparse
import difflib
import json
import random
import re
from pathlib import Path

from tprag.eval.faithfulness import claims, mentions, record_context
from tprag.eval.judge import PRESERVE_SCORE, PRESERVE_SYSTEM, Preservation, preserve_prompt
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
OPUS_ALL_DIR = REVIEW_DIR / "opus_all"  # el resto de los scripts juzgados (judge-vs-opus)
JUDGE_VS_OPUS_OUT = ROOT / "results" / "study" / "judge_vs_opus"
# Acuerdo (metricas.md §3.8): el veredicto como escala no < parcial < si, para el acuerdo "a un nivel"
LEVEL = {"no": 0, "parcial": 1, "si": 2}
RATERS = ("humano", "gpt-4o-mini", "opus")
PAIRS = (("gpt-4o-mini", "humano"), ("gpt-4o-mini", "opus"), ("opus", "humano"))
BOOT = 10_000
AGREEMENT_OUT = ROOT / "results" / "study" / "review_agreement"


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


def name_hits(name: str, texts: list[str]) -> list[tuple[int, bool, str]]:
    """Cada línea de los chunks en la que `name` aparece como palabra completa (mismo criterio que la heurística):
    (número de chunk desde 1, ¿dentro de un bloque de código?, línea entera)."""
    out = []
    for k, text in enumerate(texts, 1):
        in_code = False
        for line in text.splitlines():
            is_fence = line.lstrip().startswith("```")
            if mentions(line, name):
                out.append((k, in_code or is_fence, line.strip()))
            if is_fence:
                in_code = not in_code
    return out


def highlight(line: str, name: str) -> str:
    return re.sub(rf"(?<![\w]){re.escape(name)}(?![\w])", lambda m: f"»{m.group(0)}«", line)


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
    batches = write_batches(prompts, OPUS_DIR)
    print(f"{len(prompts)} prompts en {batches} tandas → {OPUS_DIR.relative_to(ROOT)}/")
    return 0


def cmd_opus_check(opus_dir: Path = OPUS_DIR, out: Path = REVIEW_DIR / "opus_answers.jsonl") -> int:
    prompt_ids = {p["review_id"] for path in sorted(opus_dir.glob("prompts_*.jsonl")) for p in read_jsonl(path)}
    answers = {}
    for path in sorted(opus_dir.glob("answers_*.jsonl")):
        for row in read_jsonl(path):
            a = check_answer(prompt_ids, row)
            if a["review_id"] in answers:
                raise ValueError(f"{a['review_id']} respondido dos veces")
            answers[a["review_id"]] = a
    missing = sorted(prompt_ids - answers.keys())
    out.write_text("".join(json.dumps(answers[k], ensure_ascii=False) + "\n" for k in sorted(answers)),
                   encoding="utf-8")
    print(f"{len(answers)}/{len(prompt_ids)} respuestas válidas → {out.relative_to(ROOT)}"
          + (f"; faltan {', '.join(missing)}" if missing else ""))
    return 0 if not missing else 1


def remaining(judged: dict[str, dict], chosen: list[dict]) -> list[dict]:
    """Los scripts juzgados por gpt-4o-mini que no están en la muestra (esos ya tienen respuesta de Opus), en orden
    estable (versión, ítem), con la clave `<versión>:<ítem>` como review_id."""
    done = {(s["run"], s["id"]) for s in chosen}
    return [{"run": run, "id": i, "review_id": f"{run}:{i}"} for run in judged for i in sorted(judged[run])
            if (run, i) not in done]


def write_batches(prompts: list[dict], opus_dir: Path) -> int:
    opus_dir.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(prompts), OPUS_BATCH):
        path = opus_dir / f"prompts_{k // OPUS_BATCH + 1}.jsonl"
        path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prompts[k:k + OPUS_BATCH]),
                        encoding="utf-8")
    return -(-len(prompts) // OPUS_BATCH)


def cmd_opus_all(force: bool) -> int:
    if OPUS_ALL_DIR.exists() and any(OPUS_ALL_DIR.glob("answers_*.jsonl")) and not force:
        print(f"ya hay respuestas en {OPUS_ALL_DIR.relative_to(ROOT)}: no piso los prompts; usar --force")
        return 1
    chosen = json.loads((REVIEW_DIR / "sample.json").read_text(encoding="utf-8"))["items"]
    judged = {run: {r["id"]: r for r in read_jsonl(GEN_DIR / f"{name}.judge.jsonl")} for run, name in RUNS.items()}
    records = {run: {r["id"]: r for r in read_jsonl(GEN_DIR / f"{name}.items.jsonl")} for run, name in RUNS.items()}
    todo = remaining(judged, chosen)
    batches = write_batches(opus_prompts(todo, load_items(), records), OPUS_ALL_DIR)
    by_run = {run: sum(t["run"] == run for t in todo) for run in RUNS}
    print(f"juzgados por gpt-4o-mini: {', '.join(f'{run} {len(judged[run])}' for run in RUNS)}; ya respondidos por "
          f"Opus (muestra): {len(chosen)}")
    print(f"{len(todo)} prompts ({', '.join(f'{run} {n}' for run, n in by_run.items())}) en {batches} tandas de "
          f"hasta {OPUS_BATCH} → {OPUS_ALL_DIR.relative_to(ROOT)}/")
    return 0


def paired_difference(pairs: list[tuple[str, str]], boot: int = BOOT, seed: int = SEED) -> dict:
    """Diferencia pareada de preservación (primero − segundo, con si = 1, parcial = 0,5, no = 0) sobre los mismos ítems:
    media con IC 95% por bootstrap y en cuántos ítems el primero da más, lo mismo o menos."""
    diffs = [PRESERVE_SCORE[a] - PRESERVE_SCORE[b] for a, b in pairs]
    n, rng = len(diffs), random.Random(seed)
    means = [sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(boot)]
    return {"n": n, "mean": sum(diffs) / n, "ci": percentile_ci(means), "higher": sum(d > 0 for d in diffs),
            "equal": sum(d == 0 for d in diffs), "lower": sum(d < 0 for d in diffs)}


def judge_vs_opus_md(result: dict) -> str:
    lines = [
        "# gpt-4o-mini vs. Claude Opus 5.5 como jueces de preservación",
        "",
        "Todos los scripts cambiados que juzgó gpt-4o-mini en las corridas permisivas de dev, juzgados también por Opus 5.5 "
        "con el mismo prompt y el mismo esquema (metricas.md §3.8). Sin estratificar: los promedios son de la población "
        "juzgada. IC 95% por bootstrap sobre ítems. Sesgo a declarar: los scripts del dataset los escribió Claude.",
        "",
        "| Grupo | n | gpt-4o-mini | Opus 5.5 | Diferencia (gpt − Opus) | gpt más blando / igual / más estricto | "
        "Acuerdo exacto | A un nivel | Kappa |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for g, r in result["groups"].items():
        d, a = r["difference"], r["agreement"]
        lines.append(f"| {g} | {r['n']} | {fmt(r['mean']['gpt-4o-mini'])} | {fmt(r['mean']['opus'])} | "
                     f"{d['mean']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}] | {d['higher']} / {d['equal']} / "
                     f"{d['lower']} | {fmt(a['exact'], a['exact_ci'])} | {fmt(a['within_one'])} | "
                     f"{fmt(a['kappa'], a['kappa_ci'])} |")
    lines += ["", "Veredictos de cada juez:", "", "| Grupo | Juez | si | parcial | no |", "|---|---|---|---|---|"]
    for g, r in result["groups"].items():
        lines += [f"| {g} | {j} | " + " | ".join(str(r["verdicts"][j][v]) for v in VERDICTS) + " |"
                  for j in ("gpt-4o-mini", "opus")]
    conf = result["groups"]["total"]["agreement"]["confusion"]
    lines += ["", "Matriz de confusión, total (filas: gpt-4o-mini; columnas: Opus 5.5):", "",
              "| | " + " | ".join(VERDICTS) + " |", "|---|" + "---|" * len(VERDICTS)]
    lines += [f"| {r} | " + " | ".join(str(conf[r][c]) for c in VERDICTS) + " |" for r in VERDICTS]
    return "\n".join(lines) + "\n"


def cmd_judge_vs_opus() -> int:
    chosen = json.loads((REVIEW_DIR / "sample.json").read_text(encoding="utf-8"))["items"]
    judged = {run: {r["id"]: r for r in read_jsonl(GEN_DIR / f"{name}.judge.jsonl")} for run, name in RUNS.items()}
    opus = {(s["run"], s["id"]): r["output"]["preserves"] for s in chosen
            for r in read_jsonl(REVIEW_DIR / "opus_answers.jsonl") if r["review_id"] == s["review_id"]}
    opus |= {tuple(r["review_id"].split(":", 1)): r["output"]["preserves"]
             for r in read_jsonl(REVIEW_DIR / "opus_all_answers.jsonl")}
    keys = [(run, i) for run in RUNS for i in sorted(judged[run])]
    if missing := [f"{run}:{i}" for run, i in keys if (run, i) not in opus]:
        print(f"faltan {len(missing)} respuestas de Opus (correr opus-all-check): {', '.join(missing[:10])}...")
        return 1
    pairs = {k: (judged[k[0]][k[1]]["b"]["preserves"], opus[k]) for k in keys}
    groups = {"total": keys} | {run: [k for k in keys if k[0] == run] for run in RUNS}
    result = {"runs": RUNS, "boot": BOOT, "seed": SEED, "groups": {}}
    for g, ks in groups.items():
        ps = [pairs[k] for k in ks]
        result["groups"][g] = {
            "n": len(ps),
            "mean": {j: sum(PRESERVE_SCORE[p[x]] for p in ps) / len(ps) for x, j in enumerate(("gpt-4o-mini", "opus"))},
            "verdicts": {j: {v: sum(p[x] == v for p in ps) for v in VERDICTS}
                         for x, j in enumerate(("gpt-4o-mini", "opus"))},
            "difference": paired_difference(ps), "agreement": agreement(ps)}
    result["items"] = [{"run": run, "id": i, "gpt-4o-mini": pairs[(run, i)][0], "opus": pairs[(run, i)][1]}
                       for run, i in keys]
    JUDGE_VS_OPUS_OUT.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n",
                                                      encoding="utf-8")
    md = judge_vs_opus_md(result)
    JUDGE_VS_OPUS_OUT.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"→ {JUDGE_VS_OPUS_OUT.relative_to(ROOT)}.{{json,md}}")
    return 0


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Kappa de Cohen (sin pesos) entre dos evaluadores. None si el acuerdo por azar es 1 (los dos usan siempre el
    mismo veredicto): la kappa no está definida."""
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pe = sum(sum(a == v for a, _ in pairs) / n * sum(b == v for _, b in pairs) / n for v in VERDICTS)
    return None if pe == 1 else (po - pe) / (1 - pe)


def percentile_ci(values: list[float]) -> list[float] | None:
    if not values:
        return None
    s = sorted(values)
    return [s[int(0.025 * len(s))], s[min(len(s) - 1, int(0.975 * len(s)))]]


def agreement(pairs: list[tuple[str, str]], boot: int = BOOT, seed: int = SEED) -> dict:
    """Acuerdo entre dos evaluadores sobre los mismos ítems: exacto, a un nivel (no/parcial/si como escala) y kappa de
    Cohen, con IC 95% por bootstrap sobre ítems (percentiles; los remuestreos sin kappa definida no cuentan)."""
    n = len(pairs)
    exact = lambda ps: sum(a == b for a, b in ps) / len(ps)  # noqa: E731
    rng = random.Random(seed)
    exacts, kappas = [], []
    for _ in range(boot):
        resample = [pairs[rng.randrange(n)] for _ in range(n)]
        exacts.append(exact(resample))
        if (k := kappa(resample)) is not None:
            kappas.append(k)
    return {"n": n, "exact": exact(pairs), "exact_ci": percentile_ci(exacts),
            "within_one": sum(abs(LEVEL[a] - LEVEL[b]) <= 1 for a, b in pairs) / n,
            "kappa": kappa(pairs), "kappa_ci": percentile_ci(kappas),
            "confusion": {a: {b: sum(p == (a, b) for p in pairs) for b in VERDICTS} for a in VERDICTS}}


def heuristic_check(chosen: list[dict], human: dict[str, dict]) -> dict:
    """Heurística de Faithfulness contra la persona: de las filas de la tabla de cada ítem (los claims de
    sample.json, misma numeración que la planilla), cuántas marcó como error y de qué tipo."""
    rows = false_yes = false_no = 0
    with_error = []
    for s in chosen:
        errors, claims = human[s["review_id"]]["errores_heuristica"], s["claims"]
        if bad := [i for i in errors if not 1 <= i <= len(claims)]:
            raise ValueError(f"{s['review_id']}: errores_heuristica fuera de la tabla: {bad} (tiene {len(claims)} filas)")
        rows += len(claims)
        yes = sum(claims[i - 1]["in_context"] for i in errors)
        false_yes, false_no = false_yes + yes, false_no + len(errors) - yes
        if errors:
            with_error.append(s["review_id"])
    wrong = false_yes + false_no
    return {"items": len(chosen), "rows": rows, "errors": wrong, "false_yes": false_yes, "false_no": false_no,
            "accuracy": 1 - wrong / rows if rows else None, "items_with_error": with_error}


def fmt(x: float | None, ci: list[float] | None = None) -> str:
    if x is None:
        return "—"
    return f"{x:.2f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")


def agreement_md(result: dict) -> str:
    groups = list(result["faithfulness"])
    lines = [
        "# Acuerdo de la revisión manual",
        "",
        f"{result['n']} scripts de dev ({', '.join(f'{len(result['groups'][g])} de {g}' for g in groups[1:])}), "
        "muestra estratificada por (éxito del nivel 1 × veredicto de gpt-4o-mini), revisada a ciegas "
        "(metricas.md §3.8). IC 95% por bootstrap sobre ítems; con 10 ítems por versión son muy anchos.",
        "",
        "## Preservación de la tarea (Answer Relevance (b))",
        "",
        "| Par | Grupo | n | Acuerdo exacto | A un nivel | Kappa de Cohen |",
        "|---|---|---|---|---|---|",
    ]
    for pair, by_group in result["preservation"].items():
        for g, a in by_group.items():
            lines.append(f"| {pair} | {g} | {a['n']} | {fmt(a['exact'], a['exact_ci'])} | {fmt(a['within_one'])} | "
                         f"{fmt(a['kappa'], a['kappa_ci'])} |")
    lines += ["", "Puntaje medio de cada evaluador (si = 1, parcial = 0,5, no = 0):", "",
              "| Evaluador | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    for rater, by_group in result["mean_score"].items():
        lines.append(f"| {rater} | " + " | ".join(fmt(by_group[g]) for g in groups) + " |")
    for pair in result["preservation"]:
        conf = result["preservation"][pair]["total"]["confusion"]
        a, b = pair.split(" vs ")
        lines += ["", f"Matriz de confusión, total (filas: {a}; columnas: {b}):", "",
                  "| | " + " | ".join(VERDICTS) + " |", "|---|" + "---|" * len(VERDICTS)]
        lines += [f"| {r} | " + " | ".join(str(conf[r][c]) for c in VERDICTS) + " |" for r in VERDICTS]
    lines += ["", "## Faithfulness (heurística) contra la persona", "",
              "| Grupo | Ítems | Filas | Errores | Falsos sí | Falsos no | Acierto | Ítems con algún error |",
              "|---|---|---|---|---|---|---|---|"]
    for g, h in result["faithfulness"].items():
        lines.append(f"| {g} | {h['items']} | {h['rows']} | {h['errors']} | {h['false_yes']} | {h['false_no']} | "
                     f"{fmt(h['accuracy'])} | {', '.join(h['items_with_error']) or '—'} |")
    lines += ["", "## Detalle por ítem", "", "| R | Versión | Ítem | humano | gpt-4o-mini | opus | Errores heurística |",
              "|---|---|---|---|---|---|---|"]
    lines += [f"| {it['review_id']} | {it['run']} | {it['id']} | {it['humano']} | {it['gpt-4o-mini']} | {it['opus']} | "
              f"{', '.join(map(str, it['errores_heuristica'])) or 'ninguno'} |" for it in result["items"]]
    return "\n".join(lines) + "\n"


def cmd_agreement() -> int:
    chosen = json.loads((REVIEW_DIR / "sample.json").read_text(encoding="utf-8"))["items"]
    human = parse_sheet((REVIEW_DIR / "planilla.md").read_text(encoding="utf-8"))
    missing = [s["review_id"] for s in chosen if s["review_id"] not in human
               or human[s["review_id"]]["preserva"] is None or human[s["review_id"]]["errores_heuristica"] is None]
    if missing:
        print(f"falta completar en la planilla: {', '.join(missing)}")
        return 1
    judged = {run: {r["id"]: r for r in read_jsonl(GEN_DIR / f"{name}.judge.jsonl")} for run, name in RUNS.items()}
    opus = {r["review_id"]: r for r in read_jsonl(REVIEW_DIR / "opus_answers.jsonl")}
    verdicts = {s["review_id"]: {"humano": human[s["review_id"]]["preserva"],
                                 "gpt-4o-mini": judged[s["run"]][s["id"]]["b"]["preserves"],
                                 "opus": opus[s["review_id"]]["output"]["preserves"]} for s in chosen}
    groups = {"total": chosen} | {run: [s for s in chosen if s["run"] == run] for run in RUNS}
    result = {
        "n": len(chosen), "runs": RUNS, "boot": BOOT, "seed": SEED,
        "groups": {g: [s["review_id"] for s in items] for g, items in groups.items()},
        "preservation": {f"{a} vs {b}": {g: agreement([(verdicts[s["review_id"]][a], verdicts[s["review_id"]][b])
                                                         for s in items]) for g, items in groups.items()}
                         for a, b in PAIRS},
        "mean_score": {r: {g: sum(PRESERVE_SCORE[verdicts[s["review_id"]][r]] for s in items) / len(items)
                           for g, items in groups.items()} for r in RATERS},
        "faithfulness": {g: heuristic_check(items, human) for g, items in groups.items()},
        "items": [{"review_id": s["review_id"], "run": s["run"], "id": s["id"], **verdicts[s["review_id"]],
                   "errores_heuristica": human[s["review_id"]]["errores_heuristica"]} for s in chosen],
    }
    AGREEMENT_OUT.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n",
                                                  encoding="utf-8")
    md = agreement_md(result)
    AGREEMENT_OUT.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"→ {AGREEMENT_OUT.relative_to(ROOT)}.{{json,md}}")
    return 0


def cmd_context(review_id: str, show_chunks: bool) -> int:
    rid = review_id.upper()
    chosen = {s["review_id"]: s for s in json.loads((REVIEW_DIR / "sample.json").read_text(encoding="utf-8"))["items"]}
    if rid not in chosen:
        print(f"{rid} no está en la muestra (R01-R{len(chosen):02d})")
        return 1
    s = chosen[rid]  # trae también el grupo (éxito y veredicto del juez): no se imprime, la revisión es a ciegas
    record = next(r for r in read_jsonl(GEN_DIR / f"{RUNS[s['run']]}.items.jsonl") if r["id"] == s["id"])
    chunks = [ch for ctx in record.get("contexts", []) for ch in ctx["chunks"]]  # mismo orden que record_context
    texts = [ch["text"] for ch in chunks]
    rows = claim_rows(load_items()[s["id"]]["code"], record["code"], record_context(record))
    print(f"{rid}: {len(chunks)} chunks en el contexto")
    for k, ch in enumerate(chunks, 1):
        print(f"  chunk {k}: {ch['header']}")
    for i, r in enumerate(rows, 1):
        print(f"\n#{i} {r['name']} ({r['kind']}, línea {r['line']}) — heurística: {'sí' if r['in_context'] else 'no'}")
        hits = name_hits(r["name"], texts)
        for k, in_code, line in hits:
            print(f"  chunk {k} [{'código' if in_code else 'prosa'}]  {highlight(line, r['name'])}")
        print(f"  ({len(hits)} apariciones)" if hits else f"  no aparece en ninguno de los {len(chunks)} chunks")
    if show_chunks:
        for k, (ch, text) in enumerate(zip(chunks, texts), 1):
            print(f"\n{'=' * 100}\nchunk {k}/{len(chunks)} · {ch['header']} · id {ch['id']}\n{'=' * 100}\n{text}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sheet = sub.add_parser("sheet", help="elige la muestra y escribe la planilla a completar")
    sheet.add_argument("--force", action="store_true", help="pisa la planilla aunque exista")
    opus = sub.add_parser("opus", help="exporta los prompts de (b) para Claude Opus 5.5, en tandas")
    opus.add_argument("--force", action="store_true", help="vuelve a escribir los prompts aunque haya respuestas")
    sub.add_parser("opus-check", help="valida las respuestas de Opus y las une en opus_answers.jsonl")
    sub.add_parser("agreement", help="acuerdo humano / gpt-4o-mini / Opus y heurística contra la persona")
    opus_all = sub.add_parser("opus-all", help="exporta los prompts de (b) del resto de los scripts juzgados, en tandas")
    opus_all.add_argument("--force", action="store_true", help="vuelve a escribir los prompts aunque haya respuestas")
    sub.add_parser("opus-all-check", help="valida esas respuestas y las une en opus_all_answers.jsonl")
    sub.add_parser("judge-vs-opus", help="gpt-4o-mini vs. Opus sobre todos los scripts juzgados")
    context = sub.add_parser("context", help="muestra dónde aparece cada identificador en los chunks de un ítem")
    context.add_argument("review_id", help="p. ej. R02")
    context.add_argument("--chunks", action="store_true", help="además, el texto completo de cada chunk")
    args = parser.parse_args()
    if args.cmd == "sheet":
        return cmd_sheet(args.force)
    if args.cmd == "opus":
        return cmd_opus(args.force)
    if args.cmd == "opus-check":
        return cmd_opus_check()
    if args.cmd == "agreement":
        return cmd_agreement()
    if args.cmd == "opus-all":
        return cmd_opus_all(args.force)
    if args.cmd == "opus-all-check":
        return cmd_opus_check(OPUS_ALL_DIR, REVIEW_DIR / "opus_all_answers.jsonl")
    if args.cmd == "judge-vs-opus":
        return cmd_judge_vs_opus()
    if args.cmd == "context":
        return cmd_context(args.review_id, args.chunks)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
