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

`opus-run <corrida> --label <nombre>` / `opus-run-check --label <nombre>` / `judge-vs-opus-run <corrida> --label
<nombre>`: lo mismo para una corrida cualquiera (p. ej. la iteración elegida): todos sus scripts juzgados, en
review/opus_<nombre>_all/, opus_<nombre>_all_answers.jsonl y judge_vs_opus_<nombre>.{json,md}. Ninguno pisa archivos.

`opus-rubric` / `opus-rubric-check`: revisión con la rúbrica común (tprag.eval.judge.PRESERVE_RUBRIC) del reranker sin
iterar y de `__iter6` (los scripts juzgados que pasan el umbral), a ciegas (ids neutros; la clave en
opus_rubric_key.json), repartida en 3 agentes con las dos corridas intercaladas y 6 anclas que juzgan los tres (para
medir si aplicaron el mismo criterio). Respuestas en review/opus_rubric/agent_<a>/ y opus_rubric_answers.jsonl.

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


JUDGE_VS_OPUS_INTRO = (
    "Todos los scripts cambiados que juzgó gpt-4o-mini en las corridas permisivas de dev, juzgados también por Opus 5.5 "
    "con el mismo prompt y el mismo esquema (metricas.md §3.8). Sin estratificar: los promedios son de la población "
    "juzgada. IC 95% por bootstrap sobre ítems. Sesgo a declarar: los scripts del dataset los escribió Claude.")


def judge_vs_opus_md(result: dict, intro: str = JUDGE_VS_OPUS_INTRO) -> str:
    lines = [
        "# gpt-4o-mini vs. Claude Opus 5.5 como jueces de preservación",
        "",
        intro,
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


def compare_groups(pairs: dict, groups: dict[str, list]) -> dict:
    """Por grupo: preservación media de cada juez, veredictos, diferencia pareada (gpt − Opus) y acuerdo."""
    out = {}
    for g, ks in groups.items():
        ps = [pairs[k] for k in ks]
        if not ps:
            continue
        out[g] = {
            "n": len(ps),
            "mean": {j: sum(PRESERVE_SCORE[p[x]] for p in ps) / len(ps) for x, j in enumerate(("gpt-4o-mini", "opus"))},
            "verdicts": {j: {v: sum(p[x] == v for p in ps) for v in VERDICTS}
                         for x, j in enumerate(("gpt-4o-mini", "opus"))},
            "difference": paired_difference(ps), "agreement": agreement(ps)}
    return out


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
    result = {"runs": RUNS, "boot": BOOT, "seed": SEED, "groups": compare_groups(pairs, groups)}
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


# --- Una corrida cualquiera (p. ej. la iteración elegida, `__iter6`): todo en carpetas y archivos propios ------------

def run_dir(label: str) -> Path:
    return REVIEW_DIR / f"opus_{label}_all"


def run_answers(label: str) -> Path:
    return REVIEW_DIR / f"opus_{label}_all_answers.jsonl"


def run_judge_vs_opus_out(label: str) -> Path:
    return ROOT / "results" / "study" / f"judge_vs_opus_{label}"


def cmd_opus_run(run: str, label: str) -> int:
    """Los prompts de (b) de todos los scripts que juzgó gpt-4o-mini en `run`, en tandas, en opus_<label>_all/. Se niega
    si la carpeta ya existe (no pisa prompts ni respuestas)."""
    out = run_dir(label)
    if out.exists():
        print(f"ya existe {out.relative_to(ROOT)}: no se pisa (usar otro --label)")
        return 1
    judged = read_jsonl(GEN_DIR / f"{run}.judge.jsonl")
    records = {label: {r["id"]: r for r in read_jsonl(GEN_DIR / f"{run}.items.jsonl")}}
    todo = [{"run": label, "id": r["id"], "review_id": f"{label}:{r['id']}"} for r in sorted(judged, key=lambda r: r["id"])]
    batches = write_batches(opus_prompts(todo, load_items(), records), out)
    print(f"{len(todo)} prompts de {run} en {batches} tandas de hasta {OPUS_BATCH} → {out.relative_to(ROOT)}/")
    return 0


def cmd_opus_run_check(label: str) -> int:
    """Valida las respuestas de opus_<label>_all/ y, solo si están todas, las une en opus_<label>_all_answers.jsonl
    (se niega si ya existe)."""
    out = run_answers(label)
    if out.exists():
        print(f"ya existe {out.relative_to(ROOT)}: no se pisa")
        return 1
    folder = run_dir(label)
    prompt_ids = {p["review_id"] for path in sorted(folder.glob("prompts_*.jsonl")) for p in read_jsonl(path)}
    answers = {}
    for path in sorted(folder.glob("answers_*.jsonl")):
        for row in read_jsonl(path):
            a = check_answer(prompt_ids, row)
            if a["review_id"] in answers:
                raise ValueError(f"{a['review_id']} respondido dos veces")
            answers[a["review_id"]] = a
    if missing := sorted(prompt_ids - answers.keys()):
        print(f"{len(answers)}/{len(prompt_ids)} respuestas válidas; faltan {', '.join(missing)}: no se escribe nada")
        return 1
    out.write_text("".join(json.dumps(answers[k], ensure_ascii=False) + "\n" for k in sorted(answers)), encoding="utf-8")
    print(f"{len(answers)}/{len(prompt_ids)} respuestas válidas → {out.relative_to(ROOT)}")
    return 0


def cmd_judge_vs_opus_run(run: str, label: str) -> int:
    """gpt-4o-mini vs. Opus en `run`, sobre los scripts juzgados que pasan el umbral elegido (los mismos que promedia
    analyze_generation.py): total, exitosos y no exitosos. Escribe judge_vs_opus_<label>.{json,md}; no los pisa."""
    from tprag.eval.threshold import RETRIEVAL, choose, sweep, targets  # noqa: PLC0415

    out = run_judge_vs_opus_out(label)
    if any(out.with_suffix(x).exists() for x in (".json", ".md")):
        print(f"ya existe {out.relative_to(ROOT)}.json/.md: no se pisa")
        return 1
    items = load_items()
    records = {r["id"]: r for r in read_jsonl(GEN_DIR / f"{run}.items.jsonl")}
    threshold = choose(sweep(targets(list(records.values()), {i: it["code"] for i, it in items.items()})),
                       criterion=RETRIEVAL)["threshold"]
    judged = {r["id"]: r for r in read_jsonl(GEN_DIR / f"{run}.judge.jsonl")}
    opus = {r["review_id"].split(":", 1)[1]: r["output"]["preserves"] for r in read_jsonl(run_answers(label))}
    keys = [i for i in sorted(judged) if any(t["retrieval_score"] is not None and t["retrieval_score"] >= threshold
                                             for t in records[i]["targets"])]
    if missing := [i for i in keys if i not in opus]:
        print(f"faltan {len(missing)} respuestas de Opus: {', '.join(missing[:10])}")
        return 1
    pairs = {i: (judged[i]["b"]["preserves"], opus[i]) for i in keys}
    groups = {"total": keys, "exitosos": [i for i in keys if records[i]["success"]],
              "no exitosos": [i for i in keys if not records[i]["success"]]}
    result = {"run": run, "label": label, "threshold": threshold, "boot": BOOT, "seed": SEED,
              "groups": compare_groups(pairs, groups),
              "items": [{"id": i, "success": records[i]["success"], "gpt-4o-mini": pairs[i][0], "opus": pairs[i][1]}
                        for i in keys]}
    intro = (f"Los scripts cambiados de `{run}` que juzgó gpt-4o-mini y que pasan el umbral elegido ({threshold:g}), "
             "juzgados también por Opus 5.5 con el mismo prompt y el mismo esquema. Exitosos / no exitosos: éxito del "
             "nivel 1 del script. IC 95% por bootstrap sobre ítems. Sesgo a declarar: los scripts del dataset los "
             "escribió Claude.")
    md = judge_vs_opus_md(result, intro)
    out.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"→ {out.relative_to(ROOT)}.{{json,md}}")
    return 0


# --- Revisión con rúbrica común (2026-11-17): reranker sin iterar e `__iter6`, a ciegas, 3 agentes con anclas -----------

RUBRIC_RUNS = {"rerank": RUNS["rerank"], "iter6": RUNS["rerank"] + "__iter6"}
RUBRIC_DIR, RUBRIC_KEY = REVIEW_DIR / "opus_rubric", REVIEW_DIR / "opus_rubric_key.json"
RUBRIC_ANSWERS = REVIEW_DIR / "opus_rubric_answers.jsonl"
RUBRIC_AGENTS, RUBRIC_ANCHORS = 3, {"si": 2, "parcial": 2, "no": 2}  # anclas: de iter6, por el veredicto anterior


def passing(run: str, items: dict[str, dict]) -> list[str]:
    """Los scripts juzgados de `run` con algún target sobre el umbral elegido (los que promedia analyze_generation.py)."""
    from tprag.eval.threshold import RETRIEVAL, choose, sweep, targets  # noqa: PLC0415

    records = {r["id"]: r for r in read_jsonl(GEN_DIR / f"{run}.items.jsonl")}
    threshold = choose(sweep(targets(list(records.values()), {i: it["code"] for i, it in items.items()})),
                       criterion=RETRIEVAL)["threshold"]
    judge_path = GEN_DIR / f"{run}.judge.jsonl"
    judged = (sorted(r["id"] for r in read_jsonl(judge_path)) if judge_path.exists() else  # sin juicio todavía: los
              sorted(i for i, r in records.items() if r["code"] != items[i]["code"]))  # cambiados (lo que se juzga)
    return [i for i in judged if any(t["retrieval_score"] is not None and t["retrieval_score"] >= threshold
                                     for t in records[i]["targets"])]


def rubric_plan(scripts: list[tuple[str, str]], previous: dict[str, str], seed: int = SEED) -> dict:
    """A ciegas: ids neutros (`s001`...) en orden aleatorio, anclas (de iter6, estratificadas por `previous`, el
    veredicto anterior de Opus) para los tres agentes y el resto repartido en ronda, así cada agente recibe scripts de
    las dos corridas intercalados."""
    rng = random.Random(seed)
    order = scripts[:]
    rng.shuffle(order)
    blind = {key: f"s{n:03d}" for n, key in enumerate(order, start=1)}
    anchors = []
    for verdict, n in RUBRIC_ANCHORS.items():
        pool = sorted(k for k in order if k[0] == "iter6" and previous.get(k[1]) == verdict)
        anchors += rng.sample(pool, n)
    rest = [k for k in order if k not in anchors]
    agents = {a: [k for n, k in enumerate(rest) if n % RUBRIC_AGENTS == a - 1] for a in range(1, RUBRIC_AGENTS + 1)}
    for a, keys in agents.items():  # las anclas, repartidas a lo largo de la lista de cada agente
        step = max(1, len(keys) // len(anchors))
        for n, anchor in enumerate(anchors):
            keys.insert(n * (step + 1), anchor)
    return {"key": {blind[k]: {"run": k[0], "id": k[1]} for k in order}, "anchors": [blind[k] for k in anchors],
            "agents": {str(a): [blind[k] for k in keys] for a, keys in agents.items()}}


def cmd_opus_rubric() -> int:
    """Exporta las tandas de cada agente en opus_rubric/agent_<a>/prompts_<k>.jsonl (prompt de (b) con la rúbrica) y la
    clave en opus_rubric_key.json. Se niega si ya existen."""
    from tprag.eval.judge import preserve_prompt  # noqa: PLC0415

    if RUBRIC_DIR.exists() or RUBRIC_KEY.exists():
        print(f"ya existe {RUBRIC_DIR.relative_to(ROOT)} o {RUBRIC_KEY.name}: no se pisa")
        return 1
    items = load_items()
    scripts = [(label, i) for label, run in RUBRIC_RUNS.items() for i in passing(run, items)]
    previous = {r["review_id"].split(":", 1)[1]: r["output"]["preserves"]
                for r in read_jsonl(REVIEW_DIR / "opus_iter6_all_answers.jsonl")}
    plan = rubric_plan(scripts, previous)
    records = {label: {r["id"]: r for r in read_jsonl(GEN_DIR / f"{run}.items.jsonl")} for label, run in RUBRIC_RUNS.items()}
    schema = Preservation.model_json_schema()
    for agent, ids in plan["agents"].items():
        prompts = []
        for b in ids:
            k = plan["key"][b]
            r = records[k["run"]][k["id"]]
            system, user = preserve_prompt(items[k["id"]]["code"], r["code"], [t["module"] for t in r["targets"]], True)
            prompts.append({"review_id": b, "system": system, "user": user, "schema": schema})
        write_batches(prompts, RUBRIC_DIR / f"agent_{agent}")
    RUBRIC_KEY.write_text(json.dumps({"runs": RUBRIC_RUNS, "seed": SEED, **plan}, ensure_ascii=False, indent=1) + "\n",
                          encoding="utf-8")
    by_run = {label: sum(k[0] == label for k in scripts) for label in RUBRIC_RUNS}
    print(f"{len(scripts)} scripts ({', '.join(f'{k} {n}' for k, n in by_run.items())}), {len(plan['anchors'])} anclas; "
          + ", ".join(f"agente {a}: {len(ids)}" for a, ids in plan["agents"].items()) + f" → {RUBRIC_DIR.relative_to(ROOT)}/")
    return 0


def cmd_opus_rubric_check(label: str | None = None) -> int:
    """Valida las respuestas de los tres agentes y, solo si están todas, escribe opus_rubric_answers.jsonl (una fila por
    agente y script, con la corrida y el ítem). Se niega si ya existe."""
    suffix = f"_{label}" if label else ""  # con --label: la revisión de opus-rubric-run (p. ej. iter7)
    answers_path = REVIEW_DIR / f"opus_rubric{suffix}_answers.jsonl"
    rubric_dir, key_path = REVIEW_DIR / f"opus_rubric{suffix}", REVIEW_DIR / f"opus_rubric{suffix}_key.json"
    if answers_path.exists():
        print(f"ya existe {answers_path.relative_to(ROOT)}: no se pisa")
        return 1
    plan = json.loads(key_path.read_text(encoding="utf-8"))
    rows, missing = [], []
    for agent, ids in plan["agents"].items():
        folder = rubric_dir / f"agent_{agent}"
        answers = {}
        for path in sorted(folder.glob("answers_*.jsonl")):
            for row in read_jsonl(path):
                a = check_answer(set(ids), row)
                if a["review_id"] in answers:
                    raise ValueError(f"agente {agent}: {a['review_id']} respondido dos veces")
                answers[a["review_id"]] = a
        missing += [f"{agent}:{b}" for b in ids if b not in answers]
        rows += [{"blind_id": b, **plan["key"][b], "agent": int(agent), "anchor": b in plan["anchors"],
                  "model": answers[b]["model"], "output": answers[b]["output"]} for b in ids if b in answers]
    if missing:
        print(f"faltan {len(missing)} respuestas ({', '.join(missing[:10])}): no se escribe nada")
        return 1
    answers_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} respuestas válidas → {answers_path.relative_to(ROOT)}")
    return 0


def cmd_opus_rubric_run(run: str, label: str) -> int:
    """La misma revisión con rúbrica para otra corrida (p. ej. `__iter7`), a ciegas, en 3 agentes, con **las mismas 6
    anclas** de opus-rubric (de `__iter6`, ya juzgadas por otros tres agentes): si los agentes nuevos las juzgan igual,
    la escala es la misma entre revisiones. Escribe opus_rubric_<label>/agent_<a>/ y opus_rubric_<label>_key.json; se
    niega si ya existen."""
    from tprag.eval.judge import preserve_prompt  # noqa: PLC0415

    folder, key_path = REVIEW_DIR / f"opus_rubric_{label}", REVIEW_DIR / f"opus_rubric_{label}_key.json"
    if folder.exists() or key_path.exists():
        print(f"ya existe {folder.relative_to(ROOT)} o {key_path.name}: no se pisa")
        return 1
    previous = json.loads(RUBRIC_KEY.read_text(encoding="utf-8"))
    anchors = [(previous["key"][b]["run"], previous["key"][b]["id"]) for b in previous["anchors"]]
    items = load_items()
    scripts = [(label, i) for i in passing(run, items)]
    rng = random.Random(SEED)
    order = scripts[:]
    rng.shuffle(order)
    everything = order + anchors
    blind = {k: f"u{n:03d}" for n, k in enumerate(rng.sample(everything, len(everything)), start=1)}
    agents = {a: [k for n, k in enumerate(order) if n % RUBRIC_AGENTS == a - 1] for a in range(1, RUBRIC_AGENTS + 1)}
    for keys in agents.values():
        step = max(1, len(keys) // len(anchors))
        for n, anchor in enumerate(anchors):
            keys.insert(n * (step + 1), anchor)
    runs = {label: run, "iter6": RUBRIC_RUNS["iter6"]}
    records = {lab: {r["id"]: r for r in read_jsonl(GEN_DIR / f"{name}.items.jsonl")} for lab, name in runs.items()}
    schema = Preservation.model_json_schema()
    for agent, keys in agents.items():
        prompts = []
        for k in keys:
            r = records[k[0]][k[1]]
            system, user = preserve_prompt(items[k[1]]["code"], r["code"], [t["module"] for t in r["targets"]], True)
            prompts.append({"review_id": blind[k], "system": system, "user": user, "schema": schema})
        write_batches(prompts, folder / f"agent_{agent}")
    plan = {"runs": runs, "seed": SEED, "key": {blind[k]: {"run": k[0], "id": k[1]} for k in everything},
            "anchors": [blind[k] for k in anchors], "anchors_previous": previous["anchors"],
            "agents": {str(a): [blind[k] for k in keys] for a, keys in agents.items()}}
    key_path.write_text(json.dumps(plan, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(scripts)} scripts de {run} + {len(anchors)} anclas (las de opus-rubric); "
          + ", ".join(f"agente {a}: {len(k)}" for a, k in agents.items()) + f" → {folder.relative_to(ROOT)}/")
    return 0


RUBRIC_OUT = ROOT / "results" / "study" / "judge_rubric"


def cmd_rubric_compare() -> int:
    """Con Opus con rúbrica como referencia, por corrida (reranker sin iterar, `__iter6`): su preservación media (total,
    exitosos, no exitosos) y cómo se le acerca cada juez de gpt-4o-mini (sin rúbrica: el de siempre; con rúbrica:
    `.judge__rubrica.jsonl`): media, diferencia pareada y acuerdo. Anclas: el veredicto de los tres agentes (y si
    coinciden). Escribe judge_rubric.{json,md}; no los pisa."""
    if any(RUBRIC_OUT.with_suffix(x).exists() for x in (".json", ".md")):
        print(f"ya existe {RUBRIC_OUT.relative_to(ROOT)}.json/.md: no se pisa")
        return 1
    rows = read_jsonl(RUBRIC_ANSWERS)
    anchors = {}
    for r in rows:
        if r["anchor"]:
            anchors.setdefault((r["run"], r["id"]), []).append(r["output"]["preserves"])
    opus = {(r["run"], r["id"]): r["output"]["preserves"] for r in rows if not r["anchor"]}
    opus |= {k: max(set(v), key=v.count) for k, v in anchors.items()}  # anclas: el veredicto de la mayoría
    result = {"anchors": {"n": len(anchors), "all_agree": sum(len(set(v)) == 1 for v in anchors.values()),
                          "verdicts": {f"{k[0]}:{k[1]}": v for k, v in anchors.items()}}, "runs": {}}
    lines = ["# Preservación de la tarea: Opus 5.5 con rúbrica común como referencia", "",
             "Scripts cambiados que pasan el umbral elegido, juzgados a ciegas por tres agentes de Opus 5.5 con la rúbrica "
             "común (tprag.eval.judge.PRESERVE_RUBRIC), con las dos corridas intercaladas. gpt-4o-mini: el juez de "
             "siempre (sin rúbrica) y el mismo prompt con la rúbrica. IC 95% por bootstrap sobre ítems.", "",
             f"Anclas (juzgadas por los tres agentes): coinciden en {result['anchors']['all_agree']} de "
             f"{result['anchors']['n']}.", ""]
    for label, run in RUBRIC_RUNS.items():
        records = {r["id"]: r for r in read_jsonl(GEN_DIR / f"{run}.items.jsonl")}
        keys = sorted(i for (l, i) in opus if l == label)
        groups = {"total": keys, "exitosos": [i for i in keys if records[i]["success"]],
                  "no exitosos": [i for i in keys if not records[i]["success"]]}
        out = {"opus_rubric": {g: {"n": len(ks), "mean": sum(PRESERVE_SCORE[opus[(label, i)]] for i in ks) / len(ks)
                                   if ks else None,
                                   "verdicts": {v: sum(opus[(label, i)] == v for i in ks) for v in VERDICTS}}
                               for g, ks in groups.items()}, "judges": {}}
        lines += [f"## `{label}` ({run})", "", "| Grupo | n | Opus con rúbrica | si / parcial / no |", "|---|---|---|---|"]
        lines += [f"| {g} | {r['n']} | {fmt(r['mean'])} | {r['verdicts']['si']} / {r['verdicts']['parcial']} / "
                  f"{r['verdicts']['no']} |" for g, r in out["opus_rubric"].items() if r["n"]]
        lines += ["", "| Juez gpt-4o-mini vs. Opus con rúbrica | Grupo | n | gpt | Opus | gpt − Opus | Acuerdo exacto | "
                  "Kappa |", "|---|---|---|---|---|---|---|---|"]
        for name, suffix in (("sin rúbrica", ""), ("con rúbrica", "__rubrica")):
            judged = {r["id"]: r["b"]["preserves"] for r in read_jsonl(GEN_DIR / f"{run}.judge{suffix}.jsonl")}
            pairs = {i: (judged[i], opus[(label, i)]) for i in keys}
            cmp = compare_groups(pairs, groups)
            out["judges"][name] = cmp
            for g, r in cmp.items():
                d, a = r["difference"], r["agreement"]
                lines.append(f"| {name} | {g} | {r['n']} | {fmt(r['mean']['gpt-4o-mini'])} | {fmt(r['mean']['opus'])} | "
                             f"{d['mean']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}] | "
                             f"{fmt(a['exact'], a['exact_ci'])} | {fmt(a['kappa'], a['kappa_ci'])} |")
        lines.append("")
        result["runs"][label] = out
    RUBRIC_OUT.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    md = "\n".join(lines)
    RUBRIC_OUT.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"→ {RUBRIC_OUT.relative_to(ROOT)}.{{json,md}}")
    return 0


def cmd_rubric_compare_run(run: str, label: str) -> int:
    """Opus con rúbrica para la corrida de opus-rubric-run `label`: su preservación (total, exitosos, no exitosos), gpt-4o-mini
    (el juez de siempre) contra Opus, y la diferencia pareada de la preservación según Opus contra `__iter6` (de
    opus-rubric, misma escala: las mismas anclas) sobre los scripts que cambiaron las dos. Escribe
    judge_rubric_<label>.{json,md}; no los pisa."""
    out = ROOT / "results" / "study" / f"judge_rubric_{label}"
    if any(out.with_suffix(x).exists() for x in (".json", ".md")):
        print(f"ya existe {out.relative_to(ROOT)}.json/.md: no se pisa")
        return 1
    rows = read_jsonl(REVIEW_DIR / f"opus_rubric_{label}_answers.jsonl")
    anchors_new = {}
    for r in rows:
        if r["anchor"]:
            anchors_new.setdefault(r["id"], []).append(r["output"]["preserves"])
    opus = {r["id"]: r["output"]["preserves"] for r in rows if not r["anchor"]}
    previous = {}
    for r in read_jsonl(RUBRIC_ANSWERS):
        if r["run"] == "iter6":
            previous.setdefault(r["id"], []).append(r["output"]["preserves"])
    opus6 = {i: max(set(v), key=v.count) for i, v in previous.items()}
    anchors_agree = sum(all(x == opus6[i] for x in v) for i, v in anchors_new.items())
    records = {r["id"]: r for r in read_jsonl(GEN_DIR / f"{run}.items.jsonl")}
    keys = sorted(opus)
    groups = {"total": keys, "exitosos": [i for i in keys if records[i]["success"]],
              "no exitosos": [i for i in keys if not records[i]["success"]]}
    summary = {g: {"n": len(ks), "mean": sum(PRESERVE_SCORE[opus[i]] for i in ks) / len(ks) if ks else None,
                   "verdicts": {v: sum(opus[i] == v for i in ks) for v in VERDICTS}} for g, ks in groups.items()}
    judged = {r["id"]: r["b"]["preserves"] for r in read_jsonl(GEN_DIR / f"{run}.judge.jsonl")}
    gpt = compare_groups({i: (judged[i], opus[i]) for i in keys}, groups)
    both = sorted(set(opus) & set(opus6))
    vs6 = paired_difference([(opus[i], opus6[i]) for i in both])
    result = {"run": run, "label": label, "anchors": {"n": len(anchors_new), "agree_with_previous": anchors_agree},
              "opus_rubric": summary, "gpt_vs_opus": gpt, "paired_vs_iter6": vs6,
              "items": [{"id": i, "success": records[i]["success"], "opus": opus[i], "gpt-4o-mini": judged.get(i),
                         "opus_iter6": opus6.get(i)} for i in keys]}
    lines = [f"# Preservación de la tarea con rúbrica: `{label}`", "",
             f"`{run}`: scripts cambiados que pasan el umbral, juzgados a ciegas por tres agentes de Opus 5.5 con la "
             "rúbrica común. Anclas (las de la revisión de sin iterar e `__iter6`): los agentes nuevos coinciden con la "
             f"revisión anterior en {anchors_agree} de {len(anchors_new)}. IC 95% por bootstrap sobre ítems.", "",
             "| Grupo | n | Opus con rúbrica | si / parcial / no |", "|---|---|---|---|"]
    lines += [f"| {g} | {r['n']} | {fmt(r['mean'])} | {r['verdicts']['si']} / {r['verdicts']['parcial']} / "
              f"{r['verdicts']['no']} |" for g, r in summary.items() if r["n"]]
    lines += ["", "| gpt-4o-mini (juez de siempre) vs. Opus con rúbrica | n | gpt | Opus | gpt − Opus | Acuerdo exacto | "
              "Kappa |", "|---|---|---|---|---|---|---|"]
    for g, r in gpt.items():
        d, a = r["difference"], r["agreement"]
        lines.append(f"| {g} | {r['n']} | {fmt(r['mean']['gpt-4o-mini'])} | {fmt(r['mean']['opus'])} | "
                     f"{d['mean']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}] | {fmt(a['exact'], a['exact_ci'])} | "
                     f"{fmt(a['kappa'], a['kappa_ci'])} |")
    lines += ["", f"Pareado contra `__iter6` (Opus con rúbrica, los {vs6['n']} scripts que cambiaron las dos corridas): "
              f"{label} − iter6 = {vs6['mean']:+.2f} [{vs6['ci'][0]:+.2f}, {vs6['ci'][1]:+.2f}]; {label} preserva más en "
              f"{vs6['higher']}, igual en {vs6['equal']}, menos en {vs6['lower']}.", ""]
    md = "\n".join(lines)
    out.with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"→ {out.relative_to(ROOT)}.{{json,md}}")
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
    for name, help_ in (("opus-run", "exporta los prompts de (b) de todos los scripts juzgados de una corrida"),
                        ("judge-vs-opus-run", "gpt-4o-mini vs. Opus en una corrida")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("run", help="nombre de la corrida en results/eval/gen/ (sin extensión)")
        p.add_argument("--label", required=True, help="nombre corto (p. ej. iter6): carpeta y archivos propios")
    sub.add_parser("opus-rubric", help="exporta la revisión con rúbrica (reranker sin iterar + iter6), a ciegas")
    rubric_check = sub.add_parser("opus-rubric-check", help="valida las respuestas con rúbrica y las une")
    rubric_check.add_argument("--label", help="la revisión de opus-rubric-run con ese nombre (p. ej. iter7)")
    sub.add_parser("rubric-compare", help="Opus con rúbrica como referencia: gpt-4o-mini con y sin rúbrica")
    rubric_cmp = sub.add_parser("rubric-compare-run", help="Opus con rúbrica de una corrida de opus-rubric-run vs. iter6")
    rubric_cmp.add_argument("run", help="nombre de la corrida en results/eval/gen/ (sin extensión)")
    rubric_cmp.add_argument("--label", required=True)
    rubric_run = sub.add_parser("opus-rubric-run", help="la misma revisión con rúbrica para otra corrida, mismas anclas")
    rubric_run.add_argument("run", help="nombre de la corrida en results/eval/gen/ (sin extensión)")
    rubric_run.add_argument("--label", required=True, help="nombre corto (p. ej. iter7)")
    check = sub.add_parser("opus-run-check", help="valida las respuestas de opus-run y las une")
    check.add_argument("--label", required=True)
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
    if args.cmd == "opus-rubric":
        return cmd_opus_rubric()
    if args.cmd == "opus-rubric-check":
        return cmd_opus_rubric_check(args.label)
    if args.cmd == "rubric-compare":
        return cmd_rubric_compare()
    if args.cmd == "rubric-compare-run":
        return cmd_rubric_compare_run(args.run, args.label)
    if args.cmd == "opus-rubric-run":
        return cmd_opus_rubric_run(args.run, args.label)
    if args.cmd == "opus-run":
        return cmd_opus_run(args.run, args.label)
    if args.cmd == "opus-run-check":
        return cmd_opus_run_check(args.label)
    if args.cmd == "judge-vs-opus-run":
        return cmd_judge_vs_opus_run(args.run, args.label)
    if args.cmd == "context":
        return cmd_context(args.review_id, args.chunks)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
