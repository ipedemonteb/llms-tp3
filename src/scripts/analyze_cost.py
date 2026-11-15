"""Análisis de costo computacional y uso de tokens (extra de la consigna; detalle en tprag.eval.cost). No usa API ni
modelos: lee los reportes de indexado, las corridas permisivas de dev del generador (base y k1-rr30) y su análisis,
el manifiesto del estudio del reranker y el caché del LLM (`data/generate/cache/llm.sqlite`, para el uso real).

Uso:
    uv run --group analysis python src/scripts/analyze_cost.py

Escribe results/study/cost_analysis.json, results/study/cost_table.md y results/study/figures/cost/*.png. El caché
del LLM es local (no versionado): las llamadas que no están ahí se cuentan con el tokenizer y la calibración, y la
tabla dice cuántas fueron exactas.
"""

import argparse
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

from tprag.eval.cost import (calibrate, call_usage, index_row, per_thousand, reranker_latency, summarize_calls)
from tprag.eval.outputs import refuse_overwrite
from tprag.generate.llm import CACHE_PATH, CHAT_MODELS
from tprag.split.store import ROOT
from tprag.split.tokens import token_counter

from analyze_generation import load_items  # mismo directorio de scripts

INDEX_DIR, GEN_DIR, STUDY_DIR = ROOT / "results" / "index", ROOT / "results" / "eval" / "gen", ROOT / "results" / "study"
RUNS = {"base": "s3-1024__e4__r1__cap150__qA__gpt-4o-mini__dev",
        "rerank": "s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4o-mini__dev"}
RERANK_MANIFEST = STUDY_DIR / "rerank_full_dev_estudio_manifest.json"
FINAL_INDEX = "s3-1024__e4__r1__cap150"
CHOSEN_RERANKER = ("k1", 30)
LLM = CHAT_MODELS["gpt-4o-mini"]
OUT_JSON, OUT_MD, OUT_FIG = STUDY_DIR / "cost_analysis.json", STUDY_DIR / "cost_table.md", STUDY_DIR / "figures" / "cost"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cache_lookup(path: Path):
    """Lector del caché del LLM (solo lectura); sin caché, nada es exacto."""
    if not path.exists():
        return (lambda key: None), 0
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    total = db.execute("SELECT COUNT(*) FROM responses WHERE model = ?", (LLM.name,)).fetchone()[0]

    def lookup(key: str) -> dict | None:
        row = db.execute("SELECT response FROM responses WHERE model = ? AND key = ?", (LLM.name, key)).fetchone()
        return json.loads(row[0]) if row else None

    return lookup, total


def cache_spend(path: Path) -> dict:
    """Todo lo que se pagó y quedó en este caché (cada llamada se paga una sola vez)."""
    if not path.exists():
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0}
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = [json.loads(r[0])["usage"] for r in db.execute("SELECT response FROM responses WHERE model = ?",
                                                           (LLM.name,))]
    i, o = sum(u["input_tokens"] for u in rows), sum(u["output_tokens"] for u in rows)
    return {"calls": len(rows), "input_tokens": i, "output_tokens": o, "usd": LLM.usd(i, o)}


def generator_costs(items: dict[str, dict], lookup) -> tuple[dict, dict]:
    """Calibra con las llamadas del caché y cuenta todas las de las dos corridas."""
    count = token_counter(LLM.tokenizer)
    records = {v: read_jsonl(GEN_DIR / f"{name}.items.jsonl") for v, name in RUNS.items()}

    def usages(calibration):
        return {v: [u for u in (call_usage(r, items[r["id"]]["code"], count, lookup, calibration) for r in recs) if u]
                for v, recs in records.items()}

    calibration = calibrate([{"real_in": u["input"], "counted_in": u["counted_in"], "real_out": u["output"],
                              "rebuilt_out": u["rebuilt_out"]} for us in usages(None).values() for u in us])
    return calibration, {v: summarize_calls(us, LLM) | {"scripts": len(records[v])}
                         for v, us in usages(calibration).items()}


def operating_points(gen: dict) -> dict:
    """Llamadas al LLM y USD cada 1.000 scripts con imports problemáticos, con el umbral elegido en dev y con la regla
    permisiva del retrieval (pasa ≥ 95% de los positivos)."""
    out = {}
    for v, name in RUNS.items():
        analysis = json.loads((STUDY_DIR / f"generation_{name}_analysis.json").read_text(encoding="utf-8"))
        scripts = gen[v]["scripts"]
        rule = analysis["comparison"]["retrieval"]["row"]
        points = {"elegido": (analysis["chosen"]["threshold"], analysis["chosen"]["llm_calls"]),
                  "permisivo_95": (rule["threshold"], rule["llm_calls"])}
        out[v] = {k: {"threshold": t, "llm_calls": c, "scripts": scripts, "call_rate": c / scripts,
                      "usd_per_1000_scripts": per_thousand(gen[v]["per_call"]["usd_mean"], c / scripts)}
                  for k, (t, c) in points.items()}
    return out


def fmt(x: float, nd: int = 0) -> str:
    s = f"{x:,.{nd}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def table(result: dict) -> str:
    cal, gen, ops = result["calibration"], result["generator"], result["operating_points"]
    lines = ["# Costo computacional y uso de tokens", "",
             f"Generado por `src/scripts/analyze_cost.py` el {result['date']}. Sin API: reportes de indexado, corridas "
             "permisivas de dev del generador (todos los scripts con imports problemáticos llegan al LLM) y "
             "manifiesto del estudio del reranker. Precios de gpt-4o-mini: "
             f"{fmt(LLM.usd_per_m_input, 2)} / {fmt(LLM.usd_per_m_output, 2)} USD por millón de tokens de entrada / "
             "salida.", "", "## 1. Indexado (una vez por configuración)", "",
             "| Colección | Embedder | Dónde | Casos de uso | Tokens del corpus | Embebidos ahora | Del caché | USD | "
             "Segundos | Textos/s | USD del corpus completo |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["index"]:
        lines.append(f"| `{r['collection']}` | {r['model']} | {r['provider']} | {fmt(r['use_cases'])} | "
                     f"{fmt(r['tokens_total'])} | {fmt(r['embedded_texts'])} | {fmt(r['cache_hits'])} | "
                     f"{fmt(r['usd'], 4)} | {fmt(r['seconds'], 0)} | "
                     f"{fmt(r['texts_per_second'], 1) if r['texts_per_second'] else '-'} | "
                     f"{fmt(r['usd_full_corpus'], 4)} |")
    lines += ["", "Con la API (E1, text-embedding-3-small) indexar las 101 librerías cuesta ~0,065 USD; con el embedder "
              "local elegido (E4, jina-code) cuesta 0 USD y lo que se paga es tiempo de máquina.", "",
              "## 2. Generador: tokens por llamada a gpt-4o-mini", "",
              f"Entrada contada con `{LLM.tokenizer}` sobre el mensaje reconstruido + {fmt(cal['input_overhead'])} "
              f"tokens fijos (esquema JSON y formato), calibrado con {cal['n']} llamadas del caché (rango del "
              f"sobrecosto: {cal['input_overhead_range'][0]}-{cal['input_overhead_range'][1]}). Salida reconstruida × "
              f"{fmt(cal['output_ratio'], 2)} (rango real/reconstruida {fmt(cal['output_ratio_range'][0], 2)}-"
              f"{fmt(cal['output_ratio_range'][1], 2)}; error medio de la salida estimada sobre esas llamadas: "
              f"{fmt(100 * cal['output_mean_abs_error'], 0)}%). Las llamadas del caché usan el uso real.", "",
              "| Versión | Llamadas | Exactas | Entrada media | p95 | Salida media | p95 | USD / llamada | p95 | "
              "Salida / USD | USD de la corrida |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v, g in gen.items():
        p = g["per_call"]
        lines.append(f"| {v} | {g['calls']} | {g['exact']} | {fmt(p['input_mean'])} | {fmt(p['input_p95'])} | "
                     f"{fmt(p['output_mean'])} | {fmt(p['output_p95'])} | {fmt(p['usd_mean'], 5)} | "
                     f"{fmt(p['usd_p95'], 5)} | {fmt(100 * g['output_share_of_usd'], 0)}% | {fmt(g['usd'], 4)} |")
    lines += ["", "De qué está hecha la entrada (tokens medios por llamada):", "",
              "| Versión | Instrucciones | Esquema y formato | Script | Fragmentos recuperados |", "|---|---|---|---|---|"]
    for v, g in gen.items():
        pa = g["input_parts_mean"]
        lines.append(f"| {v} | {fmt(pa['system'])} | {fmt(cal['input_overhead'])} | {fmt(pa['script'])} | "
                     f"{fmt(pa['context'])} |")
    lines += ["", "## 3. Costo por cada 1.000 scripts con imports problemáticos", "",
              "Los scripts sin imports problemáticos no llaman al LLM ni al reranker (solo el validador, gratis). "
              "Con el umbral, solo llegan al LLM los imports con un fragmento suficientemente parecido.", "",
              "| Versión | Umbral | Valor | Llamadas al LLM | USD / 1.000 scripts |", "|---|---|---|---|---|"]
    for v, pts in ops.items():
        for k, o in pts.items():
            lines.append(f"| {v} | {k} | {fmt(o['threshold'], 4)} | {o['llm_calls']}/{o['scripts']} "
                         f"({fmt(100 * o['call_rate'], 0)}%) | {fmt(o['usd_per_1000_scripts'], 3)} |")
    lines += ["", "## 4. Latencia del reranker (por consulta = por import problemático)", "",
              f"Medido en el estudio (`{RERANK_MANIFEST.name}`), {result['reranker'][0]['machine']}, "
              f"{result['reranker'][0]['dtype']} en {result['reranker'][0]['device']}, de a un par. Un par por "
              "candidato reordenado.", "", "| Reranker | K | s / par | s / consulta | Memoria pico (MB) |",
              "|---|---|---|---|---|"]
    for r in result["reranker"]:
        mark = " ⭐" if (r["reranker"], r["k"]) == CHOSEN_RERANKER else ""
        lines.append(f"| {r['label']}{mark} | {r['k']} | {fmt(r['seconds_per_pair'], 3)} | "
                     f"{fmt(r['seconds_per_query'], 1)} | {fmt(r['peak_memory_mb'])} |")
    s = result["spend"]
    lines += ["", "## 5. Gasto de API del TP", "",
              "| Concepto | Llamadas / textos | Tokens | USD |", "|---|---|---|---|",
              f"| Embeddings con la API (todos los índices E1) | {fmt(s['index']['texts'])} | "
              f"{fmt(s['index']['tokens'])} | {fmt(s['index']['usd'], 4)} |",
              f"| Generador gpt-4o-mini (caché local) | {fmt(s['llm_cache']['calls'])} | "
              f"{fmt(s['llm_cache']['input_tokens'] + s['llm_cache']['output_tokens'])} | "
              f"{fmt(s['llm_cache']['usd'], 4)} |",
              f"| **Total registrado** | | | **{fmt(s['total_usd'], 4)}** de {fmt(s['budget_usd'], 0)} USD |", "",
              "El caché local tiene las llamadas hechas en esta máquina; las corridas de otros integrantes y el juez "
              "(caché propio, `judge.sqlite`) se suman con sus cachés. Las llamadas repetidas no se pagan dos veces "
              "(caché por prompt).", "",
              "## 6. Qué no está medido", "",
              "- Latencia del LLM: las corridas salen del caché (el tiempo guardado es el de leerlo).",
              "- Retrieval (embedding de la consulta con E4 + búsqueda en Chroma): no se cronometró por consulta.",
              "- Las cifras del reranker son de una Mac M3 (MPS); en CPU sin aceleración serían más altas."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="pisar las salidas si ya existen")
    parser.add_argument("--no-plots", action="store_true", help="no armar los gráficos (no necesita matplotlib)")
    args = parser.parse_args()
    if msg := refuse_overwrite([OUT_JSON, OUT_MD], args.force):
        print(msg)
        return 1

    reports = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(INDEX_DIR.glob("*.json"))]
    index = [index_row(r) for r in reports]
    full = [r for r in index if "__sub-" not in r["collection"]]
    lookup, cached_calls = cache_lookup(CACHE_PATH)
    calibration, gen = generator_costs(load_items(), lookup)
    api_index = [r for r in index if r["provider"] == "API"]
    spend_index = {"texts": sum(r["embedded_texts"] for r in api_index),
                   "tokens": sum(r["embedded_tokens"] for r in api_index),
                   "usd": round(sum(r["usd"] for r in api_index), 6)}
    llm_cache = cache_spend(CACHE_PATH)
    result = {"date": date.today().isoformat(), "llm": LLM.name, "llm_cache_calls": cached_calls,
              "calibration": calibration, "index": full, "index_all": index, "generator": gen,
              "operating_points": None,
              "reranker": reranker_latency(json.loads(RERANK_MANIFEST.read_text(encoding="utf-8"))),
              "spend": {"index": spend_index, "llm_cache": llm_cache,
                        "total_usd": round(spend_index["usd"] + llm_cache["usd"], 4), "budget_usd": 10}}
    result["operating_points"] = operating_points(gen)

    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_MD.write_text(table(result), encoding="utf-8")
    print(OUT_JSON.relative_to(ROOT), OUT_MD.relative_to(ROOT), sep="\n")
    if not args.no_plots:
        from tprag.eval.plots_cost import cost_figures  # noqa: PLC0415  (necesita el grupo `analysis`)

        for path in cost_figures(result, OUT_FIG):
            print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
