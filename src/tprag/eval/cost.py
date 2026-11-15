"""Análisis de costo computacional y uso de tokens (extra de la consigna). Lo arma `src/scripts/analyze_cost.py`.

Tres partes, todas sin API:

- **Indexado:** los reportes de `build_index.py` (`results/index/*.json`): tokens embebidos, USD pagados, caché y
  tiempo por colección.
- **Generador:** tokens por llamada a gpt-4o-mini de cada corrida de `eval_generation.py`. El detalle por ítem guarda
  los fragmentos que vio el LLM, así que el mensaje se reconstruye igual que en `generate/prompt.py` y se cuenta con
  el tokenizer del modelo. La entrada es exacta salvo un sobrecosto fijo (esquema JSON + formato de los mensajes) que
  se calibra con las respuestas del caché del LLM, que guardan el uso real. La salida no se puede reconstruir entera
  (el detalle no guarda `script_usage` ni `matching_fragment`): se cuenta lo que hay y se escala por la razón real /
  reconstruida de las llamadas del caché. Si la llamada está en el caché, se usa el uso real.
- **Reranker:** segundos por par medidos en el estudio (`rerank_*_manifest.json`) × pares por consulta (K).
"""

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean, median

from tprag.generate.llm import ChatModel, _schema
from tprag.generate.prompt import SYSTEM, user_message

Count = Callable[[str], int]
Lookup = Callable[[str], dict | None]  # clave del caché del LLM → respuesta guardada (con "usage") o None


@dataclass
class _Ctx:  # lo que `user_message` lee de un `TargetContext`
    module: str
    status: str | None
    chunks: list[dict]


def cache_key(system: str, user: str) -> str:
    """La misma clave que `CachedLLM._key` (modelo aparte)."""
    return hashlib.sha256("\x00".join((system, user, _schema())).encode()).hexdigest()


def rebuild_user(record: dict, code: str) -> str | None:
    """Mensaje que recibió el LLM en un ítem de `eval_generation.py`, o None si no se llamó al LLM o el detalle es de
    una corrida vieja que no guardaba el texto de los fragmentos."""
    if not record.get("llm_called"):
        return None
    status = {e["module"]: e["status"] for e in record["report"]}
    passed = [c for c in record["contexts"] if c["passed"]]
    if not passed or any("text" not in ch for c in passed for ch in c["chunks"]):
        return None
    return user_message(code, [_Ctx(c["module"], status.get(c["module"]), c["chunks"]) for c in passed])


def rebuild_output(record: dict) -> str:
    """Lo que se puede reconstruir de la respuesta: decisiones del reporte + reemplazos (sin `script_usage` ni
    `matching_fragment`, que el detalle no guarda)."""
    decisions = [{k: e.get(k) for k in ("module", "decision", "replacement_package_llm", "reason", "sources")}
                 for e in record["report"] if "decision" in e]
    return json.dumps({"imports": decisions, "edits": record["edits"]}, ensure_ascii=False)


def calibrate(pairs: list[dict]) -> dict:
    """Con las llamadas que están en el caché ({real_in, counted_in, real_out, rebuilt_out}): sobrecosto fijo de la
    entrada (mediana de real − contado; en la práctica es constante), razón media de la salida y el error relativo
    medio que deja esa razón sobre las mismas llamadas (dentro de la muestra: es optimista)."""
    if not pairs:
        raise ValueError("ninguna llamada de las corridas está en el caché del LLM: no hay con qué calibrar")
    overhead = [p["real_in"] - p["counted_in"] for p in pairs]
    ratios = [p["real_out"] / p["rebuilt_out"] for p in pairs if p["rebuilt_out"]]
    ratio = mean(ratios)
    errors = [abs(p["rebuilt_out"] * ratio - p["real_out"]) / p["real_out"] for p in pairs if p["real_out"]]
    return {"n": len(pairs), "input_overhead": median(overhead), "input_overhead_range": [min(overhead), max(overhead)],
            "output_ratio": ratio, "output_ratio_range": [min(ratios), max(ratios)],
            "output_mean_abs_error": mean(errors)}


def call_usage(record: dict, code: str, count: Count, lookup: Lookup, calibration: dict | None = None) -> dict | None:
    """Tokens de la llamada al LLM de un ítem: reales si está en el caché; si no, contados + calibración. Separa la
    entrada en instrucciones, script y fragmentos (contexto). None si no se llamó, no se puede reconstruir o (sin
    `calibration`) no está en el caché."""
    user = rebuild_user(record, code)
    if user is None:
        return None
    system_t, user_t, script_t = count(SYSTEM), count(user), count(code)
    parts = {"system": system_t, "script": script_t, "context": user_t - script_t}
    cached = lookup(cache_key(SYSTEM, user))
    if cached is not None:
        real = cached["usage"]
        return {"exact": True, "input": real["input_tokens"], "output": real["output_tokens"], "parts": parts,
                "counted_in": system_t + user_t, "rebuilt_out": count(rebuild_output(record))}
    if calibration is None:  # primera pasada: solo las del caché, para calibrar
        return None
    rebuilt = count(rebuild_output(record))
    return {"exact": False, "input": system_t + user_t + round(calibration["input_overhead"]),
            "output": round(rebuilt * calibration["output_ratio"]), "parts": parts}


def _pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, math.ceil(q * len(s)) - 1)]


def summarize_calls(usages: list[dict], model: ChatModel) -> dict:
    """Totales y distribución por llamada (USD con los precios de `model`)."""
    if not usages:
        return {"calls": 0}
    usd = [model.usd(u["input"], u["output"]) for u in usages]
    inputs, outputs = [u["input"] for u in usages], [u["output"] for u in usages]
    parts = {k: mean(u["parts"][k] for u in usages) for k in ("system", "script", "context")}
    return {"calls": len(usages), "exact": sum(u["exact"] for u in usages),
            "input_tokens": sum(inputs), "output_tokens": sum(outputs), "usd": round(sum(usd), 6),
            "per_call": {"input_mean": mean(inputs), "input_p50": _pct(inputs, 0.5), "input_p95": _pct(inputs, 0.95),
                         "output_mean": mean(outputs), "output_p50": _pct(outputs, 0.5),
                         "output_p95": _pct(outputs, 0.95), "usd_mean": mean(usd), "usd_p95": _pct(usd, 0.95)},
            "input_parts_mean": parts,
            "output_share_of_usd": sum(model.usd(0, o) for o in outputs) / sum(usd) if sum(usd) else 0.0}


def per_thousand(usd_per_call: float, call_rate: float) -> float:
    """USD cada 1.000 scripts con algún import problemático, si se llama al LLM en una fracción `call_rate`."""
    return round(1000 * usd_per_call * call_rate, 4)


def index_row(report: dict) -> dict:
    """Fila de costo de un reporte de `build_index.py`."""
    paid = report.get("paid", {})
    local = not report.get("api_model")
    return {"collection": report["collection"], "model": report["model"], "provider": "local" if local else "API",
            "use_cases": report["use_cases"], "tokens_total": report["tokens_total"],
            "embedded_tokens": paid.get("tokens", 0), "embedded_texts": paid.get("texts", 0),
            "cache_hits": report.get("cache_hits", 0), "usd": paid.get("usd", 0.0), "seconds": report["seconds"],
            "texts_per_second": round(paid["texts"] / report["seconds"], 1) if paid.get("texts") else None,
            "usd_full_corpus": round(report["tokens_total"] / 1e6 * report.get("usd_per_m_tokens", 0.0), 6)}


def reranker_latency(manifest: dict, ks: tuple[int, ...] = (10, 20, 30, 50)) -> list[dict]:
    """Segundos por consulta de cada reranker según cuántos candidatos reordena (un par por candidato, de a uno)."""
    rows = []
    for rid, r in sorted(manifest["rerankers"].items()):
        t = r["timing"]
        for k in ks:
            rows.append({"reranker": rid, "label": r["label"], "k": k, "seconds_per_pair": t["seconds_per_pair"],
                         "seconds_per_query": round(t["seconds_per_pair"] * k, 2), "peak_memory_mb": t["peak_mps_mb"],
                         "dtype": t["dtype"], "device": "mps" if manifest["environment"].get("mps") else "cpu",
                         "machine": manifest["environment"]["platform"]})
    return rows
