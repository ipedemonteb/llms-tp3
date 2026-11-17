"""Diagnóstico de una corrida del generador, sin API: por qué falla cada import positivo no exitoso, qué negativos
terminan cambiados, cuánto reescribe el script completo y la comparación pareada con otras corridas (iteracion_ej4.md
§1, §8.5, §9).

Cada positivo no exitoso con el umbral elegido recibe **una** causa: la primera que cumple, en el orden de `CAUSES`
(primero lo que impide llegar a una corrección: umbral, respuesta cortada, abstenciones; después los chequeos del nivel
1 en el orden de `eval.generation.check_item`). "Con / sin la correcta en el contexto": si alguno de los fragmentos que
recibió el LLM es de un paquete aceptable del dataset (`eval.noise.group`).

La comparación pareada usa el bootstrap del estudio de retrieval (`eval.analysis`: se remuestrean scripts, no targets,
10.000 veces con semilla 2026) con los mismos remuestreos para las dos corridas.
"""

from statistics import median

import numpy as np

from tprag.eval.analysis import compare as paired, interval, rate, resample_weights
from tprag.eval.noise import NO_ANSWER, group
from tprag.eval.threshold import BAD, CORRECT_ABSTENTION, GOOD, WRONG_CHANGE, Target, outcome, targets
from tprag.generate.schema import REPLACED

BELOW, CUT, POSTVAL = "debajo_del_umbral", "respuesta_cortada", "abstencion_postvalidacion"
ABSTAIN_IN, ABSTAIN_OUT = "llm_se_abstiene_con_la_correcta", "llm_se_abstiene_sin_la_correcta"
NOT_APPLIED, NOT_PARSED, HALF = "no_aplica_los_cambios", "no_parsea", "migra_a_medias"
WRONG_IN, WRONG_OUT, OTHER = "elige_mal_con_la_correcta", "elige_mal_sin_la_correcta", "otro"
CAUSES = (BELOW, CUT, POSTVAL, ABSTAIN_IN, ABSTAIN_OUT, NOT_APPLIED, NOT_PARSED, HALF, WRONG_IN, WRONG_OUT, OTHER)
LABELS = {
    BELOW: "Debajo del umbral (no llega al LLM)",
    CUT: "Respuesta cortada por el tope de salida",
    POSTVAL: "Abstención por la post-validación (después del reintento)",
    ABSTAIN_IN: "El LLM se abstiene, con la correcta en el contexto",
    ABSTAIN_OUT: "El LLM se abstiene, sin la correcta en el contexto",
    NOT_APPLIED: "No aplica los cambios (el \"antes\" no está en el script)",
    NOT_PARSED: "El código corregido no parsea",
    HALF: "Migra a medias (queda el import viejo o nombres sin definir)",
    WRONG_IN: "Elige mal la librería, con la correcta en el contexto",
    WRONG_OUT: "Elige mal la librería, sin la correcta en el contexto (retrieval)",
    OTHER: "Otro",
}
N_BOOT, SEED = 10_000, 2026
REWRITE_LINES = 10  # "reescribe de más": más líneas del original cambiadas que esto


def correct_in_context(record: dict, module: str, acceptable: list[str]) -> bool:
    ctx = next((c for c in record.get("contexts", []) if c["module"] == module), None)
    packages = [ch["package"] for ch in ctx["chunks"]] if ctx else []
    return bool(packages) and group(packages, acceptable) != NO_ANSWER


def cause(record: dict, target: dict, acceptable: list[str], threshold: float) -> str:
    """Primera causa por la que este positivo no es una corrección buena con `threshold` (ver `CAUSES`)."""
    if target["retrieval_score"] is None or target["retrieval_score"] < threshold:
        return BELOW
    if record.get("llm_error") == "length":
        return CUT
    retry = record.get("retry")
    if retry and retry.get("remaining"):
        return POSTVAL
    inside = correct_in_context(record, target["module"], acceptable)
    if target["decision"] != REPLACED:
        return ABSTAIN_IN if inside else ABSTAIN_OUT
    if not record["applicable"]:
        return NOT_APPLIED
    if not record["parses"]:
        return NOT_PARSED
    if not target["target_gone"]:
        return HALF
    if not target["package_ok"]:
        return WRONG_IN if inside else WRONG_OUT
    if record["hallucinated_imports"] or record["undefined_names"]:
        return HALF
    return OTHER


def _outcomes(records: list[dict], originals: dict[str, str], threshold: float) -> dict[tuple[str, str], tuple]:
    """(ítem, módulo) → (Target, resultado con el umbral)."""
    keys = [(r["id"], t["module"]) for r in records for t in r["targets"]]
    return {k: (t, outcome(t, threshold)) for k, t in zip(keys, targets(records, originals))}


def diagnose(records: list[dict], items: dict[str, dict], threshold: float) -> dict:
    """Causas de los positivos no exitosos, negativos cambiados y cuánto reescribe, con `threshold`."""
    originals = {i: it["code"] for i, it in items.items()}
    outcomes = _outcomes(records, originals, threshold)
    acceptable = {(i, t["module"]): t["acceptable_packages"] for i, it in items.items() for t in it["targets"]}
    causes, failures, wrong_changes = dict.fromkeys(CAUSES, 0), [], []
    for r in records:
        for t in r["targets"]:
            target, result = outcomes[(r["id"], t["module"])]
            if target.positive and result != GOOD:
                c = cause(r, t, acceptable[(r["id"], t["module"])], threshold)
                causes[c] += 1
                failures.append({"item": r["id"], "module": t["module"], "cause": c,
                                 "replacement_package": t.get("replacement_package")})
            elif not target.positive and result == WRONG_CHANGE:
                wrong_changes.append({"item": r["id"], "module": t["module"],
                                      "replacement_package": t.get("replacement_package")})
    positives = sum(t.positive for t, _ in outcomes.values())
    good = sum(t.positive and o == GOOD for t, o in outcomes.values())
    changed = [r["changed_lines"] for r in records
               if r.get("changed_lines") is not None and r["code"] != originals[r["id"]]]
    rewrite = ({"scripts": len(changed), "median": median(changed), "max": max(changed),
                f"over_{REWRITE_LINES}": sum(n > REWRITE_LINES for n in changed)} if changed else None)
    return {"threshold": threshold, "positives": positives, "good": good, "not_successful": positives - good,
            "causes": causes, "failures": failures, "wrong_changes": wrong_changes, "rewrite": rewrite}


def _values(outcomes: dict, keys: list, item_index: dict[str, int], metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Valores 0/1 por target y su ítem, para `analysis.rate`: éxito en positivos, precisión (buena entre las
    entregadas cambiadas) o Negative Rejection (abstención correcta entre los negativos)."""
    rows = []
    for k in keys:
        target, result = outcomes[k]
        if metric == "success_pos" and target.positive:
            rows.append((result == GOOD, k[0]))
        elif metric == "precision" and result in (GOOD, BAD, WRONG_CHANGE):
            rows.append((result == GOOD, k[0]))
        elif metric == "negative_rejection" and not target.positive:
            rows.append((result == CORRECT_ABSTENTION, k[0]))
    return (np.array([float(v) for v, _ in rows]), np.array([item_index[i] for _, i in rows], dtype=int))


def compare_runs(a: list[dict], b: list[dict], items: dict[str, dict], threshold_a: float, threshold_b: float,
                 n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """A − B por pares (mismos scripts, mismos remuestreos) en éxito en positivos, precisión y Negative Rejection, con
    IC 95%; `empate` si el intervalo incluye el cero (None si una de las dos no tiene casos). Además, cuántos positivos
    gana y pierde A."""
    originals = {i: it["code"] for i, it in items.items()}
    oa, ob = _outcomes(a, originals, threshold_a), _outcomes(b, originals, threshold_b)
    if set(oa) != set(ob):
        raise ValueError("las corridas no tienen los mismos targets")
    keys = sorted(oa)
    item_index = {i: n for n, i in enumerate(sorted({k[0] for k in keys}))}
    weights = resample_weights(len(item_index), n_boot, seed)
    out = {}
    for metric in ("success_pos", "precision", "negative_rejection"):
        (va, ia), (vb, ib) = _values(oa, keys, item_index, metric), _values(ob, keys, item_index, metric)
        if not len(va) or not len(vb):  # p. ej. sin negativos, o una corrida que no cambió ningún script
            out[metric] = None
            continue
        pa, ba = rate(va, ia, weights)
        pb, bb = rate(vb, ib, weights)
        out[metric] = {"a": round(pa, 4), "b": round(pb, 4), "a_ci": interval(ba), "b_ci": interval(bb),
                       **paired(pa, ba, pb, bb)}
    pos = [k for k in keys if oa[k][0].positive]
    out["positives_won"] = sum(oa[k][1] == GOOD and ob[k][1] != GOOD for k in pos)
    out["positives_lost"] = sum(oa[k][1] != GOOD and ob[k][1] == GOOD for k in pos)
    return out


__all__ = ["CAUSES", "LABELS", "Target", "cause", "compare_runs", "diagnose"]
