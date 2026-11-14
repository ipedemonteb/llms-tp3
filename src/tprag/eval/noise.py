"""Noise Robustness (metricas.md §3.5), sin API: ¿el generador elige bien cuando el contexto trae chunks de otras
librerías?

Solo positivos que llegaron al LLM y **con un paquete aceptable en el contexto** (si no hay ninguno, es un fallo del
retrieval, no del generador). Cada uno cae en un grupo según los chunks que recibió:

- `limpio`: todos los chunks son de un paquete aceptable;
- `ruido_aceptable_primero`: hay chunks de otros paquetes, pero el primero es aceptable;
- `ruido_aceptable_despues`: el primero es de otro paquete (el caso más difícil).

La métrica es el éxito del nivel 1 en cada grupo; Noise Robustness = éxito con ruido / éxito limpio.

Limitación: los chunks de la biblioteca estándar son todos del paquete `python-stdlib`, así que un chunk de `tomllib`
cuenta como aceptable para un target cuyo reemplazo es `zoneinfo`.
"""

from tprag.generate.schema import REPLACED
from tprag.validate.packages import normalize

CLEAN, NOISY_FIRST, NOISY_LATER, NO_ANSWER = ("limpio", "ruido_aceptable_primero", "ruido_aceptable_despues",
                                              "sin_aceptable_en_contexto")
NOISY = (NOISY_FIRST, NOISY_LATER)


def group(packages: list[str], acceptable: list[str]) -> str:
    """Grupo de un target según los paquetes de los chunks que recibió, en orden."""
    ok = {normalize(a) for a in acceptable}
    hits = [normalize(p) in ok for p in packages]
    if not any(hits):
        return NO_ANSWER
    if all(hits):
        return CLEAN
    return NOISY_FIRST if hits[0] else NOISY_LATER


def target_groups(records: list[dict], items: dict[str, dict], threshold: float | None = None,
                  strict: bool = True) -> list[dict]:
    """Un registro por target positivo que llegó al LLM (y, con `threshold`, cuyo puntaje lo alcanza). `items`: id →
    ítem del dataset (para `acceptable_packages`). Con `strict=False` se saltean los targets que ya no están en el
    dataset (corridas anteriores a un cambio del dataset); por defecto, error."""
    out = []
    for r in records:
        acceptable = {t["module"]: t["acceptable_packages"] for t in items[r["id"]]["targets"]}
        contexts = {c["module"]: c for c in r.get("contexts", [])}
        for t in r["targets"]:
            ctx = contexts.get(t["module"])
            if t["expected"] != REPLACED or not t["passed_threshold"] or ctx is None or not ctx["chunks"]:
                continue
            if threshold is not None and (t["retrieval_score"] is None or t["retrieval_score"] < threshold):
                continue
            if t["module"] not in acceptable:
                if strict:
                    raise KeyError(f"{r['id']}: el target {t['module']} no está en el dataset actual "
                                   "(¿corrida anterior a un cambio del dataset? usar strict=False)")
                continue
            packages = [ch["package"] for ch in ctx["chunks"]]
            out.append({"item": r["id"], "subset": r["subset"], "module": t["module"], "success": t["success"],
                        "group": group(packages, acceptable[t["module"]]),
                        "noise_fraction": sum(normalize(p) not in {normalize(a) for a in acceptable[t["module"]]}
                                              for p in packages) / len(packages)})
    return out


def _rate(rows: list[dict]) -> float | None:
    return sum(r["success"] for r in rows) / len(rows) if rows else None


def summarize(rows: list[dict]) -> dict:
    by = {g: [r for r in rows if r["group"] == g] for g in (CLEAN, NOISY_FIRST, NOISY_LATER, NO_ANSWER)}
    noisy = by[NOISY_FIRST] + by[NOISY_LATER]
    clean, noisy_rate = _rate(by[CLEAN]), _rate(noisy)
    return {
        **{g: {"targets": len(v), "success": _rate(v)} for g, v in by.items()},
        "noisy": {"targets": len(noisy), "success": noisy_rate},
        # éxito con ruido / éxito limpio (1 = el ruido no le afecta); solo con al menos un target en cada grupo
        "noise_robustness": noisy_rate / clean if clean and noisy_rate is not None else None,
    }
