"""Barrido del umbral de "sin reemplazo confiable" sobre una corrida del generador (metricas.md §4, §4.2), sin API.

Parte del detalle de `eval_generation.py` (`.items.jsonl`): cada target trae su puntaje de confianza
(`retrieval_score`: coseno del mejor chunk o, con reranker, máximo del puntaje de Qwen) y lo que respondió el LLM. Con
un umbral más alto que el de la corrida, los targets con puntaje menor pasan a "abstención por umbral" y el resto
conserva su respuesta: el LLM no ve el umbral, así que un target que pasa recibe el mismo prompt con cualquier umbral.
Aproximación: en los scripts con varios targets se toma la decisión por target de la corrida (metricas.md §4).

Un umbral más bajo que el de la corrida no se puede evaluar (esos targets no llegaron al LLM): por eso el estudio se
hace sobre una corrida permisiva.
"""

from collections import Counter
from dataclasses import dataclass

from tprag.generate.schema import REPLACED

GOOD, BAD, MISSED, WRONG_CHANGE, CORRECT_ABSTENTION = "buena", "mala", "perdida", "cambio_indebido", "abstencion_ok"
OUTCOMES = (GOOD, BAD, MISSED, WRONG_CHANGE, CORRECT_ABSTENTION)
LAMBDA = 2.0
SENSITIVITY = (1.0, 2.0, 3.0)

# Criterios para elegir el umbral (metricas.md §4.2). El principal es `utility`; los otros se agregaron después de ver
# las curvas de dev (2026-11-14) para comparar. Cada uno tiene un parámetro: λ, precisión mínima, Negative Rejection
# mínima o fracción de positivos que pasan.
UTILITY, PRECISION, NR, RETRIEVAL = "utility", "precision", "nr", "retrieval"
CRITERIA = (UTILITY, PRECISION, NR, RETRIEVAL)
DEFAULT_PARAMS = {UTILITY: LAMBDA, PRECISION: 0.8, NR: 0.85, RETRIEVAL: 0.95}


@dataclass(frozen=True)
class Target:
    item: str
    subset: str
    positive: bool  # había que reemplazar
    score: float | None  # None: el import no tuvo contexto (no se pudo puntuar); se trata como abstención
    passed_run: bool  # pasó el umbral en la corrida (llegó al LLM)
    decision: str | None
    success: bool  # éxito de punta a punta del nivel 1 (eval/generation.py)
    changed: bool  # el script que recibe el usuario es distinto del original


def targets(records: list[dict], originals: dict[str, str]) -> list[Target]:
    """Aplana el `.items.jsonl` a un registro por target. `originals`: id del ítem → código original (del dataset),
    para saber si el script cambió (el detalle guarda solo el corregido)."""
    return [Target(item=r["id"], subset=r["subset"], positive=t["expected"] == REPLACED, score=t["retrieval_score"],
                   passed_run=t["passed_threshold"], decision=t["decision"], success=t["success"],
                   changed=r["code"] != originals[r["id"]])
            for r in records for t in r["targets"]]


def outcome(t: Target, threshold: float) -> str:
    """Resultado del target con ese umbral (metricas.md §4.2). Lo que se penaliza es el daño: entregar un script
    **cambiado** que está mal. Si el LLM decidió reemplazar pero sus reemplazos no se aplicaron y el script quedó igual,
    cuenta como abstención (perdida en positivos, correcta en negativos); el reporte engañoso se cuenta aparte
    (`misleading_report` en `counts`)."""
    if t.score is None or t.score < threshold:
        return MISSED if t.positive else CORRECT_ABSTENTION
    if not t.passed_run:
        raise ValueError(f"{t.item}: umbral {threshold} por debajo del de la corrida (no llegó al LLM)")
    if t.positive:
        if t.success:
            return GOOD
        return BAD if t.changed else MISSED
    return WRONG_CHANGE if t.changed else CORRECT_ABSTENTION


def min_threshold(ts: list[Target]) -> float:
    """El umbral más bajo evaluable: el menor puntaje que llegó al LLM; -inf si todos llegaron (corrida permisiva)."""
    scored = [t for t in ts if t.score is not None]
    if all(t.passed_run for t in scored):
        return float("-inf")
    return min(t.score for t in scored if t.passed_run)


def candidates(ts: list[Target]) -> list[float]:
    """Umbrales a probar: cada puntaje distinto evaluable (con umbral = s, el target de puntaje s pasa) y +inf
    (abstenerse siempre). Entre dos puntajes consecutivos el resultado no cambia, así que alcanza con estos."""
    low = min_threshold(ts)
    scores = sorted({t.score for t in ts if t.score is not None and t.score >= low})
    return ([low] if low == float("-inf") else []) + scores + [float("inf")]


def counts(ts: list[Target], threshold: float) -> dict:
    c = Counter(outcome(t, threshold) for t in ts)
    pos, neg = sum(t.positive for t in ts), sum(not t.positive for t in ts)
    passing = {t.item for t in ts if t.score is not None and t.score >= threshold}
    items = {t.item for t in ts}
    corrections = c[GOOD] + c[BAD] + c[WRONG_CHANGE]  # scripts entregados cambiados
    return {
        "threshold": threshold,
        **{o: c[o] for o in OUTCOMES},
        "positives": pos, "negatives": neg,
        "success_pos": c[GOOD] / pos if pos else None,  # éxito en positivos
        "precision": c[GOOD] / corrections if corrections else None,  # de lo que se cambió, cuánto estaba bien
        # positivos que pasan el umbral (llegan al LLM): lo que mira la regla de calibración por retrieval
        "passed_pos": sum(t.positive and t.score is not None and t.score >= threshold for t in ts) / pos if pos else None,
        "bad_pos": c[BAD] / pos if pos else None,
        "missed_pos": c[MISSED] / pos if pos else None,  # abstención indebida (correcciones perdidas)
        "negative_rejection": c[CORRECT_ABSTENTION] / neg if neg else None,
        # el reporte dice "reemplazado" pero el script quedó igual (no daña el código, pero informa algo falso)
        "misleading_report": sum(t.decision == REPLACED and not t.changed and t.score is not None
                                 and t.score >= threshold for t in ts),
        "llm_calls": len(passing),  # scripts con al menos un target que pasa
        "llm_calls_avoided": 1 - len(passing) / len(items) if items else None,
    }


def utility(row: dict, lam: float = LAMBDA) -> float:
    """puntaje = correcciones buenas − λ × (correcciones malas + cambios indebidos); abstenerse suma 0."""
    return row[GOOD] - lam * (row[BAD] + row[WRONG_CHANGE])


def sweep(ts: list[Target], thresholds: list[float] | None = None) -> list[dict]:
    rows = [counts(ts, th) for th in (thresholds if thresholds is not None else candidates(ts))]
    for row in rows:
        row["utility"] = {lam: utility(row, lam) for lam in SENSITIVITY}
    return rows


def choose(rows: list[dict], param: float | None = None, criterion: str = UTILITY) -> dict:
    """La fila elegida con el criterio; en todos, empate → el umbral más alto (menos llamadas, más conservador).

    - `utility`: mayor puntaje con λ = `param`.
    - `precision`: más correcciones buenas con precisión ≥ `param`.
    - `nr`: más correcciones buenas con Negative Rejection ≥ `param`.
    - `retrieval`: el umbral más alto que deja pasar ≥ `param` de los positivos, sin mirar al LLM (la regla con la
      que se calibraron los umbrales en plan_cierre.md §8).

    Con `precision` y `nr`, una fila sin cambios (o sin negativos) cumple: si ningún umbral cumple, queda abstenerse
    siempre (+inf), que no tiene errores.
    """
    value = DEFAULT_PARAMS[criterion] if param is None else param
    if criterion == UTILITY:
        return max(rows, key=lambda r: (utility(r, value), r["threshold"]))
    if criterion == RETRIEVAL:
        ok = [r for r in rows if r["passed_pos"] is not None and r["passed_pos"] >= value]
        return max(ok, key=lambda r: r["threshold"]) if ok else rows[0]
    field = {PRECISION: "precision", NR: "negative_rejection"}.get(criterion)
    if field is None:
        raise ValueError(f"criterio desconocido: {criterion} (opciones: {', '.join(CRITERIA)})")
    ok = [r for r in rows if r[field] is None or r[field] >= value]
    return max(ok, key=lambda r: (r[GOOD], r["threshold"]))


def describe(criterion: str, param: float | None = None) -> str:
    value = DEFAULT_PARAMS[criterion] if param is None else param
    return {UTILITY: f"puntaje con λ = {value:g}", PRECISION: f"máximas buenas con precisión ≥ {value:g}",
            NR: f"máximas buenas con Negative Rejection ≥ {value:g}",
            RETRIEVAL: f"pasa ≥ {value:.0%} de los positivos (regla de retrieval)"}[criterion]


def compare(rows: list[dict], params: dict[str, float] | None = None) -> dict[str, dict]:
    """La fila que elige cada criterio (con `params` o los de por defecto)."""
    params = DEFAULT_PARAMS | (params or {})
    return {c: choose(rows, params[c], c) for c in CRITERIA}


def sensitivity(rows: list[dict], lams: tuple[float, ...] = SENSITIVITY) -> dict[float, float]:
    """Umbral elegido con cada λ: si casi no cambia, el valor exacto de λ no importa."""
    return {lam: choose(rows, lam)["threshold"] for lam in lams}


def by_subset(ts: list[Target], threshold: float) -> dict[str, dict]:
    return {s: counts([t for t in ts if t.subset == s], threshold) for s in sorted({t.subset for t in ts})}

