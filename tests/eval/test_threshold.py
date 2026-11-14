"""Barrido del umbral (eval/threshold.py), sobre registros armados a mano: sin API ni índice."""

import pytest

from tprag.eval.threshold import (BAD, CORRECT_ABSTENTION, GOOD, MISSED, NR, PRECISION, RETRIEVAL, UTILITY,
                                  WRONG_CHANGE, candidates, choose, compare, counts, min_threshold, outcome,
                                  sensitivity, sweep, targets, utility)
from tprag.generate.schema import NO_REPLACEMENT, REPLACED

ORIGINAL = "import viejo\n"


def record(item, targets_, changed=True, subset="synthetic"):
    return {"id": item, "subset": subset, "code": "import nuevo\n" if changed else ORIGINAL, "targets": targets_}


def target(expected, score, decision=REPLACED, success=True, passed=True):
    return {"expected": expected, "retrieval_score": score, "passed_threshold": passed, "decision": decision,
            "success": success}


def flatten(records):
    return targets(records, {r["id"]: ORIGINAL for r in records})


def one(expected, score, changed=True, **kw):
    return flatten([record("x", [target(expected, score, **kw)], changed)])[0]


def test_outcome_positive():
    assert outcome(one(REPLACED, 0.6), 0.5) == GOOD
    assert outcome(one(REPLACED, 0.6, success=False), 0.5) == BAD
    assert outcome(one(REPLACED, 0.6, changed=False, decision=NO_REPLACEMENT, success=False), 0.5) == MISSED
    assert outcome(one(REPLACED, 0.4, changed=False), 0.5) == MISSED  # se abstuvo por umbral


def test_replacement_not_applied_is_missed_not_bad():
    # El LLM decidió reemplazar, pero los reemplazos no se aplicaron: el script quedó igual (opción A, metricas.md §4.2)
    t = one(REPLACED, 0.6, changed=False, success=False)
    assert outcome(t, 0.5) == MISSED


def test_outcome_negative():
    assert outcome(one(NO_REPLACEMENT, 0.6, changed=False, decision=NO_REPLACEMENT), 0.5) == CORRECT_ABSTENTION
    assert outcome(one(NO_REPLACEMENT, 0.6, success=False), 0.5) == WRONG_CHANGE
    assert outcome(one(NO_REPLACEMENT, 0.6, success=False), 0.7) == CORRECT_ABSTENTION  # el umbral lo frena
    # decidió reemplazar pero no se aplicó: no hubo cambio
    assert outcome(one(NO_REPLACEMENT, 0.6, changed=False, success=False), 0.5) == CORRECT_ABSTENTION


def test_threshold_equal_to_score_passes():
    assert outcome(one(REPLACED, 0.5), 0.5) == GOOD


def test_no_context_is_abstention():
    assert outcome(one(REPLACED, None, changed=False, passed=False), float("-inf")) == MISSED


def test_below_run_threshold_is_not_evaluable():
    t = one(REPLACED, 0.3, changed=False, passed=False, decision=NO_REPLACEMENT, success=False)
    assert outcome(t, 0.4) == MISSED
    with pytest.raises(ValueError):
        outcome(t, 0.2)


def test_min_threshold_and_candidates():
    permissive = flatten([record("a", [target(REPLACED, 0.3)]), record("b", [target(REPLACED, 0.6)])])
    assert min_threshold(permissive) == float("-inf")
    assert candidates(permissive) == [float("-inf"), 0.3, 0.6, float("inf")]
    partial = flatten([record("a", [target(REPLACED, 0.3, passed=False)], changed=False),
                       record("b", [target(REPLACED, 0.6)])])
    assert min_threshold(partial) == 0.6
    assert candidates(partial) == [0.6, float("inf")]


def test_counts_rates_calls_and_misleading_report():
    ts = flatten([
        record("a", [target(REPLACED, 0.8)]),
        record("b", [target(REPLACED, 0.5, success=False)]),
        record("c", [target(NO_REPLACEMENT, 0.6, success=False)]),
        record("d", [target(NO_REPLACEMENT, 0.3, decision=NO_REPLACEMENT)], changed=False),
        record("e", [target(REPLACED, 0.7, success=False)], changed=False),  # dijo "reemplazado", quedó igual
    ])
    row = counts(ts, 0.55)
    assert (row[GOOD], row[BAD], row[MISSED], row[WRONG_CHANGE], row[CORRECT_ABSTENTION]) == (1, 0, 2, 1, 1)
    assert row["misleading_report"] == 1
    assert row["success_pos"] == 1 / 3 and row["negative_rejection"] == 0.5
    assert row["llm_calls"] == 3 and row["llm_calls_avoided"] == 0.4


def test_multi_target_script_counts_one_call():
    ts = flatten([record("a", [target(REPLACED, 0.8), target(REPLACED, 0.4)])])
    assert counts(ts, 0.5)["llm_calls"] == 1
    assert counts(ts, 0.9)["llm_calls"] == 0


def test_utility_penalizes_errors():
    row = {GOOD: 10, BAD: 2, WRONG_CHANGE: 1}
    assert utility(row, 2) == 10 - 2 * 3
    assert utility(row, 1) == 7


def test_choose_prefers_fewer_errors_with_higher_lambda():
    # Umbral bajo: 3 buenas y 2 cambios indebidos; umbral alto: 2 buenas y ninguno.
    ts = flatten([
        record("a", [target(REPLACED, 0.9)]), record("b", [target(REPLACED, 0.8)]),
        record("c", [target(REPLACED, 0.5)]),
        record("d", [target(NO_REPLACEMENT, 0.6, success=False)]),
        record("e", [target(NO_REPLACEMENT, 0.55, success=False)]),
    ])
    rows = sweep(ts)
    assert choose(rows, 0.4)["threshold"] == 0.5  # 3 − 0,4 × 2 = 2,2 > 2
    assert choose(rows, 2)["threshold"] == 0.8  # 2 > 3 − 2 × 2
    assert sensitivity(rows, (0.4, 2.0)) == {0.4: 0.5, 2.0: 0.8}


def test_choose_tie_takes_highest_threshold():
    ts = flatten([record("a", [target(REPLACED, 0.9)]),
                  record("b", [target(REPLACED, 0.5, decision=NO_REPLACEMENT, success=False)], changed=False)])
    rows = sweep(ts)  # con 0,5 y con 0,9 da 1 buena y 0 errores
    assert choose(rows)["threshold"] == 0.9


# Barrido para los criterios (umbral → buenas, malas, indebidos):
#   −∞ / 0,3: 3 buenas, 1 mala, 1 indebido (precisión 0,6, NR 0,5, pasan 4/4 positivos)
#   0,5:      3 buenas, 0 malas, 1 indebido (precisión 0,75, NR 0,5, 3/4)
#   0,6:      2 buenas, 0, 1                (precisión 0,67, NR 0,5, 2/4)
#   0,7:      2 buenas, 0, 0                (precisión 1, NR 1, 2/4)
#   0,9:      1 buena                       (1/4);  +∞: nada
def criteria_rows():
    return sweep(flatten([
        record("a", [target(REPLACED, 0.9)]), record("b", [target(REPLACED, 0.7)]),
        record("c", [target(REPLACED, 0.5)]), record("d", [target(REPLACED, 0.3, success=False)]),
        record("e", [target(NO_REPLACEMENT, 0.6, success=False)]),
        record("f", [target(NO_REPLACEMENT, 0.2, decision=NO_REPLACEMENT)], changed=False),
    ]))


def test_counts_precision_and_passed_pos():
    row = next(r for r in criteria_rows() if r["threshold"] == 0.5)
    assert row["precision"] == 0.75 and row["passed_pos"] == 0.75
    assert next(r for r in criteria_rows() if r["threshold"] == float("inf"))["precision"] is None


def test_choose_utility_is_the_default_and_unchanged():
    rows = criteria_rows()
    assert choose(rows) is choose(rows, 2.0) is choose(rows, 2.0, UTILITY)
    assert choose(rows)["threshold"] == 0.7  # 2 − 0 = 2 > 3 − 2 × 1 = 1


def test_choose_precision():
    rows = criteria_rows()
    assert choose(rows, 0.7, PRECISION)["threshold"] == 0.5  # 3 buenas con precisión 0,75
    assert choose(rows, 0.8, PRECISION)["threshold"] == 0.7  # 0,5 ya no alcanza: 2 buenas, precisión 1


def test_choose_negative_rejection():
    rows = criteria_rows()
    assert choose(rows, 0.5, NR)["threshold"] == 0.5  # NR 0,5 en todos los bajos: gana 0,5 por empate en 3 buenas
    assert choose(rows, 0.85, NR)["threshold"] == 0.7


def test_choose_retrieval_rule():
    rows = criteria_rows()
    assert choose(rows, 0.95, RETRIEVAL)["threshold"] == 0.3  # el más alto que deja pasar los 4 positivos
    assert choose(rows, 0.5, RETRIEVAL)["threshold"] == 0.7


def test_nobody_meets_the_condition_abstains_always():
    rows = sweep(flatten([record("a", [target(REPLACED, 0.8, success=False)])]))  # solo una corrección mala
    assert choose(rows, 0.8, PRECISION)["threshold"] == float("inf")


def test_unknown_criterion():
    with pytest.raises(ValueError):
        choose(criteria_rows(), 0.5, "otro")


def test_compare_uses_defaults_unless_given():
    rows = criteria_rows()
    assert {c: r["threshold"] for c, r in compare(rows).items()} == {
        UTILITY: 0.7, PRECISION: 0.7, NR: 0.7, RETRIEVAL: 0.3}
    assert compare(rows, {PRECISION: 0.7})[PRECISION]["threshold"] == 0.5
