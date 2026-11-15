"""Gráficos del estudio del generador (tprag.eval.plots_generation) con datos de juguete. Necesitan matplotlib (grupo
`analysis`): sin él, se saltean."""

import pytest

pytest.importorskip("matplotlib")

from tprag.eval import plots_generation as pg  # noqa: E402
from tprag.eval.threshold import Target  # noqa: E402


def row(t, good, bad, wrong=0):
    return {"threshold": t, "buena": good, "mala": bad, "cambio_indebido": wrong, "precision": 0.5, "llm_calls": 10,
            "utility": {"1.0": good - bad, "2.0": good - 2 * bad, "3.0": good - 3 * bad},
            "faithfulness": 0.9, "faithfulness_n": 8, "relevance_b": 0.6, "relevance_b_n": 8}


def analysis(scale):
    sweep = [row("-inf", 10, 8, 1), row(0.1 * scale, 10, 8, 1), row(0.5 * scale, 6, 2), row(0.9 * scale, 1, 0),
             row("inf", 0, 0)]
    return {"sweep": sweep, "chosen": sweep[3] | {"threshold": 0.9 * scale},
            "comparison": {"retrieval": {"row": sweep[1]}}}


def agreement():
    g = {"n": 10, "exact": 0.5, "exact_ci": [0.2, 0.8], "kappa": 0.3, "kappa_ci": [0.0, 0.6]}
    groups = {"total": g | {"n": 20}, "base": g, "rerank": g | {"kappa": None, "kappa_ci": None}}
    mean = {"total": 0.4, "base": 0.3, "rerank": 0.5}
    return {"preservation": {"gpt-4o-mini vs humano": groups, "gpt-4o-mini vs opus": groups},
            "mean_score": {"humano": mean, "gpt-4o-mini": mean, "opus": mean}}


def extra():
    cell = {"targets": 5, "success": 0.4}
    noise = {g: cell for g, _ in pg.NOISE_GROUPS} | {"noise_robustness": 1.1}
    negatives = [{"threshold": t, "by_threshold": b, "by_llm": 0.5 - b / 2, "negatives": 9} for t, b in
                 ((0.1, 0.0), (0.5, 0.4), (0.9, 1.0))]
    return {"negatives": negatives, "noise": noise, "relevance": {ok: {"n": 3, "a": 0.5, "b": 0.6} for ok in
                                                                    (True, False)}}


def test_generation_figures_write_seven_pngs(tmp_path):
    analyses = {"base": analysis(1), "rerank": analysis(10)}
    level1 = {v: {"anterior": row(0, 12, 4), "corregido": row(0, 8, 8)} for v in analyses}
    paths = pg.generation_figures(analyses, level1, agreement(), {v: extra() for v in analyses}, tmp_path)
    assert len(paths) == 7 and all(p.exists() and p.stat().st_size > 0 for p in paths)


def target(score, positive=False, changed=False):
    return Target(item="x", subset="negative", positive=positive, score=score, passed_run=True, decision=None,
                  success=False, changed=changed)


def test_negative_rejection_curve_splits_threshold_and_llm():
    ts = [target(0.2), target(0.5, changed=True), target(0.8), target(None), target(0.9, positive=True)]
    curve = {r["threshold"]: r for r in pg.negative_rejection_curve(ts)}
    # umbral 0.5: abstienen por umbral el de 0.2 y el sin puntaje; el de 0.5 llega y cambia; el de 0.8 llega y no
    assert (curve[0.5]["by_threshold"], curve[0.5]["by_llm"], curve[0.5]["negatives"]) == (0.5, 0.25, 4)
    assert curve[0.9]["by_threshold"] == 1.0 and float("inf") not in curve


def test_relevance_by_success_averages_judged_scripts():
    records = [{"id": "x", "success": True}, {"id": "y", "success": False}, {"id": "z", "success": False}]
    judge = [{"id": "x", "a": {"score": 0.6}, "b": {"score": 1.0}},
             {"id": "y", "a": {"score": 0.4}, "b": {"score": 0.0}}]
    got = pg.relevance_by_success(records, judge)
    assert got[True] == {"n": 1, "a": 0.6, "b": 1.0} and got[False] == {"n": 1, "a": 0.4, "b": 0.0}


def test_previous_success_ignores_only_the_undefined_names_check():
    record = {"applicable": True, "parses": True, "hallucinated_imports": []}
    target = {"expected": "reemplazado", "success": False, "decision_ok": True, "target_gone": True,
              "package_ok": True}
    assert pg.previous_success(record, target)  # falló solo por nombres sin definir
    assert not pg.previous_success(record | {"hallucinated_imports": ["x"]}, target)
    assert pg.previous_success(record, {"expected": "sin_reemplazo", "success": True})
