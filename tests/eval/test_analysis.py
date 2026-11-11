"""Bootstrap por ítems (eval/analysis.py)."""

import numpy as np

from tprag.eval.analysis import auroc, compare, interval, rate, resample_weights


def test_resample_weights_sum_to_n_items_and_are_reproducible():
    w = resample_weights(7, 50, seed=1)
    assert w.shape == (50, 7) and (w.sum(1) == 7).all()
    assert (w == resample_weights(7, 50, seed=1)).all() and (w != resample_weights(7, 50, seed=2)).any()


def test_rate_point_and_interval_cover_the_truth():
    values = np.array([1, 1, 1, 0, 0, 1, 0, 1], dtype=float)
    item_of = np.arange(8)
    point, boot = rate(values, item_of, resample_weights(8, 2000, seed=0))
    lo, hi = interval(boot)
    assert point == 0.625 and lo <= point <= hi and 0 <= lo and hi <= 1


def test_targets_of_the_same_item_are_resampled_together():
    values = np.array([1.0, 1.0, 0.0, 0.0])
    item_of = np.array([0, 0, 1, 1])  # dos ítems con dos targets cada uno: solo hay 0, 0.5 o 1
    _, boot = rate(values, item_of, resample_weights(2, 500, seed=0))
    assert set(np.round(boot, 6)) <= {0.0, 0.5, 1.0}


def test_auroc_matches_the_direct_definition():
    pos, neg = np.array([0.62, 0.55, 0.48]), np.array([0.40, 0.50])
    point, boot = auroc(pos, neg, np.array([0, 1, 2]), np.array([3, 4]), resample_weights(5, 300, seed=0))
    assert abs(point - 5 / 6) < 1e-12 and np.nanmax(boot) <= 1
    perfect, _ = auroc(np.array([0.9]), np.array([0.1]), np.array([0]), np.array([1]), resample_weights(2, 10, 0))
    assert perfect == 1.0


def test_compare_detects_a_clear_difference_and_a_tie():
    weights = resample_weights(40, 2000, seed=3)
    item_of = np.arange(40)
    a, b = np.ones(40), np.zeros(40)
    (pa, ba), (pb, bb) = rate(a, item_of, weights), rate(b, item_of, weights)
    assert compare(pa, ba, pb, bb)["verdict"] == "mejor" and compare(pb, bb, pa, ba)["verdict"] == "peor"
    assert compare(pa, ba, pa, ba) == {"diff": 0.0, "ci": [0.0, 0.0], "verdict": "empate"}
