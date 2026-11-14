"""Noise Robustness (eval/noise.py): sin API ni índice."""

import pytest

from tprag.eval.noise import CLEAN, NO_ANSWER, NOISY_FIRST, NOISY_LATER, group, summarize, target_groups
from tprag.generate.schema import NO_REPLACEMENT, REPLACED


def test_group():
    assert group(["pypdf", "pypdf"], ["pypdf"]) == CLEAN
    assert group(["pypdf", "pymupdf"], ["pypdf"]) == NOISY_FIRST
    assert group(["pymupdf", "pypdf"], ["pypdf"]) == NOISY_LATER
    assert group(["pymupdf"], ["pypdf"]) == NO_ANSWER
    assert group(["Google_Auth"], ["google-auth"]) == CLEAN  # nombres normalizados


def record(item, packages, success=True, expected=REPLACED, passed=True, score=0.6):
    return {"id": item, "subset": "synthetic",
            "targets": [{"module": "m", "expected": expected, "passed_threshold": passed, "success": success,
                         "retrieval_score": score}],
            "contexts": [{"module": "m", "chunks": [{"package": p} for p in packages]}]}


def items(*ids):
    return {i: {"targets": [{"module": "m", "acceptable_packages": ["pypdf"]}]} for i in ids}


def test_target_groups_filters_and_noise_fraction():
    records = [record("a", ["pypdf", "x"]), record("b", ["x"], passed=False),
               record("c", ["pypdf"], expected=NO_REPLACEMENT), record("d", ["pypdf"], score=0.3)]
    rows = target_groups(records, items("a", "b", "c", "d"))
    assert [r["item"] for r in rows] == ["a", "d"]  # b no llegó al LLM, c es negativo
    assert rows[0]["group"] == NOISY_FIRST and rows[0]["noise_fraction"] == 0.5
    assert [r["item"] for r in target_groups(records, items("a", "b", "c", "d"), threshold=0.5)] == ["a"]


def test_summarize_noise_robustness():
    records = [record("a", ["pypdf"]), record("b", ["pypdf"], success=False),
               record("c", ["pypdf", "x"]), record("d", ["x", "pypdf"], success=False),
               record("e", ["x"], success=False)]
    out = summarize(target_groups(records, items(*"abcde")))
    assert out[CLEAN] == {"targets": 2, "success": 0.5}
    assert out["noisy"] == {"targets": 2, "success": 0.5}
    assert out[NO_ANSWER]["targets"] == 1
    assert out["noise_robustness"] == 1.0


def test_noise_robustness_undefined_without_clean_success():
    out = summarize(target_groups([record("a", ["pypdf", "x"])], items("a")))
    assert out["noise_robustness"] is None


def test_target_missing_from_dataset():
    records = [record("a", ["pypdf"])]
    renamed = {"a": {"targets": [{"module": "otro", "acceptable_packages": ["pypdf"]}]}}
    with pytest.raises(KeyError):
        target_groups(records, renamed)
    assert target_groups(records, renamed, strict=False) == []
