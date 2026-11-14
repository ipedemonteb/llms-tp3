"""Juez de Answer Relevance (eval/judge.py) con un LLM falso: sin API (nunca se crea el cliente de OpenAI)."""

import pytest

from tprag.eval.judge import (JUDGES, PRESERVE_SCORE, JudgeLLM, cosine, describe_prompt, judged_records,
                              preserve_prompt, relevance_a)
from tprag.generate.llm import CHAT_MODELS

ORIGINAL = "from PyPDF2 import PdfReader\nreader = PdfReader('a.pdf')\n"
FIXED = "from pypdf import PdfReader\nreader = PdfReader('a.pdf')\n"


class FakeJudge(JudgeLLM):
    def __init__(self, judge, path, output):
        super().__init__(CHAT_MODELS["gpt-4o-mini"], judge, path)
        self.output, self.calls = output, 0

    def _call(self, system, user):
        self.calls += 1
        return {"output": self.output, "usage": {"input_tokens": 1000, "output_tokens": 100}}


def test_preserve_judge_caches_and_validates(tmp_path):
    out = {"lost_or_changed": [], "preserves": "si", "reason": "hace lo mismo"}
    llm = FakeJudge(JUDGES["b"], tmp_path / "judge.sqlite", out)
    prompt = preserve_prompt(ORIGINAL, FIXED, ["PyPDF2"])
    assert llm.missing([prompt]) == [prompt]
    v = llm(*prompt)
    assert v.preserves == "si" and PRESERVE_SCORE[v.preserves] == 1.0
    llm(*prompt)
    assert llm.calls == 1 and llm.hits == 1 and llm.missing([prompt]) == []
    assert llm.usd == CHAT_MODELS["gpt-4o-mini"].usd(1000, 100)


def test_invalid_output_is_rejected(tmp_path):
    llm = FakeJudge(JUDGES["b"], tmp_path / "judge.sqlite", {"lost_or_changed": [], "preserves": "tal vez",
                                                             "reason": "x"})
    with pytest.raises(ValueError):
        llm(*preserve_prompt(ORIGINAL, FIXED, ["PyPDF2"]))


def test_judges_do_not_share_cache_keys(tmp_path):
    a = FakeJudge(JUDGES["a"], tmp_path / "judge.sqlite", {"descriptions": ["x", "y", "z"]})
    b = FakeJudge(JUDGES["b"], tmp_path / "judge.sqlite", {"lost_or_changed": [], "preserves": "no", "reason": "x"})
    assert a._key("s", "u") != b._key("s", "u")


def test_estimate_uses_judge_output_size(tmp_path):
    llm = FakeJudge(JUDGES["a"], tmp_path / "judge.sqlite", {})
    est = llm.estimate([describe_prompt(FIXED)] * 3)
    assert est["calls"] == 3 and est["output_tokens_estimate"] == 3 * JUDGES["a"].output_tokens_estimate
    assert est["input_tokens"] > 0 and est["usd"] > 0


def test_prompts_do_not_say_the_script_was_generated():
    for system, user in (describe_prompt(FIXED), preserve_prompt(ORIGINAL, FIXED, ["PyPDF2"])):
        text = (system + user).lower()
        assert "llm" not in text and "generad" not in text and "modelo de lenguaje" not in text
    assert "Script original" in preserve_prompt(ORIGINAL, FIXED, ["PyPDF2"])[1]


def test_relevance_a_with_fake_embeddings():
    vectors = {"tarea": [1.0, 0.0], "igual": [1.0, 0.0], "ortogonal": [0.0, 1.0]}
    def embed(texts):
        return [vectors[t] for t in texts]
    assert relevance_a(["igual", "igual", "ortogonal"], "tarea", embed) == pytest.approx(2 / 3)
    assert cosine([1, 1], [2, 2]) == pytest.approx(1.0)


def test_judged_records_only_changed_scripts():
    records = [{"id": "a", "code": FIXED}, {"id": "b", "code": ORIGINAL}, {"id": "c", "code": FIXED}]
    originals = {"a": ORIGINAL, "b": ORIGINAL, "c": ORIGINAL}
    assert [r["id"] for r in judged_records(records, originals)] == ["a", "c"]
    assert [r["id"] for r in judged_records(records, originals, {"c"})] == ["c"]
