"""Análisis de costo (tprag.eval.cost): reconstrucción del prompt, calibración y cuentas."""

import pytest

from tprag.eval.cost import (cache_key, calibrate, call_usage, index_row, per_thousand, rebuild_user,
                             reranker_latency, summarize_calls)
from tprag.generate.context import TargetContext
from tprag.generate.llm import CHAT_MODELS, CachedLLM
from tprag.generate.prompt import SYSTEM, user_message

LLM = CHAT_MODELS["gpt-4o-mini"]
CODE = "import fuzzywuzzy\nfuzzywuzzy.fuzz.ratio('a', 'b')\n"
CHUNK = {"header": "rapidfuzz > fuzz > ratio", "package": "rapidfuzz", "id": "c1", "n_tokens": 5,
         "text": "from rapidfuzz import fuzz\nfuzz.ratio('a', 'b')\n"}


def record(**over) -> dict:
    """Ítem de eval_generation.py con lo que lee el análisis de costo."""
    base = {"id": "x", "llm_called": True,
            "report": [{"module": "fuzzywuzzy", "status": "deprecated", "decision": "reemplazado",
                        "replacement_package_llm": "rapidfuzz", "reason": "r", "sources": ["h"]}],
            "contexts": [{"module": "fuzzywuzzy", "score": 0.5, "passed": True, "chunks": [CHUNK]},
                         {"module": "otro", "score": 0.1, "passed": False, "chunks": []}],
            "edits": [{"antes": "import fuzzywuzzy\n", "despues": "from rapidfuzz import fuzz\n"}]}
    return base | over


def words(text: str) -> int:
    return len(text.split())


def test_rebuilt_message_is_the_one_the_generator_sends():
    ctx = TargetContext("fuzzywuzzy", "deprecated", "q", 0.5, True, [CHUNK])
    assert rebuild_user(record(), CODE) == user_message(CODE, [ctx])


def test_cache_key_matches_the_llm_cache(tmp_path):
    llm = CachedLLM(LLM, tmp_path / "llm.sqlite")
    assert cache_key(SYSTEM, "mensaje") == llm._key(SYSTEM, "mensaje")


@pytest.mark.parametrize("over", [{"llm_called": False},
                                  {"contexts": [{"module": "fuzzywuzzy", "passed": True,
                                                 "chunks": [{k: v for k, v in CHUNK.items() if k != "text"}]}]}])
def test_rebuild_user_gives_up_without_a_call_or_without_chunk_texts(over):
    assert rebuild_user(record(**over), CODE) is None


def test_cached_calls_use_the_real_usage_and_others_need_the_calibration():
    rec = record()
    key = cache_key(SYSTEM, rebuild_user(rec, CODE))
    cached = {key: {"usage": {"input_tokens": 999, "output_tokens": 77}}}
    exact = call_usage(rec, CODE, words, cached.get)
    assert (exact["exact"], exact["input"], exact["output"]) == (True, 999, 77)
    assert call_usage(rec, CODE, words, {}.get) is None  # sin calibración, solo las del caché

    cal = {"input_overhead": 10, "output_ratio": 2.0}
    est = call_usage(rec, CODE, words, {}.get, cal)
    assert est["exact"] is False
    assert est["input"] == words(SYSTEM) + words(rebuild_user(rec, CODE)) + 10
    assert est["output"] == 2 * exact["rebuilt_out"]
    assert est["parts"] == exact["parts"]
    assert est["parts"]["script"] == words(CODE)


def test_calibrate_takes_the_constant_overhead_and_the_mean_output_ratio():
    pairs = [{"real_in": 110, "counted_in": 100, "real_out": 30, "rebuilt_out": 20},
             {"real_in": 60, "counted_in": 50, "real_out": 10, "rebuilt_out": 10}]
    cal = calibrate(pairs)
    assert (cal["n"], cal["input_overhead"], cal["output_ratio"]) == (2, 10, 1.25)
    assert cal["output_ratio_range"] == [1.0, 1.5]
    assert cal["output_mean_abs_error"] == pytest.approx((abs(25 - 30) / 30 + abs(12.5 - 10) / 10) / 2)
    with pytest.raises(ValueError):
        calibrate([])


def test_summarize_calls_prices_input_and_output_separately():
    usages = [{"exact": True, "input": 1_000_000, "output": 0, "parts": {"system": 1, "script": 2, "context": 3}},
              {"exact": False, "input": 0, "output": 1_000_000, "parts": {"system": 1, "script": 4, "context": 5}}]
    s = summarize_calls(usages, LLM)
    assert (s["calls"], s["exact"]) == (2, 1)
    assert s["usd"] == pytest.approx(0.15 + 0.60)
    assert s["output_share_of_usd"] == pytest.approx(0.60 / 0.75)
    assert s["input_parts_mean"] == {"system": 1, "script": 3, "context": 4}
    assert summarize_calls([], LLM) == {"calls": 0}


def test_per_thousand_scales_by_the_share_of_scripts_that_reach_the_llm():
    assert per_thousand(0.0007, 1.0) == 0.7
    assert per_thousand(0.0007, 0.1) == 0.07


def test_index_row_tells_api_from_local_and_prices_the_full_corpus():
    api = {"collection": "c", "model": "m", "api_model": "openai/m", "usd_per_m_tokens": 0.02, "use_cases": 5,
           "tokens_total": 2_000_000, "paid": {"texts": 10, "tokens": 500, "usd": 0.01}, "cache_hits": 3,
           "seconds": 5.0}
    row = index_row(api)
    assert (row["provider"], row["usd_full_corpus"], row["texts_per_second"]) == ("API", 0.04, 2.0)
    local = index_row(api | {"api_model": "", "usd_per_m_tokens": 0.0, "paid": {"texts": 0, "tokens": 0, "usd": 0}})
    assert (local["provider"], local["usd_full_corpus"], local["texts_per_second"]) == ("local", 0.0, None)


def test_reranker_latency_is_seconds_per_pair_times_k():
    manifest = {"environment": {"platform": "mac", "mps": True},
                "rerankers": {"k1": {"label": "Qwen", "timing": {"seconds_per_pair": 0.5, "peak_mps_mb": 2000,
                                                                 "dtype": "float16"}}}}
    rows = reranker_latency(manifest, ks=(10, 30))
    assert [(r["k"], r["seconds_per_query"]) for r in rows] == [(10, 5.0), (30, 15.0)]
    assert rows[0]["device"] == "mps"
