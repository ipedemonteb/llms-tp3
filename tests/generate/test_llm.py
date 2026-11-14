"""Caché del LLM del generador con la llamada a la API reemplazada: sin red ni API."""

import pytest

from tprag.generate.llm import CHAT_MODELS, MAX_OUTPUT_TOKENS, CachedLLM, TruncatedOutput

OK = {"output": {"imports": [], "edits": []}, "max_tokens": MAX_OUTPUT_TOKENS,
      "usage": {"input_tokens": 100, "output_tokens": 20}}
CUT = {"output": None, "error": "length", "raw": '{"imports": [{"reason": "\\\\\\\\', "max_tokens": MAX_OUTPUT_TOKENS,
       "usage": {"input_tokens": 100, "output_tokens": MAX_OUTPUT_TOKENS}}


def llm_with(tmp_path, response):
    llm = CachedLLM(CHAT_MODELS["gpt-4o-mini"], path=tmp_path / "llm.sqlite")
    calls = []
    llm._call = lambda system, user: calls.append((system, user)) or response
    return llm, calls


def test_respuesta_normal_se_guarda_y_no_se_vuelve_a_pagar(tmp_path):
    llm, calls = llm_with(tmp_path, OK)
    assert llm("s", "u").imports == [] and llm("s", "u").edits == []
    assert len(calls) == 1 and llm.misses == 1 and llm.hits == 1 and llm.output_tokens == 20


def test_respuesta_cortada_se_paga_una_vez_y_se_levanta_tambien_desde_el_cache(tmp_path):
    llm, calls = llm_with(tmp_path, CUT)
    with pytest.raises(TruncatedOutput) as first:
        llm("s", "u")
    assert first.value.raw == CUT["raw"]
    assert llm.misses == 1 and llm.output_tokens == MAX_OUTPUT_TOKENS  # se cuenta lo pagado aunque no sirva
    with pytest.raises(TruncatedOutput):
        llm("s", "u")
    assert len(calls) == 1 and llm.hits == 1 and llm.missing([("s", "u")]) == []
