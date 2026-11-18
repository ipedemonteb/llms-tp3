"""LiveLLM: la llamada va siempre a la API y no se guarda en el caché (sin red: `_call` falso)."""

from tprag.generate.llm import CHAT_MODELS, LiveLLM
from tprag.generate.schema import GeneratorOutput


def test_live_llm_calls_every_time_and_counts_tokens(monkeypatch):
    llm = LiveLLM(CHAT_MODELS["gpt-4.1-mini"])
    calls = []
    monkeypatch.setattr(llm, "_call", lambda s, u: calls.append((s, u)) or {
        "output": {"imports": [], "edits": []}, "usage": {"input_tokens": 10, "output_tokens": 5}})
    assert isinstance(llm("s", "u"), GeneratorOutput) and isinstance(llm("s", "u"), GeneratorOutput)
    assert len(calls) == 2 and (llm.input_tokens, llm.output_tokens) == (20, 10)
    assert llm.missing([("s", "u"), ("s", "u")]) == [("s", "u")] and llm.peek("s", "u") is None
    assert not hasattr(llm, "_db")
