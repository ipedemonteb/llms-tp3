"""Motor de la interfaz (tprag.ui.engine) con un índice, PyPI, reranker y LLM falsos: sin red, modelos ni API."""

import threading

from langchain_core.documents import Document

from tprag.generate.context import Reranker
from tprag.generate.pipeline import prepare
from tprag.generate.prompt import PROMPTS
from tprag.generate.llm import CHAT_MODELS
from tprag.generate.schema import Edit, GeneratorOutput, ImportDecision, ImportDecisionV2, ScriptOutput, ScriptOutputV2
from tprag.ui import engine as eng
from tprag.validate.packages import normalize
from tprag.validate.pypi import PackageInfo

PYPI = {"PyPDF2": PackageInfo("PyPDF2", [], "2022-12-31", []), "pypdf": PackageInfo("pypdf", [], "2026-09-01", [])}
CODE = "import csv\nfrom PyPDF2 import PdfReader\n\n\ndef pages(path):\n    return PdfReader(path).pages\n"


def fetch(name):
    return {normalize(k): v for k, v in PYPI.items()}.get(normalize(name))


def doc(i, package="pypdf"):
    prefix = f"{package} > Extract text > Ejemplo {i}"
    return Document(id=f"c{i}", page_content=f"{prefix}\n\ntexto {i}",
                    metadata={"package": package, "prefix": prefix, "doc_id": "d", "start": 0, "end": 10,
                              "n_tokens": 300})


class FakeStore:
    def __init__(self, docs, best):
        self.docs, self.best, self.queries = docs, best, 0

    def similarity_search_with_score(self, query, k):
        self.queries += 1
        return [(d, 1 - self.best + 0.01 * i) for i, d in enumerate(self.docs[:k])]


class FakeScorer:
    def __init__(self):
        self.calls = 0

    def score(self, pairs):
        self.calls += 1
        return [1.0 - i for i in range(len(pairs))]


class FakeLLM:
    """Como CachedLLM: estimate / missing / contadores de tokens, y la respuesta de un reemplazo."""

    def __init__(self):
        self.input_tokens = self.output_tokens = 0

    def estimate(self, prompts):
        return {"calls": len(prompts), "input_tokens": 100, "output_tokens_estimate": 700, "usd": 0.001}

    def missing(self, prompts):
        return list(prompts)

    def __call__(self, system, user):
        self.input_tokens += 1000
        self.output_tokens += 200
        return GeneratorOutput(
            imports=[ImportDecision(module="PyPDF2", script_usage="lee un PDF", matching_fragment="pypdf > x",
                                    decision="reemplazado", replacement_package="pypdf", reason="r", sources=["h"])],
            edits=[Edit(antes="from PyPDF2 import PdfReader", despues="from pypdf import PdfReader")])


def test_timed_prepare_returns_what_prepare_returns_plus_stage_times():
    docs = [doc(0), doc(1)]
    reranker = Reranker("k1-rr2", FakeScorer(), 2)
    plain = prepare(CODE, FakeStore(docs, 0.9), fetch, 0.4, 2000, reranker)
    timed, seconds = eng.timed_prepare(CODE, FakeStore(docs, 0.9), fetch, 0.4, 2000, reranker)
    assert timed.user == plain.user and [c.chunks for c in timed.contexts] == [c.chunks for c in plain.contexts]
    assert set(seconds) == {"validador", "retrieval", "reranker", "total"}
    assert all(v >= 0 for v in seconds.values())
    assert seconds["total"] >= seconds["retrieval"] + seconds["reranker"]


def test_timed_wrappers_delegate_and_count_calls():
    timer, store, scorer = eng._Timer(), FakeStore([doc(0)], 0.9), FakeScorer()
    assert eng.TimedStore(store, timer).similarity_search_with_score("q", 1)[0][0].id == "c0"
    assert eng.TimedScorer(scorer, timer).score([("q", "d")]) == [1.0]
    assert (timer.calls, store.queries, scorer.calls) == (2, 1, 1)


def fake_engine(monkeypatch, objects):
    """Engine con objetos falsos; registra en qué hilo se pide cada uno."""
    e = eng.Engine()
    threads = []

    def get(name):
        threads.append(threading.get_ident())
        return objects[name]

    monkeypatch.setattr(e, "_get", get)
    return e, threads


def test_engine_runs_everything_in_one_worker_thread_and_reports_paid_tokens(monkeypatch):
    llm = FakeLLM()
    e, threads = fake_engine(monkeypatch, {"store": FakeStore([doc(0)], 0.9), "pypi": fetch, eng.NOT_ITERATED: llm})
    analysis = e.analyze(CODE, "base", 0.4, 2000)
    assert analysis.llm_estimate["usd"] == 0.001 and analysis.llm_cached is False
    assert analysis.seconds["reranker"] == 0.0
    fix = e.fix(analysis)
    assert "from pypdf import PdfReader" in fix.result.code
    assert fix.paid == {"input_tokens": 1000, "output_tokens": 200, "usd": CHAT_MODELS["gpt-4o-mini"].usd(1000, 200)}
    assert len(set(threads)) == 1 and threads[0] != threading.get_ident()


def test_below_the_threshold_there_is_no_llm_estimate(monkeypatch):
    e, _ = fake_engine(monkeypatch, {"store": FakeStore([doc(0)], 0.1), "pypi": fetch, eng.NOT_ITERATED: FakeLLM()})
    analysis = e.analyze(CODE, "base", 0.4, 2000)
    assert analysis.prepared.prompt is None and analysis.llm_estimate is None


def test_reranker_version_uses_the_variant(monkeypatch):
    scorer = FakeScorer()
    objects = {"store": FakeStore([doc(0), doc(1)], 0.9), "pypi": fetch, eng.NOT_ITERATED: FakeLLM(),
               "k1-rr30": Reranker("k1-rr30", scorer, 30)}
    e, _ = fake_engine(monkeypatch, objects)
    analysis = e.analyze(CODE, "k1-rr30", -100.0, 2000)
    assert scorer.calls == 1 and analysis.prepared.contexts[0].reranker == "k1-rr30"


class FakeScriptLLM(FakeLLM):
    """El LLM del generador iterado: devuelve el script completo; una respuesta por llamada, en orden."""

    def __init__(self, scripts):
        super().__init__()
        self.scripts, self.prompts = list(scripts), []

    def __call__(self, system, user):
        self.input_tokens += 1000
        self.output_tokens += 200
        self.prompts.append((system, user))
        decision = ImportDecisionV2(module="PyPDF2", script_usage="lee un PDF", candidates=[],
                                    matching_fragment="pypdf > x", decision="reemplazado", replacement_package="pypdf",
                                    reason="r", sources=["h"])
        return ScriptOutputV2(imports=[decision], script=self.scripts.pop(0))


GOOD = CODE.replace("from PyPDF2 import", "from pypdf import")
BROKEN = GOOD.replace("PdfReader(path)", "PdfLoader(path)")  # deja un nombre sin definir


def iterated_engine(monkeypatch, scripts):
    llm = FakeScriptLLM(scripts)
    e, _ = fake_engine(monkeypatch, {"store": FakeStore([doc(0)], 0.9), "pypi": fetch, eng.ITERATED: llm})
    return e, llm


def test_generators_match_the_evaluated_systems():
    iterated, plain = eng.GENERATORS[eng.ITERATED], eng.GENERATORS[eng.NOT_ITERATED]
    assert (iterated.llm, iterated.first, iterated.retry, iterated.prompt) == ("gpt-4.1-mini", "script", "script", "v3")
    assert iterated.schema() is ScriptOutputV2
    assert (plain.llm, plain.first, plain.retry, plain.prompt) == ("gpt-4o-mini", "edits", None, "v1")
    assert plain.schema() is None
    assert eng.Generator("gpt-4o-mini", first="script", retry="script").schema() is ScriptOutput


def test_iterated_generator_asks_for_the_full_script_with_the_v3_prompt(monkeypatch):
    e, llm = iterated_engine(monkeypatch, [GOOD])
    analysis = e.analyze(CODE, "base", 0.4, 2000, eng.ITERATED)
    fix = e.fix(analysis)
    assert fix.result.code == GOOD and fix.result.output_format == "script" and fix.result.retry is None
    assert llm.prompts[0][0] == PROMPTS["v3"]["script"] and len(llm.prompts) == 1
    assert fix.paid["usd"] == CHAT_MODELS["gpt-4.1-mini"].usd(1000, 200)


def test_iterated_generator_retries_once_and_counts_both_calls(monkeypatch):
    e, llm = iterated_engine(monkeypatch, [BROKEN, GOOD])
    fix = e.fix(e.analyze(CODE, "base", 0.4, 2000, eng.ITERATED))
    assert fix.result.code == GOOD and len(llm.prompts) == 2
    assert fix.result.retry["remaining"] == [] and "PdfLoader" in fix.result.retry["problems"][0]
    assert fix.paid["input_tokens"] == 2000 and fix.paid["output_tokens"] == 400


def test_iterated_generator_abstains_when_the_retry_is_still_wrong(monkeypatch):
    e, llm = iterated_engine(monkeypatch, [BROKEN, BROKEN])
    fix = e.fix(e.analyze(CODE, "base", 0.4, 2000, eng.ITERATED))
    assert fix.result.code == CODE and len(llm.prompts) == 2
    assert fix.result.retry["remaining"] and all(d["decision"] == "sin_reemplazo_confiable"
                                                 for d in fix.result.report if "decision" in d)
