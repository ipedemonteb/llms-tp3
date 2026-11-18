"""Motor de la interfaz (tprag.ui.engine) con un índice, PyPI, reranker y LLM falsos: sin red, modelos ni API."""

import threading

from langchain_core.documents import Document

from tprag.generate.context import Reranker
from tprag.generate.pipeline import prepare
from tprag.generate.prompt import PROMPTS
from tprag.generate.llm import CHAT_MODELS
from tprag.generate.schema import ImportDecisionV2, ScriptOutputV2
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


class FakeScriptLLM:
    """Como CachedLLM con el esquema del sistema final: estimate / missing / contadores de tokens, y el script
    completo; una respuesta por llamada, en orden."""

    def __init__(self, scripts):
        self.input_tokens = self.output_tokens = 0
        self.scripts, self.prompts = list(scripts), []

    def estimate(self, prompts):
        return {"calls": len(prompts), "input_tokens": 100, "output_tokens_estimate": 700, "usd": 0.001}

    def missing(self, prompts):
        return list(prompts)

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


def system_engine(monkeypatch, scripts, best=0.9):
    llm, scorer = FakeScriptLLM(scripts), FakeScorer()
    objects = {"store": FakeStore([doc(0), doc(1)], best), "pypi": fetch, "llm": llm,
               "reranker": Reranker(eng.system.RERANKER, scorer, 30)}
    e, threads = fake_engine(monkeypatch, objects)
    return e, llm, scorer, threads


def test_system_is_the_chosen_variant():
    s = eng.system
    assert (s.COLLECTION, s.RERANKER, s.BUDGET) == ("s3-1024__e4__r1__cap150", "k1-rr30", 2000)
    assert (s.LLM, s.FORMAT, s.PROMPT) == ("gpt-4.1-mini", "script", "v3")


def test_engine_runs_everything_in_one_worker_thread_and_reports_paid_tokens(monkeypatch):
    monkeypatch.setattr(eng.system, "threshold", lambda: -100.0)
    e, llm, scorer, threads = system_engine(monkeypatch, [GOOD])
    analysis = e.analyze(CODE)
    assert scorer.calls == 1 and analysis.prepared.contexts[0].reranker == "k1-rr30"
    fix = e.fix(analysis)
    assert fix.result.code == GOOD and fix.result.output_format == "script" and fix.result.retry is None
    assert llm.prompts[0][0] == PROMPTS["v3"]["script"] and len(llm.prompts) == 1
    assert fix.paid == {"input_tokens": 1000, "output_tokens": 200, "usd": CHAT_MODELS["gpt-4.1-mini"].usd(1000, 200)}
    assert len(set(threads)) == 1 and threads[0] != threading.get_ident()


def test_below_the_threshold_there_is_no_prompt_for_the_llm(monkeypatch):
    monkeypatch.setattr(eng.system, "threshold", lambda: 100.0)
    e, *_ = system_engine(monkeypatch, [])
    analysis = e.analyze(CODE)
    assert analysis.prepared.prompt is None


def test_retries_once_and_counts_both_calls(monkeypatch):
    monkeypatch.setattr(eng.system, "threshold", lambda: -100.0)
    e, llm, *_ = system_engine(monkeypatch, [BROKEN, GOOD])
    fix = e.fix(e.analyze(CODE))
    assert fix.result.code == GOOD and len(llm.prompts) == 2
    assert fix.result.retry["remaining"] == [] and "PdfLoader" in fix.result.retry["problems"][0]
    assert fix.paid["input_tokens"] == 2000 and fix.paid["output_tokens"] == 400


def test_abstains_when_the_retry_is_still_wrong(monkeypatch):
    monkeypatch.setattr(eng.system, "threshold", lambda: -100.0)
    e, llm, *_ = system_engine(monkeypatch, [BROKEN, BROKEN])
    fix = e.fix(e.analyze(CODE))
    assert fix.result.code == CODE and len(llm.prompts) == 2
    assert fix.result.retry["remaining"] and all(d["decision"] == "sin_reemplazo_confiable"
                                                 for d in fix.result.report if "decision" in d)


def test_without_imports_for_the_llm_the_fix_does_not_create_it(monkeypatch):
    monkeypatch.setattr(eng.system, "threshold", lambda: 100.0)
    e, *_ = system_engine(monkeypatch, [])
    fix = e.fix(e.analyze(CODE))
    assert fix.result.llm_called is False and fix.result.code == CODE and fix.paid["usd"] == 0.0


def test_the_ui_does_not_use_the_study_caches_for_the_llm_or_the_query_embeddings(monkeypatch):
    calls = []
    monkeypatch.setattr(eng.system, "open_index", lambda cache=True: calls.append(("index", cache)) or "store")
    monkeypatch.setattr(eng.system, "chat_llm", lambda cache=True: calls.append(("llm", cache)) or "llm")
    e = eng.Engine()
    assert e._run(e._get, "store") == "store" and e._run(e._get, "llm") == "llm"
    assert calls == [("index", False), ("llm", False)]
