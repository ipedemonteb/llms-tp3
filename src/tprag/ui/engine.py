"""Motor de la interfaz (extra de la consigna): el sistema final (`tprag.generate.system`, el mismo que corre
`fix_script.py`), en dos pasos y con tiempos.

Cachés: el de PyPI y el de puntajes del reranker se usan (el reranker tarda ~1 min por import en CPU); el LLM y los
embeddings de la consulta no (`system.open_index(cache=False)`, `system.chat_llm(cache=False)`): cada corrección es una
llamada real a la API.

1. `analyze`: validador + retrieval + reranker + umbral, sin LLM y sin costo de API (`pipeline.prepare`).
2. `fix`: la llamada al LLM, la post-validación y, si hace falta, el reintento (`system.correct`), con los tokens y USD
   pagados.

Los cachés de embeddings, del reranker y del LLM usan SQLite, que no deja compartir una conexión entre hilos, y
Streamlit atiende cada sesión en un hilo distinto. Por eso todo corre en **un solo hilo propio** (`_worker`): los
objetos con SQLite se crean ahí la primera vez que hacen falta y nunca salen de ese hilo.

Tiempos por etapa: el retrieval se mide envolviendo al índice (la búsqueda incluye embeber la consulta), el reranker
envolviendo a su puntuador, y el validador es el resto de `prepare` (PyPI con caché + armado del contexto).
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from tprag.generate import system
from tprag.generate.context import Reranker, load_reranker
from tprag.generate.llm import CHAT_MODELS
from tprag.generate.pipeline import FixResult, Prepared, prepare
from tprag.validate.pypi import CachedPyPI


class _Timer:
    def __init__(self):
        self.seconds = 0.0
        self.calls = 0

    def time(self, fn, *args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self.seconds += time.perf_counter() - start
            self.calls += 1


class TimedStore:
    """Delega en el índice y cronometra la búsqueda (lo único que usa `build_context`)."""

    def __init__(self, store, timer: _Timer):
        self._store, self._timer = store, timer

    def similarity_search_with_score(self, query, k):
        return self._timer.time(self._store.similarity_search_with_score, query, k)


class TimedScorer:
    """Delega en el puntuador del reranker y cronometra cada lote de pares."""

    def __init__(self, scorer, timer: _Timer):
        self._scorer, self._timer = scorer, timer

    def score(self, pairs):
        return self._timer.time(self._scorer.score, pairs)


@dataclass
class Analysis:
    prepared: Prepared
    seconds: dict[str, float]  # validador, retrieval, reranker, total


@dataclass
class Fix:
    result: FixResult
    seconds: float
    paid: dict = field(default_factory=dict)  # tokens y USD pagados (0 si no se llamó al LLM)


def timed_prepare(code: str, store, fetch, threshold: float, budget: int, reranker: Reranker | None) -> tuple[
        Prepared, dict[str, float]]:
    """`pipeline.prepare` con el tiempo de cada etapa. Levanta `SyntaxError` si el script no parsea."""
    search, scoring = _Timer(), _Timer()
    timed = Reranker(reranker.variant, TimedScorer(reranker.scorer, scoring), reranker.k) if reranker else None
    start = time.perf_counter()
    prepared = prepare(code, TimedStore(store, search), fetch, threshold, budget, timed)
    total = time.perf_counter() - start
    return prepared, {"validador": max(total - search.seconds - scoring.seconds, 0.0), "retrieval": search.seconds,
                      "reranker": scoring.seconds, "total": total}


class Engine:
    """Una instancia por proceso (la app la guarda con `st.cache_resource`)."""

    def __init__(self):
        self.collection = system.COLLECTION
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tprag-ui")
        self._objects: dict[str, Any] = {}

    def _run(self, fn, *args):
        return self._worker.submit(fn, *args).result()

    # --- Objetos con SQLite: se crean y se usan solo en el hilo del motor ---------------------------------------

    def _get(self, name: str):
        if name not in self._objects:
            if name == "store":
                self._objects[name] = system.open_index(cache=False)
            elif name == "pypi":
                self._objects[name] = CachedPyPI()
            elif name == "llm":
                self._objects[name] = system.chat_llm(cache=False)
            else:  # "reranker"
                self._objects[name] = load_reranker(system.RERANKER)
        return self._objects[name]

    # --- API para la app ------------------------------------------------------------------------------------

    def chunks(self) -> int:
        """Casos de uso en el índice (0 = falta armarlo con build_index.py)."""
        return self._run(lambda: self._get("store")._collection.count())

    def analyze(self, code: str) -> Analysis:
        return self._run(self._analyze, code)

    def _analyze(self, code) -> Analysis:
        prepared, seconds = timed_prepare(code, self._get("store"), self._get("pypi"), system.threshold(),
                                          system.BUDGET, self._get("reranker"))
        return Analysis(prepared, seconds)

    def fix(self, analysis: Analysis) -> Fix:
        return self._run(self._fix, analysis)

    def _fix(self, analysis: Analysis) -> Fix:
        if analysis.prepared.prompt is None:  # ningún import va al LLM: el resultado se arma sin llamarlo
            return Fix(system.correct(analysis.prepared, None), 0.0, {"input_tokens": 0, "output_tokens": 0, "usd": 0.0})
        llm = self._get("llm")
        before = (llm.input_tokens, llm.output_tokens)
        start = time.perf_counter()
        result = system.correct(analysis.prepared, llm)
        seconds = time.perf_counter() - start
        tokens_in, tokens_out = llm.input_tokens - before[0], llm.output_tokens - before[1]  # incluye el reintento
        return Fix(result, seconds, {"input_tokens": tokens_in, "output_tokens": tokens_out,
                                     "usd": CHAT_MODELS[system.LLM].usd(tokens_in, tokens_out)})
