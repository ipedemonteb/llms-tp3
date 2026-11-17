"""Motor de la interfaz (extra de la consigna): el mismo pipeline que `fix_script.py`, en dos pasos y con tiempos.

1. `analyze`: validador + retrieval (+ reranker) + umbral, sin LLM y sin costo de API (`pipeline.prepare`).
2. `fix`: la llamada al LLM y la post-validación (`pipeline.answer`), con los tokens y USD pagados.

El paso 2 tiene dos generadores (`GENERATORS`): el iterado (`__iter7`, el elegido en el Ej. 4: gpt-4.1-mini devuelve el
script completo y, si la corrección no pasa la post-validación, hay un reintento y después abstención) y el de antes de
iterar (gpt-4o-mini con lista de reemplazos). Usan los mismos prompts y esquemas que `eval_generation.py`, así que
comparten su caché de respuestas.

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

from tprag.generate.context import Reranker, dev_threshold, load_reranker, rerank_threshold
from tprag.generate.llm import CHAT_MODELS, CachedLLM
from tprag.generate.pipeline import FixResult, Prepared, answer, first_prompt, prepare
from tprag.generate.schema import ScriptOutput, ScriptOutputV2
from tprag.index.build import IndexConfig, open_collection
from tprag.index.embeddings import cached
from tprag.validate.pypi import CachedPyPI

FINAL_CONFIG = IndexConfig(chunks="s3-1024", console_mode="outputs_as_comments", model="e4", representation="r1",
                           cap=150, strategy="round_robin")  # la configuración elegida (plan_cierre.md §0)
VERSIONS = {"base": None, "k1-rr30": "k1-rr30"}  # versión → variante del reranker


@dataclass(frozen=True)
class Generator:
    """Cómo se le pide la corrección al LLM: los argumentos de `pipeline.answer`."""

    llm: str
    first: str = "edits"  # formato del primer intento: lista de reemplazos o el script completo
    retry: str | None = None  # formato del reintento después de la post-validación; None = sin reintento
    prompt: str = "v1"  # versión del primer prompt (`prompt.PROMPTS`)

    def schema(self):
        """El esquema de salida del LLM, o None para el de siempre (lista de reemplazos)."""
        if self.first != "script":
            return None
        return ScriptOutput if self.prompt == "v1" else ScriptOutputV2


ITERATED, NOT_ITERATED = "iter7", "sin-iterar"
GENERATORS = {ITERATED: Generator("gpt-4.1-mini", first="script", retry="script", prompt="v3"),
              NOT_ITERATED: Generator("gpt-4o-mini")}


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
    version: str
    threshold: float
    prepared: Prepared
    seconds: dict[str, float]  # validador, retrieval, reranker, total
    generator: str = NOT_ITERATED
    llm_estimate: dict | None = None  # tokens y USD de la primera llamada; None si ningún import pasa el umbral
    llm_cached: bool = False  # la primera llamada ya está en el caché (no se paga)


@dataclass
class Fix:
    result: FixResult
    seconds: float
    paid: dict = field(default_factory=dict)  # tokens y USD efectivamente pagados (0 si salió del caché)


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


def default_threshold(version: str, collection: str) -> float:
    """El umbral calibrado en dev para esa versión (el mismo que usa `fix_script.py` por defecto)."""
    variant = VERSIONS[version]
    return rerank_threshold(collection, variant) if variant else dev_threshold(collection)


class Engine:
    """Una instancia por proceso (la app la guarda con `st.cache_resource`)."""

    def __init__(self, config: IndexConfig = FINAL_CONFIG):
        self.config, self.collection = config, config.collection_name()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tprag-ui")
        self._objects: dict[str, Any] = {}

    def _run(self, fn, *args):
        return self._worker.submit(fn, *args).result()

    # --- Objetos con SQLite: se crean y se usan solo en el hilo del motor ---------------------------------------

    def _get(self, name: str):
        if name not in self._objects:
            if name == "store":
                self._objects[name] = open_collection(self.collection, cached(self.config.model))
            elif name == "pypi":
                self._objects[name] = CachedPyPI()
            elif name in GENERATORS:  # un caché del LLM por generador: cambia el modelo y el esquema de salida
                llm = CachedLLM(CHAT_MODELS[GENERATORS[name].llm])
                if (schema := GENERATORS[name].schema()) is not None:
                    llm.output_schema = schema
                self._objects[name] = llm
            else:  # variante del reranker
                self._objects[name] = load_reranker(name)
        return self._objects[name]

    # --- API para la app ------------------------------------------------------------------------------------

    def chunks(self) -> int:
        """Casos de uso en el índice (0 = falta armarlo con build_index.py)."""
        return self._run(lambda: self._get("store")._collection.count())

    def threshold(self, version: str) -> float:
        return default_threshold(version, self.collection)

    def analyze(self, code: str, version: str, threshold: float, budget: int,
                generator: str = NOT_ITERATED) -> Analysis:
        return self._run(self._analyze, code, version, threshold, budget, generator)

    def _analyze(self, code, version, threshold, budget, generator) -> Analysis:
        variant = VERSIONS[version]
        reranker = self._get(variant) if variant else None
        prepared, seconds = timed_prepare(code, self._get("store"), self._get("pypi"), threshold, budget, reranker)
        analysis = Analysis(version, threshold, prepared, seconds, generator)
        gen = GENERATORS[generator]
        if (prompt := first_prompt(prepared, gen.first, gen.prompt)) is not None:
            llm = self._get(generator)
            analysis.llm_estimate = llm.estimate([prompt])
            analysis.llm_cached = not llm.missing([prompt])
        return analysis

    def fix(self, analysis: Analysis) -> Fix:
        return self._run(self._fix, analysis)

    def _fix(self, analysis: Analysis) -> Fix:
        gen, llm = GENERATORS[analysis.generator], self._get(analysis.generator)
        before = (llm.input_tokens, llm.output_tokens)
        start = time.perf_counter()
        result = answer(analysis.prepared, llm, retry=gen.retry, retry_llm=llm, first=gen.first, version=gen.prompt)
        seconds = time.perf_counter() - start
        tokens_in, tokens_out = llm.input_tokens - before[0], llm.output_tokens - before[1]  # incluye el reintento
        return Fix(result, seconds, {"input_tokens": tokens_in, "output_tokens": tokens_out,
                                     "usd": CHAT_MODELS[gen.llm].usd(tokens_in, tokens_out)})
