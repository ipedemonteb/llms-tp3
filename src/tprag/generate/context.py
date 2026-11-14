"""Contexto de cada import problemático (generador.md §2.3, §2.4).

Por target: consulta (la misma del paso 7) → 50 chunks → score del mejor chunk contra el umbral → si pasa, los
primeros chunks (sin filtrar por librería) hasta el presupuesto de tokens, cada uno con su encabezado
`paquete > página > secciones`.

Con reranker (Advanced RAG, plan_cierre.md §4): los primeros `k` chunks se reordenan por el puntaje del reranker (los
mismos pares que el estudio: consulta + chunk con prefijo R1) y la confianza que se compara con el umbral pasa a ser el
máximo de ese puntaje, con su propio umbral calibrado en dev.
"""

import json
from dataclasses import dataclass, field
from typing import Protocol

from langchain_chroma import Chroma

from tprag.eval.metrics import CONTEXT_BUDGET, budget_cut
from tprag.eval.query import QUERIES
from tprag.eval.retrieval import N_RESULTS, hit, search_docs
from tprag.index.build import chunk_text
from tprag.split.store import ROOT

EVAL_RESULTS_DIR = ROOT / "results" / "eval"
STUDY_DIR = ROOT / "results" / "study"


class Scorer(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> list[float]: ...


@dataclass
class Reranker:
    variant: str  # id del estudio, p. ej. "k1-rr30"
    scorer: Scorer
    k: int  # cuántos de los primeros chunks se reordenan


def load_reranker(variant: str) -> Reranker:
    """Reranker de una variante `rr` del estudio, con el caché de puntajes del estudio. El modelo se carga recién al
    primer par que no esté en el caché (necesita el grupo `local`)."""
    from tprag.eval.rerank_study import BY_ID  # noqa: PLC0415
    from tprag.index.rerank import cached_reranker  # noqa: PLC0415

    v = BY_ID.get(variant)
    if v is None or v.mode != "rr":
        raise ValueError(f"{variant}: solo se integran las variantes de orden del reranker (k1-rr30, ...)")
    return Reranker(variant, cached_reranker(v.reranker), v.k)


def dev_threshold(collection: str, query: str = "A") -> float:
    """Umbral calibrado en dev por el paso 7 para esta configuración (`eval_retrieval.py`)."""
    path = EVAL_RESULTS_DIR / f"{collection}__q{query}__dev.json"
    if not path.exists():
        raise FileNotFoundError(f"falta {path.name}: correr eval_retrieval.py con esta colección (o pasar --threshold)")
    threshold = json.loads(path.read_text(encoding="utf-8"))["metrics"].get("threshold")
    if threshold is None:
        raise ValueError(f"{path.name} no tiene umbral: volver a correr eval_retrieval.py (o pasar --threshold)")
    return threshold["value"]


def rerank_threshold(collection: str, variant: str, split: str = "dev") -> float:
    """Umbral sobre el máximo del puntaje del reranker, calibrado en dev por `analyze_rerank.py` (≥ 95% de los
    positivos pasan). Lo busca en los análisis de esta colección; si hay varios, tienen que coincidir."""
    values = set()
    for path in sorted(STUDY_DIR.glob("rerank*_analysis.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        th = d.get("variants", {}).get(variant, {}).get("threshold_rr")
        if d.get("collection") == collection and d.get("split") == split and th:
            values.add(th["value"])
    if len(values) != 1:
        raise ValueError(f"umbral de {variant} sobre {collection}: {sorted(values) or 'no hay'} "
                         "(correr analyze_rerank.py en dev, o pasar --threshold)")
    return values.pop()


@dataclass
class TargetContext:
    module: str
    status: str  # deprecated | not_found (validador)
    query: str
    score: float | None  # confianza: coseno del mejor chunk o, con reranker, máximo de su puntaje
    passed: bool  # score ≥ umbral: va al LLM
    chunks: list[dict] = field(default_factory=list)  # {"header", "text", "package", "id", "n_tokens"}, si pasó
    hits: list[dict] = field(default_factory=list)  # los 50 recuperados, en el orden final (con reranker, reordenados)
    reranker: str | None = None


def rerank_found(found: list, query: str, reranker: Reranker) -> tuple[list, list[float]]:
    """Reordena los primeros `k` (doc, coseno) por el puntaje del reranker (empate: el orden denso, como
    `index.rerank.rerank`); los demás quedan detrás. Devuelve el orden nuevo y los puntajes de los reordenados, en ese
    orden (el primero es el máximo: la confianza)."""
    head = found[:reranker.k]
    if not head:
        return found, []
    scores = reranker.scorer.score([(query, doc.page_content) for doc, _ in head])
    order = sorted(range(len(head)), key=lambda i: (-scores[i], i))
    return [head[i] for i in order] + found[reranker.k:], [scores[i] for i in order]


def build_context(store: Chroma, code: str, module: str, status: str, threshold: float,
                  budget: int = CONTEXT_BUDGET, query: str = "A", reranker: Reranker | None = None) -> TargetContext:
    text = QUERIES[query](code, module)
    found = search_docs(store, text, N_RESULTS)
    rr_scores: list[float] = []
    if reranker is not None:
        found, rr_scores = rerank_found(found, text, reranker)
        score = max(rr_scores) if rr_scores else None
    else:
        score = found[0][1] if found else None
    passed = score is not None and score >= threshold
    chunks = []
    if passed:
        n = budget_cut([doc.metadata["n_tokens"] for doc, _ in found], budget)
        chunks = [{"header": doc.metadata["prefix"], "text": chunk_text(doc), "package": doc.metadata["package"],
                   "id": doc.id, "n_tokens": doc.metadata["n_tokens"]} for doc, _ in found[:n]]
    hits = [hit(doc, s) for doc, s in found]
    for h, r in zip(hits, rr_scores):
        h["rerank_score"] = r
    return TargetContext(module, status, text, score, passed, chunks, hits, reranker.variant if reranker else None)
