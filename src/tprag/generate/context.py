"""Contexto de cada import problemático (generador.md §2.3, §2.4).

Por target: consulta (la misma del paso 7) → 50 chunks → score del mejor chunk contra el umbral → si pasa, los
primeros chunks (sin filtrar por librería) hasta el presupuesto de tokens, cada uno con su encabezado
`paquete > página > secciones`.
"""

import json
from dataclasses import dataclass, field

from langchain_chroma import Chroma

from tprag.eval.metrics import CONTEXT_BUDGET, budget_cut
from tprag.eval.query import QUERIES
from tprag.eval.retrieval import N_RESULTS, hit, search_docs
from tprag.index.build import chunk_text
from tprag.split.store import ROOT

EVAL_RESULTS_DIR = ROOT / "results" / "eval"


def dev_threshold(collection: str, query: str = "A") -> float:
    """Umbral calibrado en dev por el paso 7 para esta configuración (`eval_retrieval.py`)."""
    path = EVAL_RESULTS_DIR / f"{collection}__q{query}__dev.json"
    if not path.exists():
        raise FileNotFoundError(f"falta {path.name}: correr eval_retrieval.py con esta colección (o pasar --threshold)")
    threshold = json.loads(path.read_text(encoding="utf-8"))["metrics"].get("threshold")
    if threshold is None:
        raise ValueError(f"{path.name} no tiene umbral: volver a correr eval_retrieval.py (o pasar --threshold)")
    return threshold["value"]


@dataclass
class TargetContext:
    module: str
    status: str  # deprecated | not_found (validador)
    query: str
    score: float | None  # del mejor chunk
    passed: bool  # score ≥ umbral: va al LLM
    chunks: list[dict] = field(default_factory=list)  # {"header", "text", "package", "id", "n_tokens"}, si pasó
    hits: list[dict] = field(default_factory=list)  # los 50 recuperados, como en el paso 7


def build_context(store: Chroma, code: str, module: str, status: str, threshold: float,
                  budget: int = CONTEXT_BUDGET, query: str = "A") -> TargetContext:
    text = QUERIES[query](code, module)
    found = search_docs(store, text, N_RESULTS)
    score = found[0][1] if found else None
    passed = score is not None and score >= threshold
    chunks = []
    if passed:
        n = budget_cut([doc.metadata["n_tokens"] for doc, _ in found], budget)
        chunks = [{"header": doc.metadata["prefix"], "text": chunk_text(doc), "package": doc.metadata["package"],
                   "id": doc.id, "n_tokens": doc.metadata["n_tokens"]} for doc, _ in found[:n]]
    return TargetContext(module, status, text, score, passed, chunks, [hit(doc, s) for doc, s in found])
