"""Estimación de costo de embeddings: tokens contados con el tokenizer del modelo (`cl100k_base` para text-embedding-3,
el mismo que usa la API, así que el conteo coincide con lo facturado)."""

from tprag.split.tokens import token_counter

from .embeddings import EmbeddingModel


def estimate(texts: list[str], model: EmbeddingModel) -> dict:
    count = token_counter(model.tokenizer)
    tokens = sum(count(t) for t in texts)
    return {"texts": len(texts), "tokens": tokens, "usd": round(tokens / 1e6 * model.usd_per_m_tokens, 6)}
