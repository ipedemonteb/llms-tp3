"""Caché de embeddings (index/embeddings.py), con un modelo falso: no usa la API."""

from langchain_core.embeddings import DeterministicFakeEmbedding

from tprag.index.cost import estimate
from tprag.index.embeddings import MODELS, CachedEmbeddings, EmbeddingModel, openai_embeddings


class Counting(DeterministicFakeEmbedding):
    calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return super().embed_documents(texts)


def test_cache_avoids_second_call(tmp_path):
    inner = Counting(size=8, calls=[])
    emb = CachedEmbeddings("fake", lambda: inner, tmp_path / "c.sqlite")
    first = emb.embed_documents(["a", "b", "a"])
    assert inner.calls == [["a", "b"]]  # repetidos dentro del lote: se piden una vez
    assert (emb.hits, emb.misses) == (1, 2)
    again = CachedEmbeddings("fake", lambda: inner, tmp_path / "c.sqlite").embed_documents(["b", "a"])
    assert inner.calls == [["a", "b"]]  # todo del caché, también en otra instancia (persistido)
    assert again[0] == first[1] and again[1] == first[0]


def test_missing_without_creating_model(tmp_path):
    def fail():
        raise AssertionError("no debería crear el modelo")

    emb = CachedEmbeddings("fake", fail, tmp_path / "c.sqlite")
    assert emb.missing(["x", "y", "x"]) == ["x", "y"]


def test_cache_separates_models(tmp_path):
    path = tmp_path / "c.sqlite"
    CachedEmbeddings("m1", lambda: DeterministicFakeEmbedding(size=4), path).embed_documents(["a"])
    assert CachedEmbeddings("m2", lambda: DeterministicFakeEmbedding(size=4), path).missing(["a"]) == ["a"]


def test_cache_key_uses_real_name_and_dimensions():
    assert MODELS["e1"].cache_key == "text-embedding-3-small"
    m = EmbeddingModel("e1", "text-embedding-3-small", "openai/text-embedding-3-small", 0.02, dimensions=512)
    assert m.cache_key == "text-embedding-3-small@512"


def test_openai_client_uses_api_name(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-falsa")  # solo se construye el cliente, no se llama a la API
    emb = openai_embeddings(MODELS["e1"])
    assert emb.model == "openai/text-embedding-3-small"
    assert emb.tiktoken_model_name == "text-embedding-3-small"


def test_estimate():
    out = estimate(["hello world"] * 2, MODELS["e1"])
    assert out["texts"] == 2 and out["tokens"] == 4  # "hello world" = 2 tokens en cl100k_base
    assert out["usd"] == round(4 / 1e6 * 0.02, 6)
