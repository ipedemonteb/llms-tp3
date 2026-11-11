"""Caché de embeddings (index/embeddings.py), con un modelo falso: no usa la API."""

from langchain_core.embeddings import DeterministicFakeEmbedding

from tprag.index.cost import estimate
from tprag.index.embeddings import (MODELS, CachedEmbeddings, EmbeddingModel, LocalEmbeddings,
                                    openai_embeddings)


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
    m = EmbeddingModel("e1", "text-embedding-3-small", "text-embedding-3-small", 0.02, dimensions=512)
    assert m.cache_key == "text-embedding-3-small@512"


def test_openai_client_uses_api_name(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-falsa")  # solo se construye el cliente, no se llama a la API
    emb = openai_embeddings(MODELS["e1"])
    assert emb.model == "text-embedding-3-small"
    assert emb.tiktoken_model_name == "text-embedding-3-small"


def test_estimate():
    out = estimate(["hello world"] * 2, MODELS["e1"])
    assert out["texts"] == 2 and out["tokens"] == 4  # "hello world" = 2 tokens en cl100k_base
    assert out["usd"] == round(4 / 1e6 * 0.02, 6)


class Prefixed(DeterministicFakeEmbedding):
    """Como un modelo asimétrico: consulta y documento del mismo texto dan vectores distintos."""

    def embed_documents(self, texts):
        return super().embed_documents(["doc: " + t for t in texts])

    def embed_query(self, text):
        return super().embed_documents(["query: " + text])[0]


def test_asymmetric_cache_keeps_query_and_document_apart(tmp_path):
    emb = CachedEmbeddings("m", lambda: Prefixed(size=8), tmp_path / "c.sqlite", asymmetric=True)
    doc, query = emb.embed_documents(["x"])[0], emb.embed_query("x")
    assert doc != query
    assert emb.embed_query("x") == query and emb.embed_documents(["x"])[0] == doc  # del caché, sin mezclarse
    assert (emb.hits, emb.misses) == (2, 2)


def test_local_models_are_registered_without_price():
    for model_id in ("e3", "e4", "e5"):
        m = MODELS[model_id]
        assert m.provider == "local" and m.usd_per_m_tokens == 0 and m.api_name == ""
    assert MODELS["e5"].asymmetric and not MODELS["e4"].asymmetric and not MODELS["e1"].asymmetric


def test_local_embeddings_add_prefixes(monkeypatch):
    class FakeST:
        seen: list[str] = []

        def encode(self, texts, **kwargs):
            import numpy as np  # noqa: PLC0415

            FakeST.seen += texts
            assert kwargs["normalize_embeddings"]
            return np.ones((len(texts), 3))

    local = LocalEmbeddings(MODELS["e5"])
    monkeypatch.setattr(local, "_load", lambda: FakeST())
    local.embed_documents(["a"])
    local.embed_query("b")
    assert FakeST.seen == ["search_document: a", "search_query: b"]


def test_cache_key_separa_ventanas_distintas():
    # la clave de los modelos existentes no cambia (el caché ya armado sigue valiendo)
    assert MODELS["e4"].cache_key == "jinaai/jina-embeddings-v2-base-code"
    assert MODELS["e4l"].name == MODELS["e4"].name
    assert MODELS["e4l"].cache_key == "jinaai/jina-embeddings-v2-base-code@seq3072"
