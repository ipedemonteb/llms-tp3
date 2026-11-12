"""Reranker (index/rerank.py): reordenamiento, fusión y caché, con un puntuador falso (no descarga modelos)."""

import pytest

from tprag.index.rerank import RERANKERS, CachedScores, fuse, rerank


def hits(n):
    return [{"id": f"c{i}", "score": round(1 - i / 100, 2)} for i in range(n)]


def test_rerank_ordena_los_primeros_k_y_deja_el_resto():
    out = rerank(hits(5), [0.1, 0.9, 0.5], k=3)
    assert [h["id"] for h in out] == ["c1", "c2", "c0", "c3", "c4"]
    assert [h["dense_rank"] for h in out] == [2, 3, 1, 4, 5]
    assert out[0]["rerank_score"] == 0.9 and "rerank_score" not in out[3]
    assert out[0]["score"] == 0.99  # conserva el coseno


def test_rerank_empate_respeta_el_orden_denso():
    assert [h["id"] for h in rerank(hits(3), [0.5, 0.5, 0.7], k=3)] == ["c2", "c0", "c1"]


def test_rerank_k_mayor_que_los_hits_y_puntajes_faltantes():
    assert [h["id"] for h in rerank(hits(2), [0.1, 0.2], k=50)] == ["c1", "c0"]
    with pytest.raises(ValueError):
        rerank(hits(5), [0.1], k=3)


def test_fuse_combina_los_dos_ordenes():
    # el reranker invierte el orden: con RRF los extremos empatan y decide el orden denso
    out = fuse(hits(4), [0.1, 0.2, 0.3, 0.4], k=3)
    assert [h["id"] for h in out] == ["c0", "c2", "c1", "c3"]
    assert out[0]["rrf"] == pytest.approx(1 / 61 + 1 / 63) and out[-1]["dense_rank"] == 4
    # si el reranker está de acuerdo con el orden denso, no cambia nada
    assert [h["id"] for h in fuse(hits(4), [0.9, 0.8, 0.7, 0.6], k=4)] == ["c0", "c1", "c2", "c3"]


class Fake:
    def __init__(self):
        self.calls = []

    def score(self, pairs):
        self.calls.append(list(pairs))
        return [float(len(q) + len(d)) for q, d in pairs]


def test_cache_no_recalcula_y_persiste(tmp_path):
    fake = Fake()
    sc = CachedScores("m", lambda: fake, tmp_path / "r.sqlite", commit_every=2)
    pairs = [("q", "aa"), ("q", "b"), ("qq", "c")]
    assert sc.score(pairs) == [3.0, 2.0, 3.0]
    assert fake.calls == [[("q", "aa"), ("q", "b")], [("qq", "c")]]  # en lotes de commit_every
    again = CachedScores("m", lambda: fake, tmp_path / "r.sqlite")
    assert again.score([("qq", "c"), ("q", "aa")]) == [3.0, 3.0] and len(fake.calls) == 2
    assert (again.hits, again.misses) == (2, 0)


def test_missing_sin_crear_el_modelo_y_separa_modelos(tmp_path):
    def fail():
        raise AssertionError("no debería crear el modelo")

    CachedScores("m1", Fake, tmp_path / "r.sqlite").score([("q", "d")])
    assert CachedScores("m1", fail, tmp_path / "r.sqlite").missing([("q", "d"), ("q", "e"), ("q", "e")]) == [("q", "e")]
    assert CachedScores("m2", fail, tmp_path / "r.sqlite").missing([("q", "d")]) == [("q", "d")]


def test_registro():
    assert RERANKERS["k1"].instruction and not RERANKERS["k2"].instruction
    assert RERANKERS["k1"].cache_key.startswith("Qwen/Qwen3-Reranker-0.6B@len2048@float16@inst-")
    assert RERANKERS["k2"].cache_key == "Alibaba-NLP/gte-reranker-modernbert-base@len2048@float16"
