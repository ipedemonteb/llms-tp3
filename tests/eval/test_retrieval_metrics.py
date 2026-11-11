"""Métricas del retrieval (eval/metrics.py) y ubicación de los gold (eval/retrieval.py), con rankings armados a mano."""

import pytest

from tprag.eval.metrics import (auroc, budget_hit, covered_spans, library_ranking, ndcg, package_rank, span_rank,
                                summarize, target_metrics)
from tprag.eval.retrieval import gold_spans
from tprag.eval.verify import code_block_spans


def hit(package, doc_id="d", start=0, end=10, n_tokens=100, score=0.5):
    return {"id": f"{package}-{start}", "package": package, "score": score, "doc_id": doc_id, "start": start,
            "end": end, "n_tokens": n_tokens}


PACKAGES = ["big", "big", "small", "big", "other", "small"]


def test_best_chunk_es_el_orden_de_primera_aparicion():
    assert library_ranking(PACKAGES, "best_chunk") == ["big", "small", "other"]


def test_votacion_por_cantidad_y_suma_ponderada():
    assert library_ranking(PACKAGES, "count", 6) == ["big", "small", "other"]
    # top-3: big 1 + 1/2, small 1/3
    assert library_ranking(PACKAGES, "sum_rr", 3) == ["big", "small", "other"]
    # top-1: solo big votó; las demás siguen en orden de primera aparición
    assert library_ranking(["small", "big", "big", "big"], "count", 4) == ["big", "small"]
    assert library_ranking(["small", "big", "big", "big"], "count", 1) == ["small", "big"]


def test_estrategia_desconocida():
    with pytest.raises(ValueError):
        library_ranking(PACKAGES, "max", 3)


def test_package_rank():
    assert package_rank(["big", "small", "other"], ["small", "x"]) == 2
    assert package_rank(["big"], ["small"]) is None


def test_cobertura_del_bloque_gold():
    spans = [("d", 100, 200)]
    assert covered_spans(hit("p", start=150, end=300), spans) == {0}  # 50% justo
    assert covered_spans(hit("p", start=151, end=300), spans) == set()
    assert covered_spans(hit("p", doc_id="otro", start=0, end=999), spans) == set()


def test_span_rank_ndcg_y_presupuesto():
    spans = [("d", 0, 10), ("d", 100, 110)]
    hits = [hit("p", start=500, end=600), hit("p", start=0, end=50), hit("p", start=0, end=20),
            hit("p", start=90, end=120)]
    assert span_rank(hits, spans) == 2
    # suman la 2.ª y la 4.ª (la 3.ª repite el mismo bloque); ideal: las dos primeras posiciones
    ideal = 1 + 1 / 1.584962500721156
    assert ndcg(hits, spans) == pytest.approx((1 / 1.584962500721156 + 1 / 2.321928094887362) / ideal)
    assert budget_hit(hits, spans, budget=200)
    assert not budget_hit(hits, spans, budget=150)  # el 2.º ya no entra
    assert not budget_hit([hit("p", start=0, end=10, n_tokens=5000)], [("d", 500, 600)])


def test_el_primer_chunk_entra_aunque_supere_el_presupuesto():
    assert budget_hit([hit("p", start=0, end=10, n_tokens=5000)], [("d", 0, 10)], budget=2000)


def test_auroc():
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auroc([0.5], [0.5]) == 0.5
    assert auroc([0.9], []) is None


def record(subset, lang, hits, acceptable, spans):
    return {"subset": subset, "lang": lang, "metrics": target_metrics(hits, acceptable, spans)}


def test_summarize_separa_positivos_y_negativos():
    pos_hit = record("manual_deprecated", "es", [hit("pypdf", score=0.8)], ["pypdf"], [("d", 0, 10)])
    pos_miss = record("manual_deprecated", "en", [hit("other", start=50, end=60, score=0.6)], ["pypdf"],
                      [("d", 0, 10)])
    neg = record("negative", "en", [hit("other", score=0.4)], [], [])
    out = summarize([pos_hit, pos_miss, neg])
    assert out["all"]["targets"] == 2
    assert out["all"]["package"]["best_chunk"]["hit@1"] == 0.5
    assert out["all"]["package"]["best_chunk"]["mrr"] == 0.5
    assert out["all"]["span"]["hit@1"] == 0.5  # el que no encontró el gold cuenta como fallo
    assert out["by_lang"]["es"]["package"]["best_chunk"]["hit@1"] == 1.0
    assert out["by_subset"]["negative"]["targets"] == 0
    assert out["auroc"] == {"value": 1.0, "positives": 2, "negatives": 1}


def test_el_presupuesto_es_configurable_y_2000_es_el_default():
    hits = [hit("p", start=0, end=10, n_tokens=900), hit("p", start=20, end=30, n_tokens=900),
            hit("p", start=40, end=50, n_tokens=900)]
    gold = [("d", 40, 50)]  # lo cubre el 3.º chunk: 900 + 900 + 900 = 2.700 tokens
    by_default = target_metrics(hits, ["p"], gold)["span_hit@budget"]
    assert by_default is False and target_metrics(hits, ["p"], gold, budget=3000)["span_hit@budget"] is True
    records = [{"subset": "x", "lang": "en", "metrics": target_metrics(hits, ["p"], gold, budget=3000)}]
    assert summarize(records, budget=3000)["all"]["span"]["hit@3000tok"] == 1.0
    assert "hit@2000tok" in summarize(records)["all"]["span"]


DOC = "# T\n\n```python\na = 1\n```\n\ntexto\n\n```python\nb = 2\n```\n"


def test_code_block_spans_incluye_las_vallas():
    spans = code_block_spans(DOC)
    assert [code for _, _, code in spans] == ["a = 1", "b = 2"]
    assert DOC[spans[1][0]:spans[1][1]].startswith("```python\nb = 2\n```")


def test_gold_spans_ubica_el_bloque_aunque_se_haya_corrido():
    item = {"id": "x", "targets": [{"module": "m", "gold": [
        {"doc_id": "d", "source_path": "README.md", "block": 0, "code": "b = 2"}]}]}
    start, end, _ = code_block_spans(DOC)[1]
    assert gold_spans([item], {"d": DOC}) == {("x", "m"): [("d", start, end)]}
    item["targets"][0]["gold"][0]["code"] = "c = 3"
    with pytest.raises(ValueError):
        gold_spans([item], {"d": DOC})


def test_context_efficiency():
    from tprag.eval.metrics import context_efficiency

    spans = [("d", 0, 100)]
    # un chunk de 400 caracteres que trae el bloque de 100: 25%
    assert context_efficiency([hit("a", start=0, end=400, n_tokens=100)], spans) == pytest.approx(0.25)
    # un chunk de 100 caracteres que es justo el bloque: 100%
    assert context_efficiency([hit("a", start=0, end=100, n_tokens=50)], spans) == pytest.approx(1.0)
    # fuera del presupuesto no cuenta: el segundo chunk no entra (100 + 1950 > 2000)
    hits = [hit("a", start=0, end=100, n_tokens=100), hit("a", start=100, end=300, n_tokens=1950)]
    assert context_efficiency(hits, spans, 2000) == pytest.approx(1.0)
    # otro documento o sin chunks: 0
    assert context_efficiency([hit("a", doc_id="otro", start=0, end=100)], spans) == 0.0
    assert context_efficiency([], spans) == 0.0


def test_context_efficiency_no_cuenta_dos_veces_el_overlap():
    from tprag.eval.metrics import context_efficiency

    hits = [hit("a", start=0, end=80, n_tokens=10), hit("a", start=40, end=120, n_tokens=10)]
    # el bloque gold (0-100) queda cubierto una vez: 100 caracteres sobre 160 de contexto
    assert context_efficiency(hits, [("d", 0, 100)]) == pytest.approx(100 / 160)


def test_specificity_at_recall():
    from tprag.eval.metrics import specificity_at_recall

    pos = [0.9, 0.8, 0.7, 0.6]  # al 75% de recall el umbral es 0.7
    assert specificity_at_recall(pos, [0.75, 0.5, 0.4, 0.3], 0.75) == pytest.approx(0.75)  # solo 0.75 pasa
    assert specificity_at_recall(pos, [0.1, 0.2], 0.75) == 1.0
    assert specificity_at_recall(pos, [], 0.75) is None
    assert specificity_at_recall([], [0.1]) is None
