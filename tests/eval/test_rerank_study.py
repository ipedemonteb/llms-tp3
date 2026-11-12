"""Grilla del estudio del reranker y regla de decisión (eval/rerank_study.py)."""

from tprag.eval.rerank_study import BY_ID, KS, VARIANTS, choose


def test_grilla():
    assert len(VARIANTS) == 2 * len(KS) * 2 and len(BY_ID) == len(VARIANTS)
    v = BY_ID["k1-rr30"]
    assert (v.reranker, v.k, v.mode) == ("k1", 30, "rr")
    hits = [{"id": f"c{i}", "score": 1.0} for i in range(40)]
    out = v.apply(hits, [float(i) for i in range(40)])
    assert out[0]["id"] == "c29" and out[30]["id"] == "c30"  # reordena solo los 30 primeros
    assert BY_ID["k2-rrf10"].apply(hits, [0.0] * 40)[0]["id"] == "c0"


def row(span_verdict, pkg="empate", seconds=1.0, span=0.6):
    return {"span_verdict": span_verdict, "pkg_verdict": pkg, "seconds": seconds, "span": span}


def test_choose_exige_mejora_significativa_sin_empeorar_la_libreria():
    assert choose({"a": row("empate"), "b": row("peor")}) is None
    assert choose({"a": row("mejor", pkg="peor")}) is None
    assert choose({"a": row("mejor", pkg="empate")}) == "a"


def test_choose_prefiere_la_mas_barata_y_desempata_por_span():
    rows = {"cara": row("mejor", seconds=5.0, span=0.8), "barata": row("mejor", seconds=1.0, span=0.65),
            "barata2": row("mejor", seconds=1.0, span=0.7)}
    assert choose(rows) == "barata2"
