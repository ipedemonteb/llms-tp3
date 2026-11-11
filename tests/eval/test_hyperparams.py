"""Grilla del estudio de hiperparámetros y regla de finalistas (eval/hyperparams.py)."""

from tprag.eval.hyperparams import BASE, BY_ID, FACTORS, VARIANTS, finalists


def test_nombres_de_la_grilla():
    assert BASE.chunks == "s3-1024"
    assert BY_ID["size512"].chunks == "s3-512"
    assert BY_ID["min64"].chunks == "s3-1024-min64"
    assert BASE.index_config(["a"]).collection_name().startswith("s3-1024__e4__r1__cap150__sub-")
    assert BY_ID["nocap"].index_config(None).collection_name() == "s3-1024__e4__r1__nocap"
    assert BY_ID["raw"].index_config(None).collection_name() == "s3-1024__e4__raw__cap150"
    assert BY_ID["size2048"].index_config(None).collection_name() == "s3-2048__e4l__r1__cap150"


def test_cada_variante_cambia_un_solo_factor():
    for v in VARIANTS:
        changed = [f for f, base_value, value in (
            ("size", BASE.size, v.size), ("min_tokens", BASE.min_tokens, v.min_tokens), ("cap", BASE.cap, v.cap),
            ("representation", BASE.representation, v.representation)) if base_value != value]
        assert changed == ([] if v.id == "base" else [v.factor])


def test_los_grupos_usan_variantes_que_existen():
    for _, ids, labels in FACTORS.values():
        assert set(ids) <= set(BY_ID) and set(labels) == set(ids) and "base" in ids


def row(span, pkg="empate", vs_best="peor", tokens=1.0):
    return {"span": span, "pkg_verdict": pkg, "span_vs_best": vs_best, "tokens": tokens}


def test_finalistas_mejor_primero_y_empatados_por_costo():
    rows = {"base": row(0.56, vs_best="empate", tokens=1.7), "a": row(0.60, tokens=2.0),
            "b": row(0.59, vs_best="empate", tokens=3.0), "c": row(0.58, vs_best="empate", tokens=1.0),
            "d": row(0.30)}
    # a es la mejor; b, c y base empatan con ella (nota: en la realidad "empate" sale del bootstrap pareado)
    assert finalists(rows) == ["a", "c", "base"]
    assert finalists(rows, n=5) == ["a", "c", "base", "b", "d"]


def test_finalistas_descarta_las_peores_en_paquete():
    rows = {"base": row(0.50, vs_best="empate"), "a": row(0.70, pkg="peor"), "b": row(0.55)}
    assert finalists(rows) == ["b", "base"]
    assert finalists({"a": row(0.7, pkg="peor")}) == []
