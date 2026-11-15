"""Gráficos del análisis de costo (tprag.eval.plots_cost) con datos de juguete. Necesitan matplotlib (grupo
`analysis`): sin él, se saltean."""

import pytest

pytest.importorskip("matplotlib")

from tprag.eval import plots_cost as pc  # noqa: E402


def generator(context, output):
    return {"input_parts_mean": {"system": 500, "script": 300, "context": context},
            "per_call": {"output_mean": output}}


def point(usd, calls):
    return {"usd_per_1000_scripts": usd, "llm_calls": calls, "scripts": 100}


RESULT = {
    "calibration": {"input_overhead": 482},
    "generator": {"base": generator(2000, 250), "rerank": generator(2050, 280)},
    "operating_points": {v: {"permisivo_95": point(0.6, 90), "elegido": point(0.05, 8)} for v in ("base", "rerank")},
    "reranker": [{"reranker": r, "label": r.upper(), "k": k, "seconds_per_query": s * k}
                 for r, s in (("k1", 0.5), ("k2", 0.1)) for k in (10, 30)],
}


def test_token_segments_put_the_input_parts_first_and_the_output_last():
    segs = pc.token_segments(RESULT)["base"]
    assert [name for name, _, _ in segs] == ["Instrucciones", "Esquema y formato", "Script",
                                             "Fragmentos recuperados", "Salida"]
    assert [t for _, t, _ in segs] == [500, 482, 300, 2000, 250]


def test_cost_figures_write_one_png_each(tmp_path):
    paths = pc.cost_figures(RESULT, tmp_path / "cost")
    assert [p.name for p in paths] == ["1_tokens_por_llamada.png", "2_latencia_reranker.png",
                                       "3_usd_cada_1000_scripts.png"]
    assert all(p.stat().st_size > 0 for p in paths)
