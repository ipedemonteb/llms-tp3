"""Diagnóstico de una corrida del generador (tprag.eval.diagnosis), con registros armados a mano: sin API."""

import importlib.util
import sys
from pathlib import Path

import pytest

from tprag.eval import diagnosis as dg

ORIGINAL = "import viejo\n"


def record(item, *, score=1.0, decision="reemplazado", success=False, changed=True, applicable=True, parses=True,
           target_gone=True, package_ok=True, undefined=(), llm_error=None, retry=None, packages=("bueno",),
           expected="reemplazado", changed_lines=None):
    t = {"module": "viejo", "expected": expected, "decision": decision, "replacement_package": "x",
         "retrieval_score": score, "passed_threshold": True, "success": success, "target_gone": target_gone,
         "package_ok": package_ok}
    r = {"id": item, "subset": "synthetic", "targets": [t], "llm_error": llm_error, "applicable": applicable,
         "parses": parses, "hallucinated_imports": [], "undefined_names": list(undefined), "retry": retry,
         "code": "import nuevo\n" if changed else ORIGINAL,
         "contexts": [{"module": "viejo", "chunks": [{"package": p} for p in packages]}]}
    if changed_lines is not None:
        r["changed_lines"] = changed_lines
    return r


def items(*ids, acceptable=("bueno",)):
    return {i: {"code": ORIGINAL, "targets": [{"module": "viejo", "acceptable_packages": list(acceptable)}]}
            for i in ids}


def causes(*records, threshold=0.0):
    d = dg.diagnose(list(records), items(*(r["id"] for r in records)), threshold)
    return [f["cause"] for f in d["failures"]], d


def test_una_causa_por_import_en_orden():
    found, d = causes(
        record("a", score=-1.0),                                   # debajo del umbral, aunque el LLM haya contestado
        record("b", llm_error="length", changed=False),
        record("c", retry={"remaining": ["no parsea"]}, changed=False),
        record("d", decision="sin_reemplazo_confiable", changed=False),
        record("e", decision="sin_reemplazo_confiable", changed=False, packages=("otro",)),
        record("f", applicable=False, changed=False),
        record("g", parses=False),
        record("h", target_gone=False),
        record("i", package_ok=False),
        record("j", package_ok=False, packages=("otro",)),
        record("k", undefined=("np",)),
        record("ok", success=True),
    )
    assert found == [dg.BELOW, dg.CUT, dg.POSTVAL, dg.ABSTAIN_IN, dg.ABSTAIN_OUT, dg.NOT_APPLIED, dg.NOT_PARSED,
                     dg.HALF, dg.WRONG_IN, dg.WRONG_OUT, dg.HALF]
    assert d["good"] == 1 and d["not_successful"] == 11 and sum(d["causes"].values()) == 11


def test_puntaje_igual_al_umbral_pasa():
    found, _ = causes(record("a", score=0.5, decision="sin_reemplazo_confiable", changed=False), threshold=0.5)
    assert found == [dg.ABSTAIN_IN]  # como el sistema: pasa si puntaje ≥ umbral


def test_negativo_cambiado_y_cuanto_reescribe():
    neg = record("n", expected="sin_reemplazo_confiable", changed=True, changed_lines=12)
    ok = record("ok", success=True, changed_lines=3)
    same = record("s", decision="sin_reemplazo_confiable", changed=False, changed_lines=0)  # no cuenta: no cambió
    d = dg.diagnose([neg, ok, same], items("n", "ok", "s") | {"n": {"code": ORIGINAL, "targets": [
        {"module": "viejo", "acceptable_packages": []}]}}, 0.0)
    assert [w["item"] for w in d["wrong_changes"]] == ["n"]
    assert d["rewrite"] == {"scripts": 2, "median": 7.5, "max": 12, "over_10": 1}


def test_comparacion_pareada():
    ids = [f"i{n}" for n in range(20)]
    a = [record(i, success=True) for i in ids]
    b = [record(i, success=n < 10, package_ok=n < 10) for n, i in enumerate(ids)]
    c = dg.compare_runs(a, b, items(*ids), 0.0, 0.0, n_boot=500)
    assert c["success_pos"]["a"] == 1.0 and c["success_pos"]["b"] == 0.5 and c["success_pos"]["verdict"] == "mejor"
    assert c["positives_won"] == 10 and c["positives_lost"] == 0
    assert dg.compare_runs(a, a, items(*ids), 0.0, 0.0, n_boot=500)["success_pos"]["verdict"] == "empate"
    with pytest.raises(ValueError):
        dg.compare_runs(a, a[:5], items(*ids), 0.0, 0.0, n_boot=500)


def test_el_script_no_pisa_salidas(tmp_path, monkeypatch):
    path = Path(__file__).parents[2] / "src" / "scripts"
    monkeypatch.syspath_prepend(str(path))
    spec = importlib.util.spec_from_file_location("diagnose_generation", path / "diagnose_generation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "STUDY_DIR", tmp_path)
    monkeypatch.setattr(mod, "load_run", lambda run: ("corrida", [record("ok", success=True)]))
    monkeypatch.setattr(mod, "load_items", lambda: items("ok"))
    monkeypatch.setattr(sys, "argv", ["diagnose_generation.py", "corrida"])
    assert mod.main() == 0 and (tmp_path / "diagnosis_corrida.json").exists()
    before = (tmp_path / "diagnosis_corrida.md").read_text()
    (tmp_path / "diagnosis_corrida.md").write_text("no tocar")
    assert mod.main() == 1 and (tmp_path / "diagnosis_corrida.md").read_text() == "no tocar"
    assert before.startswith("# Diagnóstico del generador")
