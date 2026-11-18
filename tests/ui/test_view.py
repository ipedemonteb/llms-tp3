"""Vistas de la interfaz (tprag.ui.view): funciones puras."""

import json

from tprag.generate.pipeline import FixResult
from tprag.ui.view import decision_rows, fixed_name, outcome, report_json, unified_diff


def test_decision_rows_only_for_imports_with_a_decision():
    report = [{"module": "csv"},
              {"module": "PyPDF2", "decision": "reemplazado", "replacement_package": "pypdf", "reason": "r",
               "sources": ["a", "b"]},
              {"module": "x", "decision": "sin_reemplazo_confiable", "replacement_package": None, "reason": "",
               "sources": []},
              {"module": "mock", "decision": "reemplazado", "replacement_package": None,
               "replacement_package_llm": "unittest", "reason": "", "sources": []}]
    assert [(r["Import"], r["Decisión"], r["Reemplazo"], r["Fuentes"]) for r in decision_rows(report)] == [
        ("PyPDF2", "reemplazado", "pypdf", 2), ("x", "sin reemplazo confiable", "-", 0),
        ("mock", "reemplazado", "unittest", 0)]


def test_unified_diff_and_names():
    assert unified_diff("a = 1\n", "a = 1\n") == ""
    diff = unified_diff("import PyPDF2\nx = 1", "import pypdf\nx = 1", "s.py")
    assert "-import PyPDF2\n" in diff and "+import pypdf\n" in diff and diff.startswith("--- a/s.py")
    assert fixed_name("s.py") == "s.fixed.py" and fixed_name("s") == "s.fixed.py"


def test_report_json_is_the_fix_script_report():
    out = json.loads(report_json(FixResult("code", [{"module": "m"}], True, edit_errors=["e"])))
    assert out == {"imports": [{"module": "m"}], "edit_errors": ["e"]}


def test_outcome_says_what_happened_in_one_line():
    assert outcome(FixResult("c", [{"module": "csv"}], False)) == [
        ("success", "No hay dependencias deprecadas ni inexistentes: el script no necesita cambios.")]
    report = [{"module": "PyPDF2", "decision": "reemplazado", "replacement_package": "pypdf"},
              {"module": "mock", "decision": "reemplazado", "replacement_package": None,
               "replacement_package_llm": "unittest"},
              {"module": "serialx", "decision": "sin_reemplazo_confiable", "replacement_package": None}]
    assert outcome(FixResult("c", report, True)) == [
        ("success", "Se reemplazó `PyPDF2` por `pypdf` y `mock` por `unittest`."),
        ("warning", "No se encontró un reemplazo confiable para `serialx`: queda igual en el script.")]
    assert outcome(FixResult("c", report[:1], True, edit_errors=["e"]))[0][0] == "warning"
