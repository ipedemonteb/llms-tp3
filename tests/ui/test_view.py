"""Vistas de la interfaz (tprag.ui.view): funciones puras."""

import json

from tprag.generate.context import TargetContext
from tprag.generate.pipeline import FixResult
from tprag.ui.view import (decision_rows, fixed_name, import_rows, load_examples, report_json, retry_note,
                           target_rows, unified_diff)
from tprag.validate.validate import ImportResult


def test_import_rows_say_what_goes_to_the_rag():
    rows = import_rows([ImportResult("csv", 1, [], [], "stdlib"),
                        ImportResult("PyPDF2", 2, [], [], "deprecated", package="PyPDF2", reason="siguió como pypdf",
                                     warnings=[{"code": "no_recent_release", "detail": "x"}])])
    assert [(r["import"], r["estado"], r["va al RAG"]) for r in rows] == [("csv", "biblioteca estándar", "no"),
                                                                          ("PyPDF2", "deprecado", "sí")]
    assert rows[1]["advertencias"] == "no_recent_release" and rows[0]["paquete"] == "-"


def test_target_rows_compare_the_score_with_the_threshold():
    chunks = [{"package": "pypdf", "n_tokens": 300}, {"package": "pypdf", "n_tokens": 200},
              {"package": "pdfplumber", "n_tokens": 100}]
    rows = target_rows([TargetContext("PyPDF2", "deprecated", "q", 0.51234, True, chunks),
                        TargetContext("serialx", "not_found", "q", 0.2, False)], threshold=0.45)
    assert rows[0]["pasa"] == "sí" and rows[0]["confianza"] == 0.5123
    assert rows[0]["tokens de contexto"] == 600 and rows[0]["librerías en el contexto"] == "pypdf, pdfplumber"
    assert rows[1]["pasa"] == "no" and rows[1]["librerías en el contexto"] == "-"


def test_decision_rows_only_for_imports_with_a_decision():
    report = [{"module": "csv"},
              {"module": "PyPDF2", "decision": "reemplazado", "replacement_package": "pypdf", "reason": "r",
               "sources": ["a", "b"]},
              {"module": "x", "decision": "sin_reemplazo_confiable", "replacement_package": None, "reason": "",
               "sources": []},
              {"module": "mock", "decision": "reemplazado", "replacement_package": None,
               "replacement_package_llm": "unittest", "reason": "", "sources": []}]
    assert [(r["import"], r["decisión"], r["reemplazo"], r["fuentes"]) for r in decision_rows(report)] == [
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


def test_load_examples_keeps_the_handwritten_subsets(tmp_path):
    items = [{"id": "dep-1", "subset": "manual_deprecated", "task": "t", "code": "c", "split": "dev"},
             {"id": "neg-1", "subset": "negative", "task": "t", "code": "c", "split": "test"},
             {"id": "syn-1", "subset": "synthetic", "task": "t", "code": "c", "split": "dev"}]
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n")
    assert sorted(load_examples(tmp_path)) == ["dep-1", "neg-1"]


def test_retry_note_says_whether_the_retry_fixed_it_or_the_system_abstained():
    assert retry_note(FixResult("x", [], True)) is None
    fixed = FixResult("x", [], True, retry={"problems": ["nombres sin definir: `a`"], "remaining": []})
    assert retry_note(fixed) == (False, ["nombres sin definir: `a`"])
    abstained = FixResult("x", [], True, retry={"problems": ["primero"], "remaining": ["sigue mal"]})
    assert retry_note(abstained) == (True, ["sigue mal"])
