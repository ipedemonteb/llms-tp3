"""Chequeos del dataset de evaluación (eval/verify.py), sin red ni doc normalizada real."""

from tprag.eval.verify import check_gold, check_hints, check_validator, code_blocks
from tprag.validate.validate import ImportResult

DOC = """\
# Título

```bash
pip install pypdf
```

- Un paso de una lista:

  ```python
  reader = PdfReader("a.pdf")
  ```

~~~python
print(reader.pages[0].extract_text())
~~~
"""


def gold(block, code, doc_id="d1"):
    return {"doc_id": doc_id, "source_path": "README.md", "block": block, "code": code}


def test_code_blocks_numera_todos_los_bloques_en_orden():
    assert code_blocks(DOC) == [
        "pip install pypdf",
        '  reader = PdfReader("a.pdf")',
        "print(reader.pages[0].extract_text())",
    ]


def test_gold_en_su_lugar():
    assert check_gold(gold(2, "print(reader.pages[0].extract_text())"), {"d1": DOC}) == (None, None)


def test_gold_que_se_corrio_se_reubica_con_aviso():
    error, note = check_gold(gold(0, "print(reader.pages[0].extract_text())"), {"d1": DOC})
    assert error is None and "→ bloque 2" in note


def test_gold_que_no_esta_es_error():
    error, _ = check_gold(gold(1, "otra cosa"), {"d1": DOC})
    assert "revisar a mano" in error
    error, _ = check_gold(gold(0, "x", doc_id="nope"), {"d1": DOC})
    assert "no está en la doc normalizada" in error


def result(module, status):
    return ImportResult(module=module, line=1, names=[module], usages=[], status=status)


ITEM = {"id": "x", "code": "", "targets": [
    {"module": "PyPDF2", "expected_status": "deprecated", "acceptable_packages": ["pypdf"], "gold": []}]}


def test_validador_ok():
    assert check_validator(ITEM, [result("os", "stdlib"), result("PyPDF2", "deprecated"),
                                  result("pandas", "ok")]) == []


def test_validador_target_con_otro_estado_y_falsa_alarma():
    errors = check_validator(ITEM, [result("PyPDF2", "ok"), result("requests", "unknown")])
    assert len(errors) == 2
    assert "se esperaba deprecated" in errors[0] and "requests" in errors[1]


def test_validador_target_ausente():
    assert "no encontró" in check_validator(ITEM, [result("os", "stdlib")])[0]


def test_pistas():
    assert check_hints({**ITEM, "code": "from PyPDF2 import PdfReader  # TODO migrar a pypdf"})
    assert not check_hints({**ITEM, "code": "from PyPDF2 import PdfReader\nimport mypypdf_utils"})
    item = {**ITEM, "targets": [{**ITEM["targets"][0], "acceptable_packages": ["google-api-python-client"]}]}
    assert check_hints({**item, "code": "import google_api_python_client"})
