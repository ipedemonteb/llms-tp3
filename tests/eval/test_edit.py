"""Export / import del dataset de evaluación (eval/edit.py), sin doc normalizada real."""

import pytest

from tprag.eval.edit import export_items, import_items

DOC = "# T\n\n```python\nimport a\n```\n\ntexto\n\n```python\nb = 2\n```\n"
DOCS = {("pkg", "README.md"): ("d1", DOC)}
ITEM = {"id": "x-01", "subset": "manual_deprecated", "task": "t", "code": "import old\nold.run()\n",
        "targets": [{"module": "old", "expected_status": "deprecated", "acceptable_packages": ["pkg"],
                     "gold": [{"doc_id": "d1", "package": "pkg", "source_path": "README.md", "block": 1,
                               "code": "b = 2"}]}],
        "provenance": {"author": "a"}, "split": "dev"}


def test_ida_y_vuelta_sin_cambios(tmp_path):
    export_items([ITEM], tmp_path)
    assert (tmp_path / "x-01.py").read_text() == ITEM["code"]
    assert import_items(tmp_path, DOCS, ["x-01"]) == [ITEM]


def test_cambiar_el_bloque_recalcula_el_gold(tmp_path):
    export_items([ITEM], tmp_path)
    meta = (tmp_path / "x-01.json").read_text().replace('"block": 1', '"block": 0')
    (tmp_path / "x-01.json").write_text(meta)
    gold = import_items(tmp_path, DOCS, ["x-01"])[0]["targets"][0]["gold"][0]
    assert gold["block"] == 0 and gold["code"] == "import a"


def test_bloque_inexistente(tmp_path):
    export_items([ITEM], tmp_path)
    (tmp_path / "x-01.json").write_text((tmp_path / "x-01.json").read_text().replace('"block": 1', '"block": 9'))
    with pytest.raises(ValueError, match="2 bloques"):
        import_items(tmp_path, DOCS, ["x-01"])
