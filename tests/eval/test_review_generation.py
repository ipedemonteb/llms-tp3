"""Revisión manual (src/scripts/review_generation.py): muestra y parseo de la planilla, sin API ni archivos reales."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[2] / "src" / "scripts"
sys.path.insert(0, str(SCRIPTS))  # el script importa analyze_generation del mismo directorio
spec = importlib.util.spec_from_file_location("review_generation", SCRIPTS / "review_generation.py")
review = importlib.util.module_from_spec(spec)
sys.modules["review_generation"] = review
spec.loader.exec_module(review)

QUOTAS = {"base": {(True, "si"): 1, (False, "no"): 1}, "rerank": {(True, "si"): 2}}


def rows(prefix, n, success, verdict):
    return [{"id": f"{prefix}{i}", "success": success, "b": {"preserves": verdict}} for i in range(n)]


def judged():
    shared = rows("x", 2, True, "si")  # ítems juzgados en las dos corridas
    return {"base": shared + rows("b", 3, False, "no") + rows("p", 4, True, "parcial"),
            "rerank": shared + rows("r", 2, True, "si")}


def test_sample_respects_quotas_and_does_not_repeat_items():
    s = review.sample(judged(), QUOTAS, seed=1)
    assert len(s) == 4 and len({x["id"] for x in s}) == 4
    assert [x["review_id"] for x in s] == ["R01", "R02", "R03", "R04"]
    by_run = {run: sorted((x["group"]["success"], x["group"]["judge_b"]) for x in s if x["run"] == run)
              for run in ("base", "rerank")}
    assert by_run == {"base": [(False, "no"), (True, "si")], "rerank": [(True, "si"), (True, "si")]}


def test_sample_is_reproducible():
    assert review.sample(judged(), QUOTAS, seed=1) == review.sample(judged(), QUOTAS, seed=1)


def test_sample_fails_if_a_group_is_too_small():
    with pytest.raises(ValueError):
        review.sample(judged(), {"base": {(True, "no"): 1}}, seed=1)


def test_claim_rows_say_where_each_name_appears():
    original = "from PyPDF2 import PdfReader\nr = PdfReader(f)\nprint(r.numPages)\n"
    fixed = "from pypdf import PdfReader\nr = PdfReader(f)\nprint(len(r.pages))\n"
    context = "Usá `reader = PdfReader(path)` y `reader.pages`.\nOtra línea."
    got = {r["name"]: r for r in review.claim_rows(original, fixed, context)}
    assert got["pages"]["in_context"] and got["pages"]["where"].startswith("Usá")
    assert got["pypdf"]["in_context"] is False and got["pypdf"]["where"] is None


SHEET = """# Revisión

## R01

- preserva: sí
- que_se_perdio: línea 22: devuelve una sola sugerencia
- errores_heuristica: 2, 1

## R02

- preserva: `parcial`
- que_se_perdio:
- errores_heuristica: ninguno

## R03

- preserva:
- que_se_perdio:
- errores_heuristica:

## R04

- preserva: no
- que_se_perdio: todo
"""


def test_parse_sheet():
    got = review.parse_sheet(SHEET)
    assert got["R01"] == {"preserva": "si", "que_se_perdio": "línea 22: devuelve una sola sugerencia",
                          "errores_heuristica": [1, 2]}
    assert got["R02"]["preserva"] == "parcial" and got["R02"]["errores_heuristica"] == []
    assert got["R03"] == {"preserva": None, "que_se_perdio": "", "errores_heuristica": None}  # sin completar
    assert got["R04"]["errores_heuristica"] == []  # sin tabla de identificadores: nada que marcar


@pytest.mark.parametrize("line", ["- preserva: tal vez", "- errores_heuristica: el segundo"])
def test_parse_sheet_rejects_invalid_values(line):
    with pytest.raises(ValueError):
        review.parse_sheet(f"## R01\n\n{line}\n")


def test_opus_prompts_are_the_judge_prompts():
    from tprag.eval.judge import Preservation, preserve_prompt

    items = {"x": {"code": "import viejo\n"}}
    records = {"base": {"x": {"code": "import nuevo\n", "targets": [{"module": "viejo"}]}}}
    [p] = review.opus_prompts([{"review_id": "R01", "run": "base", "id": "x"}], items, records)
    assert (p["system"], p["user"]) == preserve_prompt("import viejo\n", "import nuevo\n", ["viejo"])
    assert p["review_id"] == "R01" and p["schema"] == Preservation.model_json_schema()


def test_check_answer_validates_with_the_judge_schema():
    ok = {"review_id": "R01", "output": {"lost_or_changed": [], "preserves": "si", "reason": "igual"}}
    assert review.check_answer({"R01"}, ok)["model"] == review.OPUS_MODEL
    with pytest.raises(ValueError):
        review.check_answer({"R01"}, ok | {"output": {"lost_or_changed": [], "preserves": "tal vez", "reason": "x"}})
    with pytest.raises(ValueError):
        review.check_answer({"R02"}, ok)
