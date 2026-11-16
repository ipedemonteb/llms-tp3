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


def test_name_hits_marks_code_and_prose_in_every_chunk():
    texts = ["El lector tiene `pages`.\n```python\nreader = PdfReader(f)\nreader.pages[0]\n```\nFin: pages.",
             "Otro chunk sin el nombre.", "```\npages = 3\n```"]
    assert review.name_hits("pages", texts) == [
        (1, False, "El lector tiene `pages`."), (1, True, "reader.pages[0]"), (1, False, "Fin: pages."),
        (3, True, "pages = 3")]


def test_name_hits_uses_whole_words_like_the_heuristic():
    texts = ["jwt_x = 1\nmyjwt.decode()", "import jwt"]
    assert review.name_hits("jwt", texts) == [(2, False, "import jwt")]
    assert review.name_hits("decode", ["nada"]) == []


def test_highlight_marks_whole_words_only():
    assert review.highlight("jwt.decode(jwt_x)", "jwt") == "»jwt«.decode(jwt_x)"


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


def test_kappa_is_one_with_full_agreement_and_undefined_without_variation():
    assert review.kappa([("si", "si"), ("no", "no"), ("parcial", "parcial")]) == pytest.approx(1)
    assert review.kappa([("si", "si"), ("si", "si")]) is None


def test_agreement_counts_exact_within_one_and_kappa():
    pairs = [("si", "si"), ("no", "parcial"), ("si", "no"), ("parcial", "parcial")]
    a = review.agreement(pairs, boot=200)
    assert a["n"] == 4 and a["exact"] == 0.5 and a["within_one"] == 0.75
    # po = 0,5; pe = 0,5·0,25 (si) + 0,25·0,5 (parcial) + 0,25·0,25 (no) = 0,3125
    assert a["kappa"] == pytest.approx((0.5 - 0.3125) / (1 - 0.3125))
    assert a["confusion"]["si"]["no"] == 1 and a["confusion"]["no"]["parcial"] == 1
    lo, hi = a["exact_ci"]
    assert 0 <= lo <= a["exact"] <= hi <= 1
    assert review.agreement(pairs, boot=200) == a  # semilla fija


def test_heuristic_check_splits_false_yes_and_false_no():
    claims = [{"in_context": True}, {"in_context": False}, {"in_context": True}]
    chosen = [{"review_id": "R01", "claims": claims}, {"review_id": "R02", "claims": claims[:1]}]
    human = {"R01": {"errores_heuristica": [1, 2]}, "R02": {"errores_heuristica": []}}
    h = review.heuristic_check(chosen, human)
    assert (h["rows"], h["errors"], h["false_yes"], h["false_no"]) == (4, 2, 1, 1)
    assert h["accuracy"] == 0.5 and h["items_with_error"] == ["R01"]
    with pytest.raises(ValueError, match="fuera de la tabla"):
        review.heuristic_check(chosen, {"R01": {"errores_heuristica": [4]}, "R02": {"errores_heuristica": []}})


def test_remaining_skips_the_sample_and_keys_by_version_and_item():
    judged = {"base": {"b2": {}, "b1": {}, "x": {}}, "rerank": {"x": {}, "r1": {}}}
    chosen = [{"run": "base", "id": "x", "review_id": "R01"}, {"run": "rerank", "id": "r1", "review_id": "R02"}]
    todo = review.remaining(judged, chosen)
    assert [t["review_id"] for t in todo] == ["base:b1", "base:b2", "rerank:x"]
    assert todo[2] == {"run": "rerank", "id": "x", "review_id": "rerank:x"}


def test_write_batches_splits_by_batch_size(tmp_path):
    prompts = [{"review_id": f"base:{i}"} for i in range(review.OPUS_BATCH * 2 + 1)]
    assert review.write_batches(prompts, tmp_path) == 3
    sizes = [len(review.read_jsonl(tmp_path / f"prompts_{k}.jsonl")) for k in (1, 2, 3)]
    assert sizes == [review.OPUS_BATCH, review.OPUS_BATCH, 1]


def test_paired_difference_counts_direction_and_mean():
    d = review.paired_difference([("si", "no"), ("parcial", "no"), ("no", "no"), ("no", "parcial")], boot=200)
    assert (d["n"], d["higher"], d["equal"], d["lower"]) == (4, 2, 1, 1)
    assert d["mean"] == pytest.approx((1 + 0.5 + 0 - 0.5) / 4)
    lo, hi = d["ci"]
    assert lo <= d["mean"] <= hi
    assert review.paired_difference([("si", "no")] * 3, boot=200)["ci"] == [1, 1]
