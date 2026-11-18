"""src/scripts/review_generation.py: muestra, prompts y respuestas de Opus, sin API ni archivos reales."""

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


# --- Revisión con rúbrica común (a ciegas, 3 agentes, anclas) -----------------------------------------------------

def rubric_scripts():
    scripts = [("rerank", f"r{i}") for i in range(20)] + [("iter6", f"i{i}") for i in range(24)]
    previous = {f"i{i}": ("si", "parcial", "no")[i % 3] for i in range(24)}
    return scripts, previous


def test_rubric_plan_a_ciegas_con_anclas_y_corridas_intercaladas():
    scripts, previous = rubric_scripts()
    plan = review.rubric_plan(scripts, previous)
    assert plan == review.rubric_plan(scripts, previous)  # semilla fija
    key, anchors = plan["key"], plan["anchors"]
    assert len(key) == len(scripts) and all(b.startswith("s") and len(b) == 4 for b in key)
    assert sorted((v["run"], v["id"]) for v in key.values()) == sorted(scripts)
    assert [previous[key[b]["id"]] for b in anchors].count("no") == 2 and all(key[b]["run"] == "iter6" for b in anchors)
    rest = [b for ids in plan["agents"].values() for b in ids if b not in anchors]
    assert sorted(rest) == sorted(set(key) - set(anchors))  # cada script no ancla, una sola vez
    for ids in plan["agents"].values():
        assert set(anchors) <= set(ids)  # las anclas, en los tres agentes
        assert {key[b]["run"] for b in ids if b not in anchors} == {"rerank", "iter6"}  # las dos corridas


def test_prompt_con_rubrica_solo_cambia_las_instrucciones():
    from tprag.eval.judge import PRESERVE_RUBRIC, PRESERVE_SYSTEM, preserve_prompt

    plain, rubric = preserve_prompt("a", "b", ["m"]), preserve_prompt("a", "b", ["m"], rubric=True)
    assert plain[0] == PRESERVE_SYSTEM and rubric[0] == PRESERVE_SYSTEM + "\n\n" + PRESERVE_RUBRIC
    assert plain[1] == rubric[1]


# --- Validación de la Faithfulness contra Opus --------------------------------------------------------------------

def faith_fixture():
    faith = []
    for n in range(12):
        heuristic = {"mod": True, "f": True, "limit": n % 3 != 0}  # algunos con un nombre "no fiel"
        faith.append({"id": f"s{n}", "claims": [{"name": k, "kind": "kwarg", "line": 1} for k in heuristic],
                      "heuristic": heuristic, "changed": [(1, "x")], "fragments": [("pkg > a", "texto")]})
    return faith


def test_faith_plan_a_ciegas_con_anclas():
    faith = faith_fixture()
    plan = review.faith_plan(faith)
    assert plan == review.faith_plan(faith)
    assert len(plan["anchors"]) == 4 and all(b.startswith("f") for b in plan["key"])
    ids = {b: plan["key"][b] for b in plan["anchors"]}
    by_id = {f["id"]: f for f in faith}
    assert sum(not all(by_id[i]["heuristic"].values()) for i in ids.values()) == 2  # 2 con algún "no fiel"
    rest = [b for a in plan["agents"].values() for b in a if b not in plan["anchors"]]
    assert sorted(rest) == sorted(set(plan["key"]) - set(plan["anchors"]))
    assert all(set(plan["anchors"]) <= set(a) for a in plan["agents"].values())


def test_faith_respuesta_con_otros_nombres_se_rechaza():
    names = {"f001": ["mod", "limit"]}
    ok = {"review_id": "f001", "output": {"claims": [
        {"name": "mod", "supported": "si", "fragment": 1, "reason": "r"},
        {"name": "limit", "supported": "no", "fragment": None, "reason": "r"}]}}
    assert review.check_faith_answer(names, ok)["output"]["claims"][1]["supported"] == "no"
    with pytest.raises(ValueError):
        review.check_faith_answer(names, {**ok, "output": {"claims": ok["output"]["claims"][:1]}})


def test_faith_compare_cuenta_falsos_fiel_y_mayoria_en_anclas():
    plan = {"key": {"f001": "a", "f002": "b"}, "anchors": ["f002"],
            "names": {"f001": ["x", "y"], "f002": ["z"]},
            "heuristic": {"f001": {"x": True, "y": True}, "f002": {"z": False}}}
    claim = lambda n, s: {"name": n, "supported": s, "fragment": None, "reason": "r"}  # noqa: E731
    rows = [{"blind_id": "f001", "output": {"claims": [claim("x", "si"), claim("y", "no")]}}]
    rows += [{"blind_id": "f002", "output": {"claims": [claim("z", v)]}} for v in ("no", "no", "si")]
    r = review.faith_compare(plan, rows, boot=200)
    assert r["confusion"] == {"fiel_respaldado": 1, "falso_fiel": 1, "falso_no_fiel": 0, "no_fiel_no_respaldado": 1}
    assert r["claims"] == 3 and abs(r["accuracy"] - 2 / 3) < 1e-9
    assert r["anchors"] == {"scripts": 1, "claims": 1, "claims_all_agree": 0}
