"""Salidas de las evaluaciones (tprag.eval.outputs): no se pisan resultados ya corridos."""

import argparse

import pytest

from tprag.eval.outputs import refuse_overwrite, tag_arg, tagged


def test_tagged_appends_the_tag_like_the_v1_v2_v3_runs():
    assert tagged("corrida__dev", None) == "corrida__dev"
    assert tagged("corrida__dev", "nivel1-anterior") == "corrida__dev__nivel1-anterior"


def test_refuse_overwrite_only_when_an_output_exists_and_no_force(tmp_path):
    new, old = tmp_path / "nueva.json", tmp_path / "vieja.json"
    old.write_text("{}")
    assert refuse_overwrite([new], force=False) is None
    assert "vieja.json" in refuse_overwrite([new, old], force=False)
    assert refuse_overwrite([new, old], force=True) is None


@pytest.mark.parametrize("bad", ["", "con espacio", "../fuera", "-empieza-con-guion"])
def test_tag_arg_rejects_names_that_break_file_names(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        tag_arg(bad)
    assert tag_arg("nivel1-anterior_v2.1") == "nivel1-anterior_v2.1"
