"""Notebooks, galerías de ejemplos y docstrings (sources.py). Usan pandoc."""

import json

from tprag.normalize.sources import (
    docstrings_to_rst, gallery_to_rst, normalize_docstrings, normalize_gallery, normalize_notebook, notebook_to_md,
)


# --- Notebooks ----------------------------------------------------------------------------------------

NOTEBOOK = json.dumps({
    "metadata": {"kernelspec": {"language": "python"}},
    "cells": [
        {"cell_type": "markdown", "source": ["# Reading data\n", "\n", "Load a CSV with `read_csv`."]},
        {"cell_type": "code", "source": ["import pandas as pd\n", "df = pd.read_csv('data.csv')"], "outputs": []},
        {"cell_type": "raw", "source": [".. toctree::"]},
        {"cell_type": "code", "source": ["# Hidden cell to avoid CSS clashes"], "metadata": {"nbsphinx": "hidden"}},
        {"cell_type": "code", "source": ["setup_plots()"], "metadata": {"tags": ["remove-cell"]}},
        {"cell_type": "code", "source": [], "outputs": []},
    ],
})


def test_notebook_to_md():
    assert notebook_to_md(NOTEBOOK) == (
        "# Reading data\n\nLoad a CSV with `read_csv`.\n\n"
        "```python\nimport pandas as pd\ndf = pd.read_csv('data.csv')\n```\n")


def test_normalize_notebook():
    norm = normalize_notebook(NOTEBOOK)
    assert norm.block_langs == ["python"]
    assert "toctree" not in norm.markdown


# --- Galerías ---------------------------------------------------------------------------------------

SPHINX_GALLERY = '''\
"""
==================
Simple line plot
==================

Plot a line with :func:`matplotlib.pyplot.plot`.
"""
# sphinx_gallery_thumbnail_number = 2
import matplotlib.pyplot as plt

# %%
# Draw the data
# -------------
#
# The line uses the default style.

plt.plot([1, 2, 3])  # inline comment stays in the code

###############################################################################
# Show the figure.

plt.show()
'''


def test_gallery_to_rst_splits_text_and_code():
    rst = gallery_to_rst(SPHINX_GALLERY)
    assert "Simple line plot" in rst and "sphinx_gallery" not in rst
    assert "Draw the data\n-------------\n\nThe line uses the default style." in rst
    assert "   plt.plot([1, 2, 3])  # inline comment stays in the code" in rst


def test_normalize_gallery():
    norm = normalize_gallery(SPHINX_GALLERY)
    assert norm.block_langs == ["python", "python", "python"]
    assert "`matplotlib.pyplot.plot`" in norm.markdown
    assert "## Draw the data" in norm.markdown and "Show the figure." in norm.markdown


def test_plain_script_with_license_header():
    # xlsxwriter: bloque de comentarios inicial con la explicación y la licencia
    src = ("#" * 78 + "\n#\n# An example of writing cell comments.\n#\n"
           "# SPDX-License-Identifier: BSD-2-Clause\n# Copyright (c) 2013-2025, John McNamara\n#\n"
           "import xlsxwriter\n\nworkbook = xlsxwriter.Workbook('comments.xlsx')\n")
    norm = normalize_gallery(src)
    assert "An example of writing cell comments." in norm.markdown
    assert "SPDX" not in norm.markdown and "Copyright" not in norm.markdown
    assert norm.block_langs == ["python"]


def test_license_only_code_block_is_skipped():
    # scikit-learn: el bloque `# Authors: … / # SPDX-License-Identifier: …` no es un ejemplo de código
    src = ('"""\nTitle\n=====\n\nIntro.\n"""\n\n# Authors: The scikit-learn developers\n'
           "# SPDX-License-Identifier: BSD-3-Clause\n\n# %%\n# Load data.\n\nX = load()\n")
    norm = normalize_gallery(src)
    assert norm.block_langs == ["python"]
    assert "SPDX" not in norm.markdown and "Authors" not in norm.markdown


# --- Docstrings -------------------------------------------------------------------------------------

RAPIDFUZZ = '''\
# SPDX-License-Identifier: MIT
def ratio(s1, s2, *, processor=None, score_cutoff=None):
    """
    Calculates the normalized Indel similarity.

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.

    Examples
    --------
    >>> fuzz.ratio("this is a test", "this is a test!")
    96.55171966552734
    """


def _private(x):
    """Not documented on the site."""


class Scorer:
    """A scorer."""

    def score(self, a):
        """Scores a."""
'''


def test_docstrings_to_rst():
    rst = docstrings_to_rst(RAPIDFUZZ, "src/rapidfuzz/fuzz_py.py")
    assert rst.startswith("rapidfuzz.fuzz\n==============")
    assert "``rapidfuzz.fuzz.ratio(s1, s2, *, processor=None, score_cutoff=None)``" in rst
    assert "_private" not in rst
    assert "``rapidfuzz.fuzz.Scorer.score(self, a)``" in rst


def test_normalize_docstrings():
    norm = normalize_docstrings(RAPIDFUZZ, "src/rapidfuzz/fuzz_py.py")
    assert norm.block_langs == ["python"]  # el ejemplo `>>>`
    assert 'fuzz.ratio("this is a test", "this is a test!")\n# 96.55171966552734' in norm.markdown
