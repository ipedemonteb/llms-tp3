"""Extracción de imports (validate/imports.py)."""

import pytest

from tprag.validate.imports import extract_imports

SCRIPT = """\
import numpy as np
import os, google.oauth2.service_account
from PyPDF2 import PdfReader, PdfWriter as W
from . import local
from .utils import helper

def f():
    import yaml
    return yaml.safe_load(np.array([1]))

try:
    from fastjsonparser import *
except ImportError:
    pass

r = PdfReader("a.pdf")
w = W()
creds = google.oauth2.service_account.Credentials
"""


def test_extracts_modules_names_and_lines():
    refs = {r.module: r for r in extract_imports(SCRIPT)}
    assert list(refs) == ["numpy", "os", "google.oauth2.service_account", "PyPDF2", "yaml", "fastjsonparser"]
    assert refs["numpy"].names == ["np"]
    assert refs["google.oauth2.service_account"].names == ["google"]  # `import a.b` deja definido `a`
    assert refs["PyPDF2"].names == ["PdfReader", "W"] and refs["PyPDF2"].line == 3
    assert refs["yaml"].line == 8  # dentro de una función
    assert refs["fastjsonparser"].names == ["*"]  # dentro de un try


def test_relative_imports_are_ignored():
    assert not {"local", "utils", ""} & {r.module for r in extract_imports(SCRIPT)}


def test_usages():
    refs = {r.module: r for r in extract_imports(SCRIPT)}
    assert refs["numpy"].usages == [9]
    assert refs["PyPDF2"].usages == [16, 17]
    assert refs["google.oauth2.service_account"].usages == [18]
    assert refs["fastjsonparser"].usages == []


def context(source, module):
    """Texto de las líneas A2 de un import (más fácil de leer que los números)."""
    lines = source.splitlines()
    ref = next(r for r in extract_imports(source) if r.module == module)
    return [lines[i - 1].strip() for i in ref.context_lines]


PYPDF2 = """\
import pandas as pd
from PyPDF2 import PdfReader

def extract_rows(pdf_path):
    reader = PdfReader(pdf_path)
    rows = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        rows.append({"page": i, "chars": len(text)})
    df = pd.DataFrame(rows)
    return df
"""


def test_a2_ejemplo_pypdf2():  # dataset_eval.md §2.2
    assert context(PYPDF2, "PyPDF2") == [
        "from PyPDF2 import PdfReader",
        "reader = PdfReader(pdf_path)",
        "for i, page in enumerate(reader.pages):",
        "text = page.extract_text()",
        'rows.append({"page": i, "chars": len(text)})',  # `i` sale del mismo `for` (nivel 2)
    ]


def test_a2_no_incluye_el_cuerpo_ni_otras_librerias():
    ctx = context(PYPDF2, "PyPDF2")
    assert "df = pd.DataFrame(rows)" not in ctx and "rows = []" not in ctx


def test_a2_corta_en_el_nivel_3():
    source = """\
import lib
a = lib.make()
b = a.child()
c = b.value
print(c)
"""
    assert context(source, "lib") == ["import lib", "a = lib.make()", "b = a.child()", "c = b.value"]


def test_a2_with_alias_y_desempaquetado():
    source = """\
import numpy
from fuzzywuzzy import process as p
with p.session() as s:
    pass
best, score = p.extractOne("x", s.choices)
if score > 80:
    log(best)
numpy.zeros(3)
"""
    assert context(source, "fuzzywuzzy") == [
        "from fuzzywuzzy import process as p",
        "with p.session() as s:",
        'best, score = p.extractOne("x", s.choices)',
        "if score > 80:",
        "log(best)",
    ]


def test_a2_sentencia_de_varias_lineas_y_except():
    source = """\
from fuzzywuzzy import fuzz, process
try:
    res = process.extractOne(q, names, scorer=fuzz.token_set_ratio,
                             score_cutoff=85)
except process.SomeError as exc:
    handle(exc)
    other()
"""
    assert context(source, "fuzzywuzzy") == [
        "from fuzzywuzzy import fuzz, process",
        "res = process.extractOne(q, names, scorer=fuzz.token_set_ratio,",
        "score_cutoff=85)",
        "except process.SomeError as exc:",
    ]


def test_a2_decorador():
    source = """\
import mock

@mock.patch("requests.get")
def test_x(get):
    assert get.called
"""
    assert context(source, "mock") == ["import mock", '@mock.patch("requests.get")']


def test_syntax_error():
    with pytest.raises(SyntaxError):
        extract_imports("import numpy as\n")
