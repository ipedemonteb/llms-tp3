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


def test_syntax_error():
    with pytest.raises(SyntaxError):
        extract_imports("import numpy as\n")
