"""Consulta A del paso 7 (eval/query.py) e idioma del ítem (eval/retrieval.py), sin API."""

import pytest

from tprag.eval.query import query_a
from tprag.eval.retrieval import item_lang

CODE = """\
import csv
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError


def extract(path):
    reader = PdfReader(path)  # abre el archivo
    rows = []
    for page in reader.pages:
        rows.append(page.extract_text())
    try:
        pass
    except PdfReadError:
        return []
    return rows
"""


def test_query_a_junta_los_imports_del_modulo_y_saca_la_sangria():
    assert query_a(CODE, "PyPDF2").split("\n") == [
        "from PyPDF2 import PdfReader",
        "from PyPDF2.errors import PdfReadError",
        "reader = PdfReader(path)  # abre el archivo",  # sin limpieza: el comentario queda
        "for page in reader.pages:",
        "rows.append(page.extract_text())",
        "except PdfReadError:",
    ]


def test_query_a_no_mezcla_otros_modulos():
    assert query_a(CODE, "csv") == "import csv"


def test_query_a_modulo_que_no_esta():
    with pytest.raises(ValueError):
        query_a(CODE, "pypdf")


def test_item_lang():
    assert item_lang('"""Lee los PDFs de una carpeta y devuelve el texto."""\nx = 1  # página por página\n') == "es"
    assert item_lang('"""Read the PDFs in a folder and return the text."""\nx = 1  # one page at a time\n') == "en"
