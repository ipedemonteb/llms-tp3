"""Validación completa (validate/validate.py) con una consulta a PyPI falsa: sin red."""

from datetime import date

from tprag.validate.packages import normalize
from tprag.validate.pypi import PackageInfo, PyPIError
from tprag.validate.validate import validate

TODAY = date(2026, 10, 27)
PYPI = {
    "numpy": PackageInfo("numpy", [], "2026-09-01", []),
    "scikit-learn": PackageInfo("scikit-learn", [], "2026-07-01", []),
    "PyPDF2": PackageInfo("PyPDF2", [], "2022-12-31", ["PYSEC-2026-1835"]),
    "oauth2client": PackageInfo("oauth2client", ["Development Status :: 7 - Inactive"], "2018-09-07", []),
    "six": PackageInfo("six", [], "2024-12-04", []),
    "python-docx": PackageInfo("python-docx", [], "2025-01-01", []),
    "fastjsonparser": None, "python-fastjsonparser": None,
}


def fetch(name):
    """Como PyPI: el nombre se normaliza (pipreqs da `Requests`)."""
    if normalize(name) == "requests":
        raise PyPIError("sin red")
    return {normalize(k): v for k, v in PYPI.items()}[normalize(name)]


SCRIPT = """\
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from PyPDF2 import PdfReader
from oauth2client.client import GoogleCredentials
import six
import fastjsonparser
import requests
import docx

reader = PdfReader("a.pdf")
data = fastjsonparser.loads(reader.pages[0].extract_text())
"""


def results():
    return {r.module: r for r in validate(SCRIPT, fetch, today=TODAY)}


def test_statuses():
    r = results()
    assert {m: x.status for m, x in r.items()} == {
        "os": "stdlib", "numpy": "ok", "sklearn.linear_model": "ok", "PyPDF2": "deprecated",
        "oauth2client.client": "deprecated", "six": "ok", "fastjsonparser": "not_found", "requests": "unknown",
        "docx": "ok",
    }
    assert [m for m, x in r.items() if x.to_rag] == ["PyPDF2", "oauth2client.client", "fastjsonparser"]


def test_trap_package_resolves_to_the_real_one():
    # `import sklearn` es correcto: el paquete es scikit-learn (nunca se consulta el `sklearn` vacío de PyPI)
    r = results()["sklearn.linear_model"]
    assert (r.package, r.resolved_by) == ("scikit-learn", "manual")


def test_deprecated_details():
    r = results()
    assert r["PyPDF2"].replacement == ["pypdf"] and r["PyPDF2"].usages == [11]
    assert r["oauth2client.client"].replacement == ["google-auth"]  # está en la lista; el clasificador también
    assert r["fastjsonparser"].usages == [12] and r["fastjsonparser"].resolved_by == "variants"


def test_inactive_classifier_without_manual_entry():
    pypi = {"oldlib": PackageInfo("oldlib", ["Development Status :: 7 - Inactive"], "2026-01-01", [])}
    [r] = validate("import oldlib\n", pypi.get, today=TODAY)
    assert r.status == "deprecated" and r.replacement is None and "inactivo" in r.reason


def test_warnings_do_not_change_status():
    r = results()
    assert r["six"].warnings == []  # release en 2024
    codes = {w["code"] for w in r["PyPDF2"].warnings}
    assert codes == {"no_recent_release", "known_vulnerabilities"}  # 2022-12-31 > 3 años antes de TODAY
    assert [w["code"] for w in validate("import six\n", fetch, years=1, today=TODAY)[0].warnings] == [
        "no_recent_release"]


def test_variants_find_python_prefix():
    pypi = {"foo": None, "python-foo": PackageInfo("python-foo", [], "2026-01-01", [])}
    [r] = validate("import foo\n", pypi.get, today=TODAY)
    assert (r.status, r.package) == ("ok", "python-foo")


def test_each_module_is_checked_once():
    calls = []

    def counting(name):
        calls.append(name)
        return fetch(name)

    validate("import numpy\nimport numpy as np\n", counting, today=TODAY)
    assert calls == ["numpy"]
