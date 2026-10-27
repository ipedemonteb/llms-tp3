"""Consulta a PyPI con caché (validate/pypi.py). Sin red: la consulta es una función falsa."""

import json
from pathlib import Path

import pytest

from tprag.validate.pypi import TTL_FOUND, TTL_MISSING, CachedPyPI, PackageInfo, PyPIError, parse

FIXTURES = Path(__file__).parent / "fixtures"
DAY = 86400


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"pypi_{name}.json").read_text(encoding="utf-8"))


def test_parse_real_responses():
    oauth = parse(load("oauth2client"))
    assert oauth.name == "oauth2client" and oauth.last_release == "2018-09-07"
    assert "Development Status :: 7 - Inactive" in oauth.classifiers
    assert len(parse(load("pycrypto")).vulnerabilities) == 4
    assert parse(load("six")).last_release == "2024-12-04"


class Fake:
    def __init__(self, known: dict[str, PackageInfo | None]):
        self.known, self.calls = known, []

    def __call__(self, name):
        self.calls.append(name)
        if name not in self.known:
            raise PyPIError("sin red")
        return self.known[name]


def test_cache_and_ttls(tmp_path):
    clock = [0.0]
    fake = Fake({"six": PackageInfo("six", [], "2024-12-04", []), "fakelib": None})
    pypi = CachedPyPI(fake, tmp_path, now=lambda: clock[0])
    assert pypi("six").name == "six" and pypi("fakelib") is None
    clock[0] = TTL_MISSING - 1
    pypi("Six"), pypi("fakelib")  # mismo nombre normalizado: del caché
    assert fake.calls == ["six", "fakelib"]
    clock[0] = TTL_MISSING + 1  # el 404 vence a los 1 día, el que existe no
    pypi("six"), pypi("fakelib")
    assert fake.calls == ["six", "fakelib", "fakelib"]
    clock[0] = TTL_FOUND + 1
    pypi("six")
    assert fake.calls[-1] == "six"


def test_offline_uses_only_cache(tmp_path):
    CachedPyPI(Fake({"six": PackageInfo("six", [], None, [])}), tmp_path)("six")
    offline = CachedPyPI(Fake({}), tmp_path, offline=True)
    assert offline("six").name == "six"
    with pytest.raises(PyPIError):
        offline("numpy")


def test_errors_are_not_cached(tmp_path):
    fake = Fake({})
    pypi = CachedPyPI(fake, tmp_path)
    for _ in range(2):
        with pytest.raises(PyPIError):
            pypi("numpy")
    assert fake.calls == ["numpy", "numpy"]
