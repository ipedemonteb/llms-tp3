"""Chequeos del nivel 1 de la evaluación de la generación (eval/generation.py), sin red ni API."""

from langchain_core.documents import Document

from tprag.eval.generation import acceptable_import, check_item, summarize
from tprag.generate.pipeline import FixResult, prepare
from tprag.validate.packages import normalize
from tprag.validate.pypi import PackageInfo
from tprag.validate.validate import ImportResult

PYPI = {"PyPDF2": PackageInfo("PyPDF2", [], "2022-12-31", []), "pypdf": PackageInfo("pypdf", [], "2026-09-01", [])}


def fetch(name):
    return {normalize(k): v for k, v in PYPI.items()}.get(normalize(name))


class FakeStore:
    def similarity_search_with_score(self, query, k):
        prefix = "pypdf > Extract text > Ejemplo"
        return [(Document(id="c0", page_content=f"{prefix}\n\ntexto",
                          metadata={"package": "pypdf", "prefix": prefix, "doc_id": "d", "start": 0, "end": 10,
                                    "n_tokens": 100}), 0.4)]


CODE = "from PyPDF2 import PdfReader\n\nreader = PdfReader('a.pdf')\n"
FIXED = "from pypdf import PdfReader\n\nreader = PdfReader('a.pdf')\n"


def item(acceptable):
    return {"id": "x", "subset": "manual_deprecated" if acceptable else "negative", "split": "dev", "code": CODE,
            "targets": [{"module": "PyPDF2", "acceptable_packages": acceptable}]}


def entry(decision, package=None, sources=()):
    return {"module": "PyPDF2", "status": "deprecated", "decision": decision, "replacement_package": package,
            "sources": list(sources)}


def run(acceptable, code, decision, package=None, sources=(), errors=()):
    prep = prepare(CODE, FakeStore(), fetch, threshold=0.5)
    fix = FixResult(code, [entry(decision, package, sources)], True, [], list(errors))
    return check_item(item(acceptable), prep, fix, fetch)


def test_positivo_bien_corregido():
    r = run(["pypdf"], FIXED, "reemplazado", "pypdf", ["--- [1] pypdf > Extract text > Ejemplo"])
    assert r["success"] and r["sources_ok"] and r["targets"][0]["package_ok"] and r["targets"][0]["target_gone"]


def test_positivo_con_paquete_inventado():
    r = run(["pypdf"], FIXED.replace("pypdf", "pypdfx"), "reemplazado", "pypdfx")
    assert not r["success"] and r["hallucinated_imports"] == ["pypdfx"]


def test_positivo_que_no_parsea_o_no_se_aplico():
    assert not run(["pypdf"], "from pypdf import (", "reemplazado")["parses"]
    r = run(["pypdf"], CODE, "reemplazado", errors=["reemplazo 1: el fragmento no está en el script"])
    assert not r["success"] and not r["applicable"] and not r["targets"][0]["target_gone"]


def test_fuente_inventada():
    assert not run(["pypdf"], FIXED, "reemplazado", "pypdf", ["pypdf > Otra página"])["sources_ok"]


def test_negativo_se_abstiene_sin_tocar_el_codigo():
    assert run([], CODE, "sin_reemplazo_confiable")["success"]
    assert not run([], FIXED, "reemplazado", "pypdf")["success"]


def test_paquete_aceptable_de_la_stdlib_por_modulo():
    def stdlib(module, names):
        return ImportResult(module, 1, names, [], "stdlib")

    acceptable = ["python-stdlib", "unittest.mock"]
    assert acceptable_import(stdlib("unittest", ["mock"]), acceptable)
    assert acceptable_import(stdlib("unittest.mock", ["patch"]), acceptable)
    assert not acceptable_import(stdlib("tomllib", ["tomllib"]), acceptable)


def test_summarize():
    records = [run(["pypdf"], FIXED, "reemplazado", "pypdf"), run([], CODE, "sin_reemplazo_confiable")]
    out = summarize(records)
    assert out["positives"]["success"] == 1.0 and out["negatives"]["success"] == 1.0
    # score 0,6 (distancia 0,4) contra umbral 0,5: el negativo llega al LLM y el LLM lo rechaza
    assert out["negatives"]["passed_threshold"] == 1 and out["negatives"]["rejected_by_llm"] == 1
