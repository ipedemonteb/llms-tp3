"""Generador (generate/) con un índice, PyPI y LLM falsos: sin red ni API."""

from langchain_core.documents import Document

from tprag.generate.context import build_context
from tprag.generate.pipeline import apply_edits, finish, fix, prepare, resolve_package
from tprag.generate.prompt import user_message
from tprag.generate.schema import Edit, GeneratorOutput, ImportDecision
from tprag.validate.packages import normalize
from tprag.validate.pypi import PackageInfo

PYPI = {"PyPDF2": PackageInfo("PyPDF2", [], "2022-12-31", []), "pypdf": PackageInfo("pypdf", [], "2026-09-01", [])}


def fetch(name):
    return {normalize(k): v for k, v in PYPI.items()}.get(normalize(name))


def doc(i, package="pypdf", n_tokens=800):
    prefix = f"{package} > Extract text > Ejemplo {i}"
    return Document(id=f"c{i}", page_content=f"{prefix}\n\ntexto {i}",
                    metadata={"package": package, "prefix": prefix, "doc_id": "d", "start": 0, "end": 10,
                              "n_tokens": n_tokens})


class FakeStore:
    """Devuelve siempre los mismos chunks; `distance` = 1 - score del mejor."""

    def __init__(self, docs, best_score):
        self.docs, self.best = docs, best_score

    def similarity_search_with_score(self, query, k):
        return [(d, 1 - self.best + 0.01 * i) for i, d in enumerate(self.docs[:k])]


CODE = """\
import csv
from PyPDF2 import PdfReader


def pages(path):
    reader = PdfReader(path)
    return [p.extract_text() for p in reader.pages]
"""

OUTPUT = GeneratorOutput(
    imports=[ImportDecision(module="PyPDF2", script_usage="Extrae el texto de cada página.",
                            matching_fragment="pypdf > Extract text > Ejemplo 0", decision="reemplazado",
                            replacement_package="pypdf",
                            reason="PyPDF2 siguió como pypdf.", sources=["pypdf > Extract text > Ejemplo 0"])],
    edits=[Edit(antes="from PyPDF2 import PdfReader", despues="from pypdf import PdfReader")],
)


def test_apply_edits_todo_o_nada():
    assert apply_edits("a = 1\nb = 2\n", [{"antes": "a = 1", "despues": "a = 3"}]) == ("a = 3\nb = 2\n", [])
    code, errors = apply_edits("a = 1\nb = 2\n", [{"antes": "a = 1", "despues": "a = 3"},
                                                   {"antes": "c = 4", "despues": "c = 5"}])
    assert code == "a = 1\nb = 2\n" and errors == ["reemplazo 2: el fragmento no está en el script"]
    _, errors = apply_edits("x\nx\n", [{"antes": "x", "despues": "y"}])
    assert errors == ["reemplazo 1: el fragmento aparece 2 veces en el script"]


def test_contexto_hasta_el_presupuesto_sin_filtrar_por_libreria():
    store = FakeStore([doc(0), doc(1, "pdfplumber"), doc(2), doc(3)], best_score=0.6)
    ctx = build_context(store, CODE, "PyPDF2", "deprecated", threshold=0.5, budget=2000)
    assert ctx.passed and ctx.score == 0.6
    assert [c["package"] for c in ctx.chunks] == ["pypdf", "pdfplumber"]  # 800 + 800; el tercero ya no entra
    assert ctx.chunks[0]["text"] == "texto 0"  # sin el prefijo, que va como encabezado
    assert "reader = PdfReader(path)" in ctx.query


def test_bajo_el_umbral_no_llama_al_llm():
    def llm(system, user):
        raise AssertionError("no tendría que llamarse")

    result = fix(CODE, FakeStore([doc(0)], best_score=0.3), fetch, llm, threshold=0.4)
    assert not result.llm_called and result.code == CODE
    entry = next(e for e in result.report if e["module"] == "PyPDF2")
    assert entry["decision"] == "sin_reemplazo_confiable" and entry["retrieval_score"] == 0.3
    assert next(e for e in result.report if e["module"] == "csv") == {
        "module": "csv", "line": 1, "status": "stdlib", "package": None, "validator_reason": None, "warnings": []}


def test_sobre_el_umbral_aplica_los_reemplazos():
    result = fix(CODE, FakeStore([doc(0)], best_score=0.6), fetch, lambda s, u: OUTPUT, threshold=0.4)
    assert result.llm_called and "from pypdf import PdfReader" in result.code
    entry = next(e for e in result.report if e["module"] == "PyPDF2")
    assert entry["decision"] == "reemplazado" and entry["replacement_package"] == "pypdf"
    assert entry["validator_reason"] == "abandonado; el proyecto siguió como pypdf"  # al reporte sí, al LLM no
    assert entry["replacement_package_llm"] == "pypdf"
    assert entry["new_imports"] == [{"module": "pypdf", "package": "pypdf", "status": "ok"}]


def test_el_paquete_del_reporte_sale_del_codigo_y_no_del_llm():
    out = OUTPUT.model_copy(deep=True)
    out.imports[0].replacement_package = "pypdf.reader"  # un módulo, no un paquete
    result = fix(CODE, FakeStore([doc(0)], best_score=0.6), fetch, lambda s, u: out, threshold=0.4)
    entry = next(e for e in result.report if e["module"] == "PyPDF2")
    assert entry["replacement_package"] == "pypdf" and entry["replacement_package_llm"] == "pypdf.reader"


def test_resolve_package():
    google = [{"module": "google.oauth2", "package": "google-auth", "status": "ok"},
              {"module": "googleapiclient.discovery", "package": "google-api-python-client", "status": "ok"}]
    assert resolve_package("google.oauth2.service_account", google) == "google-auth"
    assert resolve_package("google-api-python-client", google) == "google-api-python-client"
    assert resolve_package("otro", google) is None  # dos paquetes nuevos y ninguno coincide
    stdlib = [{"module": "zoneinfo", "package": "zoneinfo", "status": "stdlib"}]
    assert resolve_package("python-stdlib", stdlib) == "zoneinfo"  # un solo paquete nuevo
    assert resolve_package("pypdf", []) is None  # no quedó ningún import nuevo en el código


def test_el_llm_se_abstiene_y_no_se_toca_el_codigo():
    out = GeneratorOutput(imports=[ImportDecision(module="PyPDF2", script_usage="Extrae el texto.",
                                                  matching_fragment=None, decision="sin_reemplazo_confiable",
                                                  replacement_package=None, reason="No resuelve lo mismo.",
                                                  sources=[])],
                          edits=[Edit(antes="import csv", despues="import json")])
    prep = prepare(CODE, FakeStore([doc(0)], best_score=0.6), fetch, threshold=0.4)
    result = finish(prep, out)
    assert result.code == CODE  # sin ningún "reemplazado", los reemplazos no se aplican


def test_el_mensaje_no_incluye_el_reemplazo_del_validador():
    prep = prepare(CODE, FakeStore([doc(0)], best_score=0.6), fetch, threshold=0.4)
    assert "pypdf > Extract text > Ejemplo 0" in prep.user  # el fragmento sí
    assert "siguió como pypdf" not in prep.user  # el motivo de la lista manual, no
    assert prep.user == user_message(CODE, prep.contexts)


class FakeScorer:
    """Puntaje por texto del chunk; registra los pares que recibe."""

    def __init__(self, by_text):
        self.by_text, self.pairs = by_text, []

    def score(self, pairs):
        self.pairs += pairs
        return [self.by_text[d] for _, d in pairs]


def test_reranker_reordena_los_k_primeros_y_usa_su_maximo_como_confianza():
    from tprag.generate.context import Reranker

    docs = [doc(0, "pdfplumber"), doc(1), doc(2, "pymupdf"), doc(3)]
    scorer = FakeScorer({d.page_content: s for d, s in zip(docs, [-6.0, -1.0, -6.0, 9.0])})
    store = FakeStore(docs, best_score=0.2)  # el coseno no pasaría un umbral de 0,4: ahora decide el reranker
    ctx = build_context(store, CODE, "PyPDF2", "deprecated", threshold=-4.5, budget=2000,
                        reranker=Reranker("k1-rr3", scorer, 3))
    assert len(scorer.pairs) == 3 and all(q == ctx.query for q, _ in scorer.pairs)  # solo los k primeros
    assert scorer.pairs[0][1] == docs[0].page_content  # el chunk con prefijo, como en el estudio
    assert ctx.passed and ctx.score == -1.0 and ctx.reranker == "k1-rr3"
    assert [h["id"] for h in ctx.hits] == ["c1", "c0", "c2", "c3"]  # empate (c0, c2): orden denso; c3 queda detrás
    assert [h.get("rerank_score") for h in ctx.hits] == [-1.0, -6.0, -6.0, None]
    assert [c["id"] for c in ctx.chunks] == ["c1", "c0"]  # presupuesto sobre el orden nuevo


def test_reranker_bajo_su_umbral_no_llama_al_llm():
    from tprag.generate.context import Reranker

    def llm(system, user):
        raise AssertionError("no tendría que llamarse")

    docs = [doc(0), doc(1)]
    scorer = FakeScorer({d.page_content: -8.0 for d in docs})
    result = fix(CODE, FakeStore(docs, best_score=0.9), fetch, llm, threshold=-4.5,
                 reranker=Reranker("k1-rr30", scorer, 30))
    entry = next(e for e in result.report if e["module"] == "PyPDF2")
    assert not result.llm_called and entry["retrieval_score"] == -8.0 and entry["reranker"] == "k1-rr30"


def test_load_reranker_y_umbral(tmp_path, monkeypatch):
    import json

    import pytest

    from tprag.generate import context

    assert context.load_reranker("k1-rr30").k == 30
    with pytest.raises(ValueError):
        context.load_reranker("k1-rrf30")  # la fusión no se integra
    monkeypatch.setattr(context, "STUDY_DIR", tmp_path)
    row = {"collection": "col", "split": "dev", "variants": {"k1-rr30": {"threshold_rr": {"value": -4.5}}}}
    (tmp_path / "rerank_a_analysis.json").write_text(json.dumps(row))
    (tmp_path / "rerank_b_analysis.json").write_text(json.dumps(row))
    assert context.rerank_threshold("col", "k1-rr30") == -4.5
    row["variants"]["k1-rr30"]["threshold_rr"]["value"] = -3.0
    (tmp_path / "rerank_b_analysis.json").write_text(json.dumps(row))
    with pytest.raises(ValueError):
        context.rerank_threshold("col", "k1-rr30")  # dos análisis que no coinciden
