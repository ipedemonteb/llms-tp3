"""Generador (generate/) con un índice, PyPI y LLM falsos: sin red ni API."""

from langchain_core.documents import Document

from tprag.generate.context import build_context
from tprag.generate.llm import TruncatedOutput
from tprag.generate.pipeline import (answer, apply_edits, finish, first_prompt, fix, fix_problems, prepare,
                                     resolve_package,
                                     retry_prompt)
from tprag.generate.prompt import user_message
from tprag.generate.prompt import SYSTEMS
from tprag.generate.schema import Edit, GeneratorOutput, ImportDecision, ScriptOutput
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


def test_respuesta_cortada_por_el_tope_deja_el_script_igual():
    def llm(system, user):
        raise TruncatedOutput('{"imports": [{"module": "PyPDF2", "reason": "\\\\\\\\\\\\')

    result = fix(CODE, FakeStore([doc(0)], best_score=0.6), fetch, llm, threshold=0.4)
    assert result.llm_called and result.code == CODE and not result.edits
    assert result.llm_error == "length" and result.llm_raw.startswith('{"imports"')
    entry = next(e for e in result.report if e["module"] == "PyPDF2")
    assert entry["decision"] == "sin_reemplazo_confiable"
    assert entry["reason"] == "la respuesta del LLM se cortó por superar el tope de tokens de salida"


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


# --- Iteración: reintento con los problemas y abstención (iteracion_ej4.md §3.2) ---------------------------------

def with_edits(*edits):
    out = OUTPUT.model_copy(deep=True)
    out.edits = [Edit(antes=a, despues=d) for a, d in edits]
    return out


BAD = with_edits(("from PyPDF2 import PdfReader\nimport csv", "from pypdf import PdfReader"))  # el "antes" no está
GOOD = OUTPUT


def scripted(*outputs):
    """LLM falso que devuelve las respuestas en orden y guarda los mensajes que recibió."""
    calls = []

    def llm(system, user):
        calls.append(user)
        return outputs[len(calls) - 1]

    return llm, calls


def run(llm, retry, retry_llm=None):
    return answer(prepare(CODE, FakeStore([doc(0)], best_score=0.6), fetch, threshold=0.4), llm, retry=retry,
                  retry_llm=retry_llm)


def test_sin_retry_no_se_reintenta_aunque_haya_problemas():
    llm, calls = scripted(BAD, GOOD)
    result = run(llm, retry=None)
    assert len(calls) == 1 and result.code == CODE and result.edit_errors and result.retry is None


def test_retry_con_respuesta_buena_no_reintenta():
    llm, calls = scripted(GOOD)
    result = run(llm, retry="edits")
    assert len(calls) == 1 and "from pypdf import PdfReader" in result.code and result.retry is None


def test_retry_corrige_en_el_segundo_intento():
    llm, calls = scripted(BAD, GOOD)
    result = run(llm, retry="edits")
    assert len(calls) == 2 and "from pypdf import PdfReader" in result.code and not result.edit_errors
    assert "## Tu respuesta anterior" in calls[1] and "## Problemas de tu respuesta anterior" in calls[1]
    assert "Reemplazo 1: el \"antes\" no está en el script tal cual" in calls[1]
    assert calls[1].startswith(calls[0].rstrip("\n"))  # el mismo mensaje de la primera vez, más el reintento
    assert result.retry["remaining"] == [] and len(result.retry["problems"]) == 1
    assert result.retry["first_edits"][0]["antes"].endswith("import csv")


def test_retry_que_sigue_mal_se_abstiene():
    llm, calls = scripted(BAD, BAD)
    result = run(llm, retry="edits")
    assert len(calls) == 2 and result.code == CODE and result.llm_called and result.retry["remaining"]
    entry = next(e for e in result.report if e["module"] == "PyPDF2")
    assert entry["decision"] == "sin_reemplazo_confiable"
    assert entry["reason"].startswith("la corrección no pasó la post-validación después de un reintento")


def test_problemas_nombres_sin_definir_y_codigo_invalido():
    prep = prepare(CODE, FakeStore([doc(0)], best_score=0.6), fetch, threshold=0.4)
    undefined = finish(prep, with_edits(("from PyPDF2 import PdfReader", "import pypdf")))
    assert fix_problems(prep, undefined) == [
        "El script corregido usa nombres que no están definidos ni importados: `PdfReader`. Migrá también esos usos a "
        "la librería nueva, con la API que aparece en los fragmentos."]
    broken = finish(prep, with_edits(("    reader = PdfReader(path)", "    reader = PdfReader(path")))
    [problem] = fix_problems(prep, broken)
    assert problem.startswith("El script corregido no es Python válido: línea ")
    repeated = finish(prep, with_edits(("reader", "lector")))
    assert fix_problems(prep, repeated) == [
        "Reemplazo 1: el \"antes\" aparece 2 veces en el script. Agregá líneas vecinas para que aparezca una sola vez."]


def test_si_el_llm_se_abstiene_no_hay_problemas_que_reintentar():
    out = GeneratorOutput(imports=[ImportDecision(module="PyPDF2", script_usage="Extrae el texto.",
                                                  matching_fragment=None, decision="sin_reemplazo_confiable",
                                                  replacement_package=None, reason="No resuelve lo mismo.",
                                                  sources=[])],
                          edits=[Edit(antes="no está", despues="x")])
    prep = prepare(CODE, FakeStore([doc(0)], best_score=0.6), fetch, threshold=0.4)
    assert fix_problems(prep, finish(prep, out)) == [] and retry_prompt(prep, finish(prep, out)) is None


FIXED = CODE.replace("from PyPDF2 import PdfReader", "from pypdf import PdfReader")


def script_llm(*scripts):
    """LLM falso en formato "script" (`ScriptOutput`): devuelve los scripts en orden y guarda (sistema, mensaje)."""
    calls = []

    def llm(system, user):
        calls.append((system, user))
        return ScriptOutput(imports=OUTPUT.imports, script=scripts[len(calls) - 1])

    return llm, calls


def test_retry_script_usa_el_script_completo_y_las_lineas_del_import():
    llm, calls = scripted(BAD)
    retry_llm, retry_calls = script_llm(FIXED.rstrip("\n"))  # sin el salto final: se respeta el del original
    result = run(llm, retry="script", retry_llm=retry_llm)
    assert len(calls) == 1 and len(retry_calls) == 1 and result.code == FIXED and not result.edit_errors
    system, user = retry_calls[0]
    assert system == SYSTEMS["script"] and system != SYSTEMS["edits"] and 'en "script"' in system
    assert "## Líneas que usan el import a reemplazar" in user
    assert "- línea 2: from PyPDF2 import PdfReader" in user and "- línea 6: reader = PdfReader(path)" in user
    assert 'Tu reemplazo 1 no se pudo aplicar: el "antes" no está en el script tal cual.' in user
    assert "Esta vez devolvé el script completo corregido" in user
    assert result.retry["format"] == "script" and result.retry["remaining"] == [] and result.retry["changed_lines"] == 1


def test_retry_script_que_sigue_mal_se_abstiene():
    llm, _ = scripted(BAD)
    retry_llm, _ = script_llm(CODE.replace("from PyPDF2 import PdfReader", "import pypdf"))  # PdfReader sin definir
    result = run(llm, retry="script", retry_llm=retry_llm)
    assert result.code == CODE and result.retry["remaining"] and "changed_lines" not in result.retry


def test_retry_edits_no_cambia_el_mensaje_de_siempre():
    llm, calls = scripted(BAD, GOOD)
    run(llm, retry="edits")
    assert "Líneas que usan el import" not in calls[1] and "Esta vez devolvé el script completo" not in calls[1]


def test_script_igual_al_original_salvo_el_salto_final_es_el_original():
    prep = prepare(CODE, FakeStore([doc(0)], best_score=0.6), fetch, threshold=0.4)
    assert finish(prep, ScriptOutput(imports=OUTPUT.imports, script=CODE.rstrip("\n"))).code == CODE


# --- Variante uniforme: el script completo también en el primer intento -------------------------------------------

def test_uniforme_primer_intento_script_sin_problemas():
    llm, calls = script_llm(FIXED)
    result = answer(prepare(CODE, FakeStore([doc(0)], best_score=0.6), fetch, threshold=0.4), llm, retry="script",
                    retry_llm=llm, first="script")
    assert len(calls) == 1 and calls[0][0] == SYSTEMS["script"] and result.code == FIXED
    assert result.retry is None and result.changed_lines == 1 and result.output_format == "script"


def test_uniforme_reintenta_con_el_script_anterior_y_si_no_cambia_es_un_problema():
    llm, calls = script_llm(CODE, FIXED)  # primero dice "reemplazado" y devuelve el original; después lo corrige
    result = answer(prepare(CODE, FakeStore([doc(0)], best_score=0.6), fetch, threshold=0.4), llm, retry="script",
                    retry_llm=llm, first="script")
    assert len(calls) == 2 and result.code == FIXED
    assert result.retry["problems"] == ["Decidiste reemplazar al menos un import, pero devolviste el script sin cambios."]
    user = calls[1][1]
    assert "## Tu respuesta anterior\n\n```python\nimport csv" in user and "--- reemplazo" not in user


def test_sin_uniforme_el_script_sin_cambios_no_es_un_problema():
    prep = prepare(CODE, FakeStore([doc(0)], best_score=0.6), fetch, threshold=0.4)
    unchanged = finish(prep, ScriptOutput(imports=OUTPUT.imports, script=CODE))
    assert fix_problems(prep, unchanged, "script") == []  # así se comportó __iter2: no cambia
    assert fix_problems(prep, unchanged, "script", unchanged=True)


def test_sin_iteracion_el_primer_intento_es_el_de_siempre():
    llm, calls = scripted(GOOD)
    prep = prepare(CODE, FakeStore([doc(0)], best_score=0.6), fetch, threshold=0.4)
    result = answer(prep, llm)
    assert calls == [prep.prompt[1]] and first_prompt(prep) == prep.prompt and result.changed_lines is None


# --- Variante uniforme + chequeo de contexto (API fuera de los fragmentos, `__iter4`) ------------------------------

INVENTED = CODE.replace("from PyPDF2 import PdfReader", "from pypdf import PdfReader, inventado").replace(
    "    reader = PdfReader(path)", "    reader = PdfReader(path)\n    inventado(reader)")


def doc_with_api():
    """Un fragmento de pypdf que sí muestra `PdfReader` (el de `doc` no muestra ninguna API)."""
    d = doc(0)
    d.page_content = d.metadata["prefix"] + "\n\nfrom pypdf import PdfReader\nreader = PdfReader(path)"
    return d


def test_context_check_reintenta_si_usa_api_fuera_de_los_fragmentos():
    llm, calls = script_llm(INVENTED, FIXED)
    result = answer(prepare(CODE, FakeStore([doc_with_api()], best_score=0.6), fetch, threshold=0.4), llm,
                    retry="script", retry_llm=llm, first="script", context_check=True)
    assert len(calls) == 2 and result.code == FIXED  # el reintento solo usa PdfReader, que está en el fragmento
    [problem] = result.retry["problems"]
    assert problem.startswith("El script corregido usa nombres de la librería nueva que no aparecen en los fragmentos")
    assert "`inventado`" in problem and "`PdfReader`" not in problem


def test_context_check_que_sigue_fuera_de_los_fragmentos_se_abstiene():
    llm, _ = script_llm(INVENTED, INVENTED)
    result = answer(prepare(CODE, FakeStore([doc_with_api()], best_score=0.6), fetch, threshold=0.4), llm,
                    retry="script", retry_llm=llm, first="script", context_check=True)
    assert result.code == CODE and result.retry["remaining"]


def test_sin_context_check_la_api_fuera_de_los_fragmentos_no_es_un_problema():
    llm, calls = script_llm(INVENTED)
    result = answer(prepare(CODE, FakeStore([doc(0)], best_score=0.6), fetch, threshold=0.4), llm, retry="script",
                    retry_llm=llm, first="script")
    assert len(calls) == 1 and result.code == INVENTED and result.retry is None
