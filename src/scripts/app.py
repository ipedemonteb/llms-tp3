"""Interfaz web del corrector (extra de la consigna). El mismo pipeline que `fix_script.py`, mostrando cada etapa:
validador → retrieval (+ reranker) → umbral → LLM → post-validación, con tiempos, tokens y USD. El generador se elige
entre el iterado del Ej. 4 (gpt-4.1-mini, script completo, reintento y abstención) y el de antes de iterar.

Uso (desde la raíz del repo):
    uv run --group local --group ui streamlit run src/scripts/app.py

Necesita el índice `s3-1024__e4__r1__cap150` en data/index/ (build_index.py) y, para el paso 2, `OPENAI_API_KEY` en
.env (el paso 1 no usa la API). La primera vez descarga el embedder jina-code y, si se elige el reranker, Qwen3.
"""

import hashlib
import os

import streamlit as st
from dotenv import load_dotenv

from tprag.split.store import ROOT
from tprag.ui.engine import GENERATORS, ITERATED, NOT_ITERATED, VERSIONS, Engine
from tprag.ui.view import (decision_rows, fixed_name, import_rows, load_examples, report_json, retry_note, target_rows,
                           unified_diff)

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Corrector de dependencias · TP3 Grupo 7", page_icon="🐍", layout="wide")

VERSION_HELP = {"base": "Orden del índice denso (coseno). Rápido.",
                "k1-rr30": "Qwen3-Reranker-0.6B reordena los 30 primeros (~16 s por import en una Mac M3; más en CPU)."}
GENERATOR_LABEL = {ITERATED: "Iterado (Ej. 4)", NOT_ITERATED: "Sin iteración"}
GENERATOR_HELP = {ITERATED: "gpt-4.1-mini devuelve el script completo; si la corrección no parsea o deja nombres sin "
                            "definir, un reintento y, si sigue mal, abstención.",
                  NOT_ITERATED: "gpt-4o-mini devuelve una lista de reemplazos; sin reintento."}


@st.cache_resource(show_spinner="Abriendo el índice…")
def engine() -> Engine:
    return Engine()


@st.cache_data
def examples() -> dict[str, dict]:
    return load_examples(ROOT / "data" / "eval")


def seconds(s: float) -> str:
    return f"{s:.2f} s".replace(".", ",")


def sidebar(eng: Engine) -> tuple[str, str, float, int]:
    st.sidebar.header("Configuración")
    version = st.sidebar.radio("Retrieval", list(VERSIONS), format_func=lambda v: "Base" if v == "base" else
                               "Reranker k1-rr30", captions=[VERSION_HELP[v] for v in VERSIONS])
    generator = st.sidebar.radio("Generador", list(GENERATORS), format_func=GENERATOR_LABEL.get,
                                 captions=[GENERATOR_HELP[g] for g in GENERATORS])
    calibrated = eng.threshold(version)
    custom = st.sidebar.toggle("Cambiar el umbral", help="Por defecto, el calibrado en dev (pasa ≥ 95% de los "
                                                          "positivos), el mismo que usa fix_script.py.")
    threshold = st.sidebar.number_input("Umbral de confianza", value=calibrated, format="%.4f", disabled=not custom)
    budget = st.sidebar.number_input("Tokens de contexto por import", 500, 8000, 2000, step=500)
    st.sidebar.divider()
    st.sidebar.caption(f"Índice `{eng.collection}` · {eng.chunks():,} casos de uso".replace(",", "."))
    st.sidebar.caption(f"Umbral calibrado de esta versión: {calibrated:.4f}".replace(".", ","))
    st.sidebar.caption("API de OpenAI: " + ("configurada" if os.environ.get("OPENAI_API_KEY") else
                                           "**sin `OPENAI_API_KEY`** (el paso 2 solo anda con respuestas en caché)"))
    return version, generator, float(threshold if custom else calibrated), int(budget)


def script_input() -> tuple[str, str] | None:
    """(nombre, código) elegido por la persona, o None."""
    tab_example, tab_upload, tab_paste = st.tabs(["Ejemplo del dataset", "Subir un .py", "Pegar código"])
    with tab_example:
        ex = examples()
        key = st.selectbox("Script", sorted(ex), format_func=lambda k: f"{k} · {ex[k]['task']}")
        use_example = st.button("Usar este ejemplo")
    with tab_upload:
        uploaded = st.file_uploader("Archivo de Python", type=["py"])
    with tab_paste:
        pasted = st.text_area("Código", height=220, placeholder="import PyPDF2\n...")
        use_pasted = st.button("Usar este código")
    if use_example:
        st.session_state.script = (f"{key}.py", ex[key]["code"])
    elif uploaded is not None and st.session_state.get("uploaded") != uploaded.file_id:
        st.session_state.uploaded = uploaded.file_id
        st.session_state.script = (uploaded.name, uploaded.getvalue().decode("utf-8", errors="replace"))
    elif use_pasted and pasted.strip():
        st.session_state.script = ("script.py", pasted)
    return st.session_state.get("script")


def show_analysis(analysis) -> None:
    prep, sec = analysis.prepared, analysis.seconds
    cols = st.columns(4)
    cols[0].metric("Validador", seconds(sec["validador"]))
    cols[1].metric("Retrieval", seconds(sec["retrieval"]))
    cols[2].metric("Reranker", seconds(sec["reranker"]) if analysis.version != "base" else "-")
    cols[3].metric("Total sin LLM", seconds(sec["total"]))
    st.caption("La primera consulta que no está en el caché de embeddings incluye cargar el modelo (y el reranker, la "
               "primera vez que se usa).")

    st.markdown("**Imports del script** (validador: AST + API de PyPI)")
    st.dataframe(import_rows(prep.results), hide_index=True, width="stretch")
    if not prep.contexts:
        st.success("Ningún import deprecado ni inexistente: no hay nada que corregir.")
        return
    st.markdown("**Imports problemáticos** (retrieval contra el umbral)")
    st.dataframe(target_rows(prep.contexts, analysis.threshold), hide_index=True, width="stretch")
    for c in prep.contexts:
        if c.chunks:
            with st.expander(f"Fragmentos que vería el LLM para `{c.module}` ({len(c.chunks)})"):
                for ch in c.chunks:
                    st.caption(f"{ch['header']} · {ch['n_tokens']} tokens")
                    st.code(ch["text"], language="markdown")


def show_fix(fix, original: str, name: str) -> None:
    result = fix.result
    cols = st.columns(4)
    cols[0].metric("LLM", seconds(fix.seconds) if fix.paid["input_tokens"] else "del caché")
    cols[1].metric("Tokens pagados (entrada)", f"{fix.paid['input_tokens']:,}".replace(",", "."))
    cols[2].metric("Tokens pagados (salida)", f"{fix.paid['output_tokens']:,}".replace(",", "."))
    cols[3].metric("USD pagados", f"{fix.paid['usd']:.5f}".replace(".", ","),
                   help="0 si las respuestas ya estaban en el caché del LLM; incluye el reintento si lo hubo")
    st.markdown("**Decisiones**")
    st.dataframe(decision_rows(result.report), hide_index=True, width="stretch")
    if result.llm_error:
        st.warning("La respuesta del LLM se cortó por el tope de tokens: el script queda igual.")
    for msg in result.edit_errors:
        st.warning(f"No se aplicaron los cambios (todo o nada): {msg}")
    if (note := retry_note(result)) is not None:
        abstained, problems = note
        (st.warning if abstained else st.info)(
            ("La corrección no pasó la post-validación ni después del reintento: el sistema se abstiene y el script "
             "queda igual." if abstained else
             "La primera respuesta no pasó la post-validación y el reintento la corrigió.") +
            "".join(f"\n- {p.splitlines()[0]}" for p in problems))
    diff = unified_diff(original, result.code, name)
    if diff:
        st.markdown("**Cambios sugeridos** (revisalos antes de aplicarlos)")
        st.code(diff, language="diff")
    else:
        st.info("El script queda igual.")
    new = [e for e in result.report if e.get("new_imports")]
    if new:
        st.caption("Post-validación: imports nuevos en el script corregido → " +
                   ", ".join(f"`{i['module']}` ({i['status']})" for i in new[0]["new_imports"]))
    left, right = st.columns(2)
    left.download_button("Descargar el script corregido", result.code, fixed_name(name), "text/x-python")
    right.download_button("Descargar el reporte JSON", report_json(result), f"{name.removesuffix('.py')}.report.json",
                          "application/json")
    with st.expander("Reporte completo"):
        st.json(report_json(result))


def main() -> None:
    st.title("Corrector de dependencias alucinadas o deprecadas")
    st.caption("TP3 · 73.69 Large Language Models · Grupo 7. RAG sobre la documentación de 101 librerías de Python.")
    eng = engine()
    if eng.chunks() == 0:
        st.error(f"El índice `{eng.collection}` no existe o está vacío: armarlo con build_index.py.")
        return
    version, generator, threshold, budget = sidebar(eng)
    picked = script_input()
    if picked is None:
        st.info("Elegí un ejemplo, subí un archivo o pegá código.")
        return
    name, code = picked
    with st.expander(f"Script: `{name}`", expanded=False):
        st.code(code, language="python")

    key = hashlib.sha256(f"{code}\x00{version}\x00{generator}\x00{threshold}\x00{budget}".encode()).hexdigest()
    st.subheader("1 · Analizar (sin LLM, sin costo de API)")
    if st.button("Analizar", type="primary"):
        try:
            with st.spinner("Validando y recuperando…"):
                st.session_state.analysis = (key, eng.analyze(code, version, threshold, budget, generator))
        except SyntaxError as e:
            st.error(f"El script no parsea: {e}")
            return
        st.session_state.pop("fix", None)
    if st.session_state.get("analysis", (None,))[0] != key:
        return
    analysis = st.session_state.analysis[1]
    show_analysis(analysis)

    model = GENERATORS[generator].llm
    st.subheader(f"2 · Corregir con {model}")
    if analysis.llm_estimate is None:
        if analysis.prepared.contexts:
            st.info("Ningún import problemático pasa el umbral: no se llama al LLM (costo 0) y el reporte los marca "
                    "como «sin reemplazo confiable».")
        return
    est = analysis.llm_estimate
    st.caption(f"Estimado: {est['input_tokens']:,} tokens de entrada".replace(",", ".") +
               f" + ~{est['output_tokens_estimate']} de salida ≈ {est['usd']:.5f} USD".replace(".", ",") +
               (" · **ya está en el caché: no se paga**" if analysis.llm_cached else "") +
               (" · un reintento, si hace falta, se paga aparte" if GENERATORS[generator].retry else ""))
    can_call = analysis.llm_cached or bool(os.environ.get("OPENAI_API_KEY"))
    if st.button("Corregir", type="primary", disabled=not can_call):
        with st.spinner(f"Consultando a {model}…"):
            st.session_state.fix = (key, eng.fix(analysis))
    if st.session_state.get("fix", (None,))[0] == key:
        show_fix(st.session_state.fix[1], code, name)


main()
