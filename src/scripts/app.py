"""Interfaz web del corrector (extra de la consigna). El sistema final (`tprag.generate.system`, el mismo que corre
`fix_script.py`) en un solo paso: la persona pega o sube un script y recibe el script corregido y el reporte. No tiene
opciones: la configuración es la elegida en el Ej. 4, y lo interno del sistema no se muestra.

Uso (desde la raíz del repo):
    uv run --group local --group ui streamlit run src/scripts/app.py

Necesita el índice `s3-1024__e4__r1__cap150` en data/index/ (build_index.py) y `OPENAI_API_KEY` en .env: el LLM no usa
el caché de los estudios, así que cada corrección es una llamada real. Los puntajes del reranker sí salen de su caché
cuando están. La primera vez descarga el embedder jina-code y el reranker Qwen3.
"""

import hashlib
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from tprag.split.store import ROOT
from tprag.ui.engine import Engine
from tprag.ui.view import decision_rows, fixed_name, outcome, report_json, unified_diff

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Corrector de dependencias · TP3 Grupo 7", page_icon="🐍", layout="wide")

CSS = """
<style>
.block-container {max-width: 1150px; padding-top: 2.5rem;}
.app-header {margin-bottom: 1.5rem;}
.app-header h1 {font-size: 2.1rem; line-height: 1.2; margin: 0.2rem 0 0.4rem; padding: 0;}
.app-header p {font-size: 1.05rem; opacity: 0.75; margin: 0;}
[data-testid="stFileUploaderDropzone"] {min-height: 240px;}  /* la altura del área de texto del otro panel */
.stTextArea textarea {font-family: "Source Code Pro", ui-monospace, monospace; font-size: 0.85rem;}
</style>
"""

@st.cache_resource(show_spinner="Abriendo el índice…")
def engine() -> Engine:
    return Engine()


def script_input() -> tuple[str, str] | None:
    """(nombre, código) que eligió la persona, o None. Si hay un archivo subido, manda el archivo."""
    left, right = st.columns(2, gap="medium")
    with left.container(border=True):
        st.markdown(":material/content_paste: **Pegá tu código**")
        pasted = st.text_area("Código", height=240, placeholder="Pegá acá tu script de Python…",
                              label_visibility="collapsed")
    with right.container(border=True, height="stretch"):  # la misma altura que el panel del código
        st.markdown(":material/upload_file: **O subí un archivo .py**")
        uploaded = st.file_uploader("Archivo de Python", type=["py"], label_visibility="collapsed")
    if uploaded is not None:
        return uploaded.name, uploaded.getvalue().decode("utf-8", errors="replace")
    return ("script.py", pasted) if pasted.strip() else None


def show_result(fix, original: str, name: str) -> None:
    result = fix.result
    st.divider()
    st.subheader("Resultado")
    for level, text in outcome(result):
        if level == "success":
            st.success(text, icon=":material/check_circle:")
        else:
            st.warning(text, icon=":material/warning:")
    diff = unified_diff(original, result.code, name)
    if diff:
        tab_diff, tab_fixed = st.tabs([":material/difference: Cambios", ":material/code: Script corregido"])
        tab_diff.code(diff, language="diff")
        tab_fixed.code(result.code, language="python", line_numbers=True)
    if decisions := decision_rows(result.report):
        st.markdown(":material/description: **Reporte**")
        st.table(pd.DataFrame(decisions).style.hide(axis="index"))  # st.table parte el motivo en varias líneas
    left, right = st.columns(2, gap="medium")
    if diff:
        left.download_button("Descargar el script corregido", result.code, fixed_name(name), "text/x-python",
                             icon=":material/download:", type="primary", width="stretch")
    (right if diff else left).download_button("Descargar el reporte JSON", report_json(result),
                                              f"{name.removesuffix('.py')}.report.json", "application/json",
                                              icon=":material/data_object:", width="stretch")


def run(eng: Engine, code: str):
    """(análisis, corrección) o un mensaje de error para la persona."""
    try:
        analysis = eng.analyze(code)
    except SyntaxError as e:
        return f"El script no parsea como Python: {e}"
    except Exception as e:  # noqa: BLE001  (sin red para bajar el reranker, índice roto, ...)
        return f"No se pudo analizar el script: {type(e).__name__}: {e}"
    if analysis.prepared.prompt is not None and not os.environ.get("OPENAI_API_KEY"):
        return "Falta `OPENAI_API_KEY` en `.env`: sin ella no se puede consultar al LLM para corregir el script."
    try:
        return analysis, eng.fix(analysis)
    except Exception as e:  # noqa: BLE001  (la API no responde, la key no sirve, se acabó la cuota, ...)
        return f"El LLM no respondió ({type(e).__name__}: {e}). Probá de nuevo en un rato."


def main() -> None:
    st.html(CSS)
    st.html('<div class="app-header">'
            "<h1>Corrector de Dependencias Alucinadas o Deprecadas de Python</h1>"
            "<p>Sube tu código para verificar y corregir sus dependencias.</p></div>")
    try:
        eng = engine()
        empty = eng.chunks() == 0
    except Exception as e:  # noqa: BLE001  (falta el grupo `local`, el índice está roto, ...)
        st.error(f"No se pudo abrir el índice: {type(e).__name__}: {e}")
        st.caption("La app se corre con `uv run --group local --group ui streamlit run src/scripts/app.py`.")
        return
    if empty:
        st.error(f"El índice `{eng.collection}` no existe o está vacío: armarlo con build_index.py.")
        return
    picked = script_input()
    key = None if picked is None else hashlib.sha256("\x00".join(picked).encode()).hexdigest()  # nombre + código
    if st.button("Corregir", type="primary", disabled=picked is None, icon=":material/auto_fix_high:",
                 width="stretch"):
        name, code = picked
        with st.spinner("Revisando las dependencias y corrigiendo el script…"):
            st.session_state.run = (key, name, code, run(eng, code))
    if "run" not in st.session_state:
        return
    if st.session_state.run[0] != key:  # cambió la entrada: el resultado anterior se descarta (no vuelve solo)
        del st.session_state.run
        return
    _, name, code, outcome_ = st.session_state.run  # se guarda: sigue a la vista si la página se vuelve a correr
    if isinstance(outcome_, str):
        st.error(outcome_, icon=":material/error:")
        return
    show_result(outcome_[1], code, name)


main()
