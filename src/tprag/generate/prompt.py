"""Prompt del generador (generador.md §2.2-§2.5).

El mensaje no repite las líneas de la consulta (sin sangría y de lugares distintos del script): en la primera
corrida el modelo las copiaba como `antes` y los reemplazos no se encontraban (generador.md §4.1).

Al LLM solo le llega el estado del validador (`deprecated` / `not_found`), no su motivo ni su reemplazo: la lista
manual de deprecados nombra el reemplazo ("el proyecto siguió como pypdf") y eso le daría la respuesta sin pasar por
el retrieval. El motivo del validador sí va al reporte final.
"""

from .context import TargetContext

_HEAD = """\
Sos un asistente que corrige dependencias de Python deprecadas o inexistentes.

Recibís un script, la lista de sus imports problemáticos y, para cada uno, fragmentos de documentación de librerías \
reales recuperados de un índice. Cada fragmento empieza con un encabezado `paquete > página > sección`.

Para cada import problemático, en este orden:
1. "script_usage": escribí qué hace el script con ese import, en una oración.
2. "matching_fragment": el encabezado del fragmento que hace ESO MISMO con otra librería. Que un fragmento sea del \
mismo tema, o que con esa librería se pueda construir algo parecido a mano, no alcanza: tiene que mostrar la misma \
funcionalidad. Si ninguno la muestra, null.
3. "decision":
- "reemplazado": solo si hay "matching_fragment". Usá la API tal como aparece en los fragmentos.
- "sin_reemplazo_confiable": si "matching_fragment" es null. En ese caso no cambies nada relacionado con ese import.

"""

# Formato de la respuesta (iteracion_ej4.md §3.2): "edits" (reemplazos antes → después, el del sistema) o "script" (el
# script completo, solo en el reintento de la iteración). Solo cambia esta sección del mensaje de sistema.
_CHANGES = {
    "edits": """\
Cambios:
- Se expresan como una lista de reemplazos. "antes" es un fragmento copiado EXACTO de la sección "Script": líneas \
consecutivas, con su sangría y saltos de línea, que aparecen una sola vez; "despues" es el texto que lo reemplaza.
- Si hay que cambiar líneas que están en lugares distintos del script, hacé un reemplazo por cada lugar. Nunca \
juntes en un mismo "antes" líneas que en el script no están una debajo de la otra.
- Incluí el cambio de la línea del import y todos los usos que cambian con la librería nueva (nombres de clases y \
métodos, argumentos, valores de retorno).
- No toques nada que no haga falta para el reemplazo: ni estilo, ni comentarios, ni otras dependencias.
- No agregues imports de paquetes que no aparezcan en los fragmentos, salvo la biblioteca estándar.

""",
    "script": """\
Cambios:
- Devolvé el script completo corregido en "script", de la primera a la última línea.
- Cambiá solo lo necesario para el reemplazo: la línea del import y todos los usos que cambian con la librería \
nueva (nombres de clases y métodos, argumentos, valores de retorno).
- Todo lo demás tiene que quedar idéntico al original: mismas líneas, sangría, comentarios, docstrings y el resto de \
las dependencias. No reordenes, no reformatees, no agregues ni borres funcionalidad.
- No agregues imports de paquetes que no aparezcan en los fragmentos, salvo la biblioteca estándar.
- Si decidiste "sin_reemplazo_confiable" para todos los imports, devolvé el script original sin cambios.

""",
}

_REPORT = """\
Reporte, una entrada por cada import problemático recibido:
- "module": el módulo tal como figura en la lista.
- "replacement_package": el paquete de reemplazo tal como aparece al comienzo del encabezado del fragmento \
(lo que está antes del primer " > "); null si no hay reemplazo.
- "reason": en castellano, una o dos oraciones.
- "sources": los encabezados completos de los fragmentos que usaste.
"""

SYSTEMS = {fmt: _HEAD + changes + _REPORT for fmt, changes in _CHANGES.items()}
SYSTEM = SYSTEMS["edits"]

# Primer prompt v2 (`__iter5`, apagado por defecto). En iter3 el modelo resumía el script en un solo aspecto y elegía la
# librería del fragmento que cubría ese aspecto (pyarrow por "carga desde S3" en un script de streamlit), a veces una que
# no estaba en los fragmentos (numpy); se abstenía con la correcta en el contexto por el "ESO MISMO" del paso 2, y en
# los negativos reemplazaba por algo "similar" (pycryptodome para armar un ZIP cifrado). La cabecera separa elegir el
# paquete (el que cubre más usos) de decidir (reemplazar solo si los cubre todos). En el formato "script" también pide
# que el script siga haciendo lo mismo y sea Python válido (dos respuestas de iter3 rompieron una comilla y un
# corchete). El reporte es el mismo. Va con los esquemas `*V2` (campo `candidates`).
_HEAD_V2 = """\
Sos un asistente que corrige dependencias de Python deprecadas o inexistentes.

Recibís un script, la lista de sus imports problemáticos y, para cada uno, fragmentos de documentación de librerías \
reales recuperados de un índice. Cada fragmento empieza con un encabezado `paquete > página > sección`.

Para cada import problemático, en este orden:
1. "script_usage": TODOS los usos del import en el script: qué nombres usa (clases, funciones, métodos, atributos) y \
para qué. No resumas el script en un solo aspecto.
2. "candidates": por cada paquete que aparece en los encabezados de los fragmentos, qué usos del paso 1 cubre y con \
qué nombres de su API que aparecen en los fragmentos. Solo paquetes de los encabezados: nunca propongas uno que no \
esté ahí, aunque lo conozcas. Breve: una línea por paquete en "covered_usages" y solo nombres en "api_names".
3. "matching_fragment": el encabezado de un fragmento del paquete que cubre MÁS usos del paso 1, no el primero que \
parece del mismo tema. A igual cobertura, preferí el paquete cuya API se parece más a la que usa el script (nombres y \
estructura de las llamadas). null si ningún paquete cubre ninguno.
4. "decision":
- "reemplazado": solo si el paquete de "matching_fragment" cubre TODOS los usos del paso 1 con API que aparece en los \
fragmentos. Usá la API tal como aparece en los fragmentos.
- "sin_reemplazo_confiable": si cubre solo una parte de los usos (por ejemplo, cifra pero no arma el ZIP), si solo \
hace algo "similar" o si "matching_fragment" es null. Un reemplazo equivocado es peor que no reemplazar: ante la \
duda, "sin_reemplazo_confiable". En ese caso no cambies nada relacionado con ese import.

"""

_SCRIPT_V2 = _CHANGES["script"].replace("""\
- No agregues imports de paquetes""", """\
- El script corregido tiene que seguir haciendo lo mismo que el original: las mismas entradas, salidas y \
comportamiento, ahora con la librería nueva.
- Tiene que ser Python válido: revisá que las líneas que cambiaste cierren sus comillas, paréntesis y corchetes igual \
que en el original.
- No agregues imports de paquetes""")

# Primer prompt v3 (`__iter6`): v2 con otra decisión. En iter5, "reemplazar solo si cubre TODOS los usos" y "ante la
# duda, sin_reemplazo_confiable" hicieron que se abstuviera en 15 imports que iter3 corregía bien. v3 reemplaza si el
# paquete cubre el uso principal; si un uso secundario queda mal migrado, lo agarra la post-validación.
_DECISION_V2 = _HEAD_V2[_HEAD_V2.index('4. "decision":'):]
_HEAD_V3 = _HEAD_V2.replace(_DECISION_V2, """\
4. "decision":
- "reemplazado": si el paquete de "matching_fragment" cubre el uso principal del script (para qué importa la \
librería) con API que aparece en los fragmentos. Migrá todos los usos con la API de ese paquete.
- "sin_reemplazo_confiable": si ningún paquete cubre el uso principal, o si el que más cubre hace otra cosa del mismo \
tema (por ejemplo, cifra datos cuando el script arma un ZIP cifrado). Un reemplazo equivocado es peor que no \
reemplazar. En ese caso no cambies nada relacionado con ese import.

""")

# Mensajes de sistema por versión del primer prompt y formato. "v1" es el de siempre (sin la opción, nada cambia).
PROMPTS = {"v1": SYSTEMS,
           "v2": {"edits": _HEAD_V2 + _CHANGES["edits"] + _REPORT, "script": _HEAD_V2 + _SCRIPT_V2 + _REPORT},
           "v3": {"edits": _HEAD_V3 + _CHANGES["edits"] + _REPORT, "script": _HEAD_V3 + _SCRIPT_V2 + _REPORT}}

STATUS = {"deprecated": "deprecado", "not_found": "no existe en PyPI"}


def user_message(code: str, contexts: list[TargetContext]) -> str:
    """Mensaje con el script y los imports que pasaron el umbral (los demás no se le muestran al LLM)."""
    parts = ["## Script", "", "```python", code.rstrip("\n"), "```", "", "## Imports problemáticos"]
    for ctx in contexts:
        parts += ["", f"### `{ctx.module}` ({STATUS.get(ctx.status, ctx.status)})", "", "Fragmentos:"]
        for i, chunk in enumerate(ctx.chunks, start=1):
            parts += ["", f"--- [{i}] {chunk['header']}", chunk["text"].strip("\n")]
    return "\n".join(parts) + "\n"


RETRY_CLOSING = {
    "edits": ("Respondé de nuevo de forma completa (reporte y reemplazos), corrigiendo estos problemas. Si con los "
              "fragmentos no podés hacer la migración completa, respondé \"sin_reemplazo_confiable\"."),
    "script": ("Esta vez devolvé el script completo corregido (no reemplazos), migrando todas las líneas listadas en "
               "\"Líneas que usan el import a reemplazar\". Si con los fragmentos no podés hacer la migración completa, "
               "respondé \"sin_reemplazo_confiable\" y devolvé el script original."),
}


def retry_message(user: str, edits: list[dict], problems: list[str], fmt: str = "edits",
                  usages: dict[str, list[tuple[int, str]]] | None = None, previous: str | None = None) -> str:
    """Mensaje del reintento (iteración, iteracion_ej4.md §3.2): el mismo mensaje de la primera vez, los reemplazos que
    propuso el LLM y los problemas que encontró la post-validación. Un solo mensaje (no una conversación), así el caché y
    la llamada al LLM son los mismos. Con `fmt="script"` (se le pide el script completo) agrega, por módulo, las líneas
    del script que usan el import (`usages`: módulo → [(número, texto)]), para que las migre todas. Con `previous` (la
    variante uniforme: la primera respuesta ya fue el script completo), en "Tu respuesta anterior" va ese script."""
    parts = [user.rstrip("\n")]
    if fmt == "script":
        parts += ["", "## Líneas que usan el import a reemplazar"]
        for module, lines in (usages or {}).items():
            parts += ["", f"### `{module}`"] + [f"- línea {n}: {text}" for n, text in lines]
    parts += ["", "## Tu respuesta anterior"]
    if previous is not None:
        parts += ["", "```python", previous.rstrip("\n"), "```"]
    for i, e in enumerate(edits, start=1):
        parts += ["", f"--- reemplazo {i}", "antes:", "```python", e["antes"].rstrip("\n"), "```", "despues:",
                  "```python", e["despues"].rstrip("\n"), "```"]
    parts += ["", "## Problemas de tu respuesta anterior", ""] + [f"- {p}" for p in problems]
    parts += ["", RETRY_CLOSING[fmt]]
    return "\n".join(parts) + "\n"
