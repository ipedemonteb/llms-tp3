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
