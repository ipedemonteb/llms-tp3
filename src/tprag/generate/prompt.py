"""Prompt del generador (generador.md §2.2-§2.5).

El mensaje no repite las líneas de la consulta (sin sangría y de lugares distintos del script): en la primera
corrida el modelo las copiaba como `antes` y los reemplazos no se encontraban (generador.md §4.1).

Al LLM solo le llega el estado del validador (`deprecated` / `not_found`), no su motivo ni su reemplazo: la lista
manual de deprecados nombra el reemplazo ("el proyecto siguió como pypdf") y eso le daría la respuesta sin pasar por
el retrieval. El motivo del validador sí va al reporte final.
"""

from .context import TargetContext

SYSTEM = """\
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

Cambios:
- Se expresan como una lista de reemplazos. "antes" es un fragmento copiado EXACTO de la sección "Script": líneas \
consecutivas, con su sangría y saltos de línea, que aparecen una sola vez; "despues" es el texto que lo reemplaza.
- Si hay que cambiar líneas que están en lugares distintos del script, hacé un reemplazo por cada lugar. Nunca \
juntes en un mismo "antes" líneas que en el script no están una debajo de la otra.
- Incluí el cambio de la línea del import y todos los usos que cambian con la librería nueva (nombres de clases y \
métodos, argumentos, valores de retorno).
- No toques nada que no haga falta para el reemplazo: ni estilo, ni comentarios, ni otras dependencias.
- No agregues imports de paquetes que no aparezcan en los fragmentos, salvo la biblioteca estándar.

Reporte, una entrada por cada import problemático recibido:
- "module": el módulo tal como figura en la lista.
- "replacement_package": el paquete de reemplazo tal como aparece al comienzo del encabezado del fragmento \
(lo que está antes del primer " > "); null si no hay reemplazo.
- "reason": en castellano, una o dos oraciones.
- "sources": los encabezados completos de los fragmentos que usaste.
"""

STATUS = {"deprecated": "deprecado", "not_found": "no existe en PyPI"}


def user_message(code: str, contexts: list[TargetContext]) -> str:
    """Mensaje con el script y los imports que pasaron el umbral (los demás no se le muestran al LLM)."""
    parts = ["## Script", "", "```python", code.rstrip("\n"), "```", "", "## Imports problemáticos"]
    for ctx in contexts:
        parts += ["", f"### `{ctx.module}` ({STATUS.get(ctx.status, ctx.status)})", "", "Fragmentos:"]
        for i, chunk in enumerate(ctx.chunks, start=1):
            parts += ["", f"--- [{i}] {chunk['header']}", chunk["text"].strip("\n")]
    return "\n".join(parts) + "\n"
