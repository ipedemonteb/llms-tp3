# Prompt synthetic_v1: reescritura de un bloque de documentación como script con una librería inventada

Usado el 2026-11-10 con agentes Claude Haiku (4 tandas de 25) para el dataset sintético de dev
(`docs/agents/plan_experimentos_indice.md` §2, `data/eval/synthetic_sample.json`). Se guarda el prompt y la salida, no la
generación (`dataset_eval.md` §2.1).

## Tu tarea

Para cada bloque de tu archivo de entrada, escribí **un script de Python realista que hace la misma tarea que el bloque,
pero usando una librería inventada** en lugar de la librería real. Es material de prueba para un sistema que detecta
imports de librerías que no existen y busca en la documentación la librería correcta.

## Reglas del script

1. **Librería inventada.** Elegí un nombre de módulo en minúsculas, plausible y poco común (por ejemplo `plotnimbus`,
   `quillframe`, `veloxgrid`), que muy probablemente NO exista en PyPI. Usalo con `import nombre` o
   `from nombre import algo` (o un submódulo: `from nombre.io import algo`).
2. **Nunca nombres la librería real**: ni su paquete ni su módulo de import (te los indicamos en cada bloque), ni en
   imports, ni en comentarios, ni en strings, ni en nombres de variables. Tampoco `pd`/`np` si el bloque los usa.
3. **API inventada pero análoga.** La librería inventada tiene que hacer lo mismo que hace el bloque, con la misma
   secuencia de operaciones, pero con nombres de clases, funciones y argumentos propios y distintos de los del bloque
   (no copies los identificadores característicos). El código que sigue debe leerse como el uso natural de esa API.
4. **Script realista, no un snippet:** de 30 a 50 líneas, con al menos una función, un `main()` y
   `if __name__ == "__main__":`, más 3 o 4 imports de la **biblioteca estándar** (`os`, `sys`, `logging`, `json`,
   `pathlib`, `argparse`, `datetime`, `collections`, etc.) que se usen de verdad. **No importes ninguna otra librería de
   terceros.** Agregá código que no tiene que ver con la tarea (argumentos, logging, manejo de errores).
5. **Sintaxis válida de Python 3** (el script se parsea con `ast`). No lo ejecutes: la librería no existe.
6. **Idioma de los comentarios y de los strings:** el que diga cada bloque (`Comment language`). Los nombres de
   variables y funciones, siempre en inglés.
7. **Bloque que no sirve.** Si el bloque no es una tarea real que se pueda reescribir (solo imprime una salida,
   es una configuración, un fragmento sin sentido aislado), no escribas script: la primera línea del archivo debe ser
   `# SKIP: <motivo en una frase>` y nada más.

## Formato de salida

Un archivo por bloque, `<carpeta de salida>/<INDEX con 3 dígitos>.py` (por ejemplo `012.py`), escrito con la herramienta
Write. Las dos primeras líneas son comentarios con metadatos, y después va el script:

```
# TASK: <la tarea en una frase, en castellano>
# INVENTED: <nombre del módulo inventado, exactamente como aparece en el import>
import os
...
```

No escribas nada más (ni README, ni resúmenes). Al terminar, respondé solo con la cantidad de archivos escritos y cuántos
fueron SKIP.
