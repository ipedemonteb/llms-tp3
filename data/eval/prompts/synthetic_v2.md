# Prompt synthetic_v2: igual que synthetic_v1, pero sin copiar la API de la librería real

Segunda pasada del 2026-11-10 sobre los ítems de `synthetic_v1` cuya consulta (import + líneas de uso) compartía 3 o más
nombres específicos con el bloque de la documentación (`DataLoader`, `from_pretrained`, `add_chart`...): la "librería
inventada" era en la práctica la real con otro nombre de módulo, lo que hace trivial la búsqueda.

## Tu tarea

Leé y seguí `data/eval/prompts/synthetic_v1.md` (mismo formato de salida, mismas reglas). Estas reglas nuevas tienen
prioridad sobre la regla 3 de v1:

1. Cada bloque trae una lista **NOMBRES PROHIBIDOS**: identificadores de la librería real (clases, funciones, métodos,
   argumentos, constantes) y datos de ejemplo del bloque. **Tu script no puede contener ninguno de esos nombres**,
   ni en el código ni en los comentarios ni en los strings.
2. Inventá nombres propios para toda la API de la librería inventada: clases, funciones, métodos y argumentos con
   palabras distintas (sinónimos o términos del dominio), pero que hagan lo mismo y se lean naturales. Los nombres de la
   biblioteca estándar de Python y los builtins sí se pueden usar.
3. Cambiá también los datos de ejemplo (nombres de personas, ciudades, textos, valores) por otros distintos.
4. **No hagas SKIP**: reescribí siempre el bloque, aunque la tarea sea simple.
