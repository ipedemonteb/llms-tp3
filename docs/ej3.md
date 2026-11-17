# 73.69 - Large Language Models

**Trabajo Práctico N°3 - Ejercicio 3: Responsible AI, Safety y puesta en producción**
**Instituto Tecnológico de Buenos Aires**

**Grupo 7**

**Integrantes:**

- Barnatán, Martín Alejandro (64463)
- Cantarella, Juan Ignacio (64509)
- Pedemonte Berthoud, Ignacio (64908)

**Fecha de Entrega:** 17/11/2026

---

El sistema recibe un script de Python, detecta los imports inexistentes o deprecados (validador sobre el AST + API de PyPI) y propone un reemplazo a partir de la documentación de 101 librerías (Chroma, reranker Qwen3 y un LLM de OpenAI). Como modifica código y recomienda dependencias, sus errores llegan directo a la cadena de suministro del usuario: una abstención deja el import marcado para que lo revise una persona, mientras que una corrección mala rompe el código o introduce una librería que nadie verificó. El umbral del retrieval es permisivo (deja pasar el 95% de los positivos de dev, para no perder correcciones), así que ese riesgo se controla después del LLM y con revisión humana.

## 1. Responsible AI y Safety

**Abstenerse antes que corregir mal.** En test, con ese umbral, la versión con reranker hace 22 correcciones buenas (pasan los chequeos automáticos) y 37 malas (precisión 0,37). En la iteración del Ej. 4, gpt-4.1-mini devuelve el script completo y, si la corrección no parsea o deja nombres sin definir, recibe el problema una sola vez; si persiste, el sistema se abstiene y entrega el script original. En el mismo test, la iteración hace 44 buenas y 11 malas (precisión 0,79), y cada versión cambia 1 de los 10 negativos. Falta mejorar la capacidad de abstención del generador: en 7 de esas 11 malas la librería correcta no estaba en los fragmentos y el modelo reemplazó igual, cuando tenía que abstenerse.

**No crear el riesgo que se quiere evitar.** El sistema nunca instala ni ejecuta el código: el validador lee el AST y consulta la API JSON de PyPI. El corpus son 101 librerías del top de PyPI fijadas por commit, y el prompt solo admite reemplazos que aparezcan en los fragmentos recuperados. La post-validación de la iteración es un guardrail determinístico: ya bloquea lo que no parsea o deja nombres sin definir, pero todavía no los imports nuevos: en test, una corrección agregó `from lazy_group import LazyGroup`, un módulo de un ejemplo de la documentación de click que no existe en PyPI. En producción tiene que rechazar también todo import de un paquete fuera de esa lista cerrada, para que el LLM no pueda introducir un nombre inventado, que es justo el vector del *slopsquatting*. Como siguiente paso, un estado `suspicious` marcaría paquetes que existen pero tienen señales de riesgo (una sola release, creación reciente, nombre casi igual a uno popular); hoy solo se advierten vulnerabilidades conocidas y la falta de releases recientes.

**Entradas no confiables y prompt injection.** El script (comentarios, strings, docstrings) puede traer una inyección directa y la documentación indexada, una indirecta: los dos son texto de terceros. Hoy el LLM no tiene herramientas ni red, su salida es estructurada (script corregido + reporte) y se valida en código sin ejecutarla, el script va en un bloque separado de las instrucciones y el corpus se actualiza desde commits revisados, nunca desde la web en vivo (limita el data poisoning). Falta indicarle al modelo de forma explícita que trate ese bloque como dato. Antes de desplegar hay que armar un conjunto de *red teaming* con instrucciones inyectadas en comentarios y en la documentación, y medirlo como los negativos.

**Transparencia y supervisión humana.** Cada import del reporte JSON lleva su estado, el motivo, la librería elegida y los fragmentos usados, y el script corregido se entrega como sugerencia (diff), nunca se aplica solo. Sin la iteración, en test 19 reportes decían «reemplazado» con el script intacto; con ella, ninguno. Los chequeos automáticos no ejecutan el script, así que marcan un techo y, por la ley de Goodhart, no conviene optimizarlos solos: de los 42 scripts corregidos por la iteración que los pasan en test, según Claude Opus con una rúbrica, 22 conservan la tarea, 4 a medias y 16 no (sin iterar la conservan 4 de 22). Como juez, gpt-4o-mini casi no coincide con la persona en la revisión manual de 20 scripts (κ = 0,26) y Opus sí (κ = 0,88). Por eso decide el desarrollador, y todo juez automático se valida contra una muestra humana antes de usarse.

**Privacidad, licencias y alcance.** El script viaja a la API de OpenAI. En producción hay que enmascarar secretos y datos personales antes de enviar, no guardar código en los logs (solo hashes y métricas), acordar retención cero con el proveedor y ofrecer un modo local para código propietario (solo faltaría un generador local). La documentación indexada es de proyectos open source; se respetan sus licencias y el reporte cita el origen de cada fragmento. Los límites se le comunican al usuario: un solo archivo de Python, documentación en inglés y 101 librerías populares, con sesgo hacia lo más descargado. Fuera de esa cobertura, la respuesta correcta es abstenerse.

## 2. Estrategias para poner el modelo en producción

**Producto.** Se distribuye como CLI (`fix_script.py` y una interfaz en Streamlit ya existen) y como pre-commit hook y GitHub Action que comenta el PR con el reporte y el diff sugerido. Detrás hay un servicio sin estado (FastAPI) con autenticación, límite de tamaño de entrada y rate limiting. Si la API del LLM no responde, devuelve solo el diagnóstico del validador.

**Dos pipelines separados.** El offline descarga la documentación por commit o tag, la normaliza, la parte con el splitter S3, calcula los embeddings y publica el índice como artefacto versionado, junto con su umbral calibrado y su evaluación. Se re-ejecuta cuando una librería publica una release, y la lista de deprecados se mantiene con revisión humana. El online corre el validador y, solo si hay imports problemáticos, el retrieval, el reranker, el LLM y la post-validación. Un script sin imports problemáticos se responde sin LLM ni reranker, a costo casi nulo. Chroma local pasa a un servidor (Qdrant o pgvector) y el caché de PyPI a un Redis compartido.

**Latencia y costo.** El cuello de botella es el reranker: Qwen3-Reranker-0.6B tarda ~16 s por consulta en una Mac M3, y más en CPU. Se puede usar una GPU dedicada o cuantizar a ONNX, recortar los chunks antes de puntuarlos o caer al reranker liviano (gte-modernbert, ~3,7 s) cuando hay carga. El LLM costaba ~0,0007 USD por script con gpt-4o-mini; con la iteración (gpt-4.1-mini, script completo y reintentos) cuesta ~0,003, unos 3 USD cada mil scripts que llegan al LLM. Se mantienen el caché de respuestas por prompt, un presupuesto por usuario y alarmas de gasto.

**Calidad antes y después de desplegar.** Cada cambio de prompt, modelo, índice o umbral pasa por los tests (440) y por la evaluación en dev con caché como gate de CI; test se reserva para comparar versiones. El despliegue es primero en *shadow mode* y después canary, con rollback a la colección anterior. Con LangSmith u OpenTelemetry se registran la latencia por etapa, los tokens y USD, los scores del retrieval, la tasa de abstención, los rechazos de la post-validación y la tasa de sugerencias aceptadas en los PRs. Hay alertas ante drift en la distribución de scores (de dev a test, los positivos que superan el umbral bajaron de 95% a 67%), y una revisión humana periódica de una muestra alimenta el dataset de evaluación.
