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

El sistema recibe un script de Python, detecta los imports inexistentes o deprecados (validador sobre el AST + API de PyPI) y propone un reemplazo a partir de la documentación de 101 librerías (Chroma, reranker Qwen3 y un LLM de OpenAI). Como modifica código y recomienda dependencias, sus errores llegan directo a la cadena de suministro del usuario. El criterio de todo el informe es que un reemplazo equivocado es peor que no reemplazar: abstenerse deja el import marcado para que lo revise una persona, mientras que una corrección mala rompe el código o introduce una librería que nadie verificó.

## 1. Responsible AI y Safety

**Abstenerse antes que corregir mal.** En dev, con el umbral que usamos (el que deja pasar el 95% de los positivos), la versión con reranker hacía 35 correcciones buenas y 48 malas (precisión 0,42). En la iteración del Ej. 4, gpt-4.1-mini devuelve el script completo y, si la corrección no parsea o deja nombres sin definir, recibe el problema una sola vez; si persiste, el sistema se abstiene y entrega el script original. En dev pasa a 82 buenas y 8 malas (precisión 0,91) sin cambiar ningún negativo. En test, corrido una sola vez con el umbral de dev, hace 44 buenas y 11 malas (precisión 0,79) y cambia 1 de los 10 negativos. También se abstiene mucho más: solo el 67% de los positivos supera el umbral, contra el 95% en dev. El umbral se recalibra con cada cambio de configuración.

**No crear el riesgo que se quiere evitar.** El sistema nunca instala ni ejecuta el código: el validador lee el AST y consulta la API JSON de PyPI. El corpus son 101 librerías del top de PyPI fijadas por commit, y el prompt solo admite reemplazos que aparezcan en los fragmentos recuperados. La post-validación de la iteración ya bloquea lo que no parsea o deja nombres sin definir, pero todavía no para los imports nuevos: en test, una corrección agregó `from lazy_group import LazyGroup`, un módulo de un ejemplo de la documentación de click que no existe en PyPI. En producción tiene que rechazar todo import de un paquete fuera de esa lista cerrada, para que el LLM no pueda introducir un nombre inventado, que es justo el vector del *slopsquatting*. Como siguiente paso, un estado `suspicious` marcaría paquetes que existen pero tienen señales de riesgo (una sola release, creación reciente, sin repositorio, nombre a poca distancia de edición de uno popular), cruzado con avisos de vulnerabilidades (OSV). Hoy solo se advierten `known_vulnerabilities` y `no_recent_release`.

**Entradas no confiables y prompt injection.** Tanto el script (comentarios, strings, docstrings) como la documentación indexada son texto de terceros que puede traer instrucciones. Hoy el LLM no tiene herramientas ni red, su salida es estructurada (script corregido + reporte) y se valida en código sin ejecutarla, el script va en un bloque separado de las instrucciones y el corpus se actualiza desde commits revisados, nunca desde la web en vivo. Falta indicarle al modelo de forma explícita que trate ese bloque como dato. Antes de desplegar hay que armar un conjunto de *red teaming* con instrucciones inyectadas en comentarios y en páginas de documentación, y medirlo igual que los negativos (¿cambia algo que no debía?).

**Transparencia y supervisión humana.** Cada import del reporte JSON lleva su estado, el motivo, la librería elegida y los fragmentos de documentación usados, y el script corregido se entrega como sugerencia (diff), nunca se aplica solo. Sin la iteración, 23 reportes (con reranker) decían «reemplazado» con el script intacto; con ella, ninguno. La evaluación justifica la supervisión: en la revisión manual de 20 scripts, gpt-4o-mini como juez casi no coincide con la persona sobre si se conserva la tarea (κ = 0,26) y Opus sí (κ = 0,88); en test, de las 42 correcciones que pasan todos los chequeos automáticos, Opus encuentra 16 que no la conservan. Por eso decide el desarrollador, y todo juez automático se valida contra una muestra humana antes de usarse.

**Privacidad, licencias y alcance.** El script viaja a la API de OpenAI. En producción hay que enmascarar secretos antes de enviar, no guardar código en los logs (solo hashes y métricas), acordar retención cero con el proveedor y ofrecer un modo local para código propietario (el embedder y el reranker ya corren localmente; faltaría un generador local). La documentación indexada es de proyectos open source; se respetan sus licencias y el reporte cita el origen de cada fragmento. Los límites se le comunican al usuario: un solo archivo de Python, documentación en inglés y 101 librerías populares, con sesgo hacia lo más descargado. Fuera de esa cobertura, la respuesta correcta es abstenerse.

## 2. Estrategias para poner el modelo en producción

**Producto.** Se distribuye como CLI (`fix_script.py` ya existe) y como pre-commit hook y GitHub Action que comenta el PR con el reporte y el diff sugerido: el desarrollador acepta o descarta. Detrás hay un servicio sin estado (FastAPI) con autenticación, límite de tamaño de entrada y rate limiting.

**Dos pipelines separados.** El offline descarga la documentación por commit o tag, la normaliza, la parte con el splitter S3, calcula los embeddings y publica el índice como artefacto versionado, junto con su umbral calibrado y su evaluación. Se re-ejecuta de forma programada cuando una librería publica una release, y la lista de deprecados se mantiene con revisión humana. El online corre el validador y, solo si hay imports problemáticos, el retrieval, el reranker, el LLM y la post-validación. La mayoría de los scripts no tiene imports problemáticos y se responde sin LLM ni reranker, a costo casi nulo. Chroma local pasa a un servidor (Qdrant o pgvector) y el caché de PyPI a un Redis compartido.

**Latencia y costo.** El cuello de botella es el reranker: Qwen3-Reranker-0.6B tarda ~16 s por consulta en una Mac M3, y más en CPU. Se puede usar una GPU dedicada o cuantizar a ONNX, recortar los chunks antes de puntuarlos o caer al reranker liviano (gte-modernbert, ~3,7 s con los mismos 30 candidatos) cuando hay carga. El LLM costaba ~0,0007 USD por script con gpt-4o-mini; con la iteración (gpt-4.1-mini, script completo y reintentos) cuesta ~0,003, unos 3 USD cada mil scripts que llegan al LLM. Se mantienen el caché de respuestas por prompt, un presupuesto por usuario y alarmas de gasto.

**Calidad antes y después de desplegar.** Cada cambio de prompt, modelo, índice o umbral pasa por los tests (435) y por la evaluación en dev con caché como gate de CI; test se reserva para comparar versiones. El despliegue es primero en *shadow mode* y después canary, con rollback a la colección anterior. Con LangSmith u OpenTelemetry se registran la latencia por etapa, los tokens y USD, los scores del retrieval, la tasa de abstención, los rechazos de la post-validación y, como señal de calidad real, la tasa de sugerencias aceptadas en los PRs. Hay alertas ante drift en la distribución de scores (lo que pasó entre dev y test), y una revisión humana periódica de una muestra alimenta el dataset de evaluación.
