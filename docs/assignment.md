# Implementación Transformer

**Trabajo Práctico 3**
**73.69 Large Language Models - 2026**

## 1. Implementación Transformer

El objetivo de este Trabajo Práctico es implementar una aplicación haciendo uso de un modelo de Transformer con la arquitectura de RAG para la resolución de un problema. El problema será propuesto por cada grupo y podrá utilizar texto, imágenes y/o audios, según las necesidades del proyecto.

Para el desarrollo de este trabajo se podrán utilizar las librerías y herramientas vistas en clase y la cátedra disponibilizará API Keys de OpenAI a cada grupo.

### 1.1. Ejercicio 1

Se deberá consultar previamente con las profesoras el alcance y la naturaleza del problema seleccionado antes de iniciar su desarrollo. Para ello deberán definir el proyecto a desarrollar, incluyendo como mínimo:

- Descripción del problema.
- Input y output de la aplicación.
- Fuente de los datos y formato.
- ¿Qué van a persistir en la base de datos vectorial?
- Alcance inicial y principales desafíos.
- Diagrama de arquitectura preliminar.

**Aclaración:** Debido a su simplicidad, no se permitirá la implementación de un sistema de Question and Answering (Q&A) genérico. En su lugar, el trabajo deberá abordar un problema específico, claramente definido y con mayor grado de elaboración.

Se entiende por Q&A a un sistema cuyo objetivo es responder preguntas en lenguaje natural a partir de una base de conocimiento.

**Ejemplos de Q&A (no permitido)**

**Ejemplo 1.** Dado un conjunto de documentos, el sistema responde consultas como:

- “¿Qué es el aprendizaje automático?”
- “¿En qué consiste el TP1?”
- “¿Quiénes son los profesores de la materia?”

**Ejemplo 2.** Dado un conjunto de documentos corporativos (por ejemplo, misión, visión, valores y reportes internos), el sistema responde preguntas generales como:

- “¿Cuál es el objetivo de la empresa?”
- “¿Cómo es la cultura organizacional?”
- “¿Cuáles son los valores de la compañía?”

**Ejemplos que NO son Q&A desarrollados en cuatrimestres anteriores (disparadores)**

- Generación de unit tests en Solidity a partir de lineamientos establecidos por una cátedra de Criptografía y repositorios curados.
- Similitud entre logos de empresas personalizada para evitar falsificaciones o plagio.
- Recomendación de videos en base a requerimientos de aprendizaje del usuario y características de los videos (duración, nivel de dificultad, etc.).
- Autocompletado de formularios para aplicaciones de trabajo.
- Dado un snippet de código HTML, se obtiene en qué parte de una codebase (Angular o React) se generó.

### 1.2. Ejercicio 2

Se tendrán en cuenta para la evaluación los siguientes conceptos:

- **Retrieval Augmented Generation (RAG):** Integración de técnicas de Data Loading, Text Splitting, Bases de Datos Vectoriales y Retrieval.
- **Técnicas de Preprocesamiento y Postprocesamiento:** Implementación adecuada de métodos para preparar y refinar los datos antes y después de su procesamiento por el modelo.
- **Dataset:** Creación y utilización de un dataset que puede incluir texto, imágenes y/o audios, según las necesidades del problema.
- **Métricas:** Establecimiento de métricas claras y precisas para evaluar el desempeño del modelo.
- **Evaluación:** Estudio de ablación claro donde se incorporan diferentes configuraciones del modelo como cambios incrementales y su comparación.
- **Advanced RAG:** Implementación de al menos una técnica avanzada, por ejemplo, query rewriting, query expansion, reranking o un enfoque similar.

#### 1.2.1. Extras

Como componentes adicionales opcionales, se valorará la incorporación de alguno de los siguientes elementos:

- Integración de LangSmith para monitoreo y observabilidad.
- Análisis de costo computacional y uso de tokens.
- Integración de una interfaz de usuario (UI).
- Investigación extra relacionada al caso de uso.

**Aclaración:** Se deberá tener todo el pipeline de RAG finalizado antes de agregar Advanced RAG o los extras para asegurar un desarrollo ordenado y eficiente. El foco está en la correcta implementación de RAG y la compresión de todos sus módulos.

### 1.3. Ejercicio 3

Asimismo, se deberán definir (no implementar) los siguientes aspectos:

1. Consideraciones para asegurar que el modelo cumpla con los principios de Responsible AI y Safety.
2. Estrategias para poner el modelo en producción.

La entrega de este ejercicio será un informe de máximo 2 hojas entregado un día antes de la presentación al oral.

### 1.4. Ejercicio 4

Una vez desarrollado todo el sistema se deberá iterar sobre el mismo para mejorar su desempeño y/o incorporar nuevas funcionalidades. Se deberá documentar el proceso de iteración y los resultados obtenidos.

Para ello, se deberá identificar el problema o limitación encontrado en la versión anterior, describir la modificación propuesta junto con la justificación de dicha decisión y detallar la configuración utilizada antes y después del cambio, de manera que sea posible analizar claramente el impacto de la modificación sobre el sistema, evaluando los resultados obtenidos y comparándolos con la versión anterior.

## 2. Entregas

- **Entrega Idea (Ejercicio 1): 7/10, 18:59 hs.** Instancia oral donde se presentará el proyecto a desarrollar. Deberán cargar previamente en Campus un informe breve (máximo 2 hojas) con la descripción del proyecto y el diagrama de arquitectura preliminar.
- **Entrega Ejercicio 3:** El informe del ejercicio 3 se deberá cargar en Campus un día antes de la presentación para optimizar el tiempo de exposición (**17/11 18:59 hs**). Este ejercicio no se expone de forma oral, pero se evaluará su contenido.
- **Entrega final: 18/11, 18:59 hs.** Instancia oral donde se presentará el proyecto. Deberán cargar previamente en Campus la presentación, repositorio y hash del commit.

La presentación oral será de aproximadamente 30 minutos y deberá contener:

- Explicación del problema a resolver.
- Desafíos encontrados.
- Decisiones de implementación correspondientes bien justificadas.
- Resultados obtenidos en ambas versiones.
- Limitaciones y trabajo futuro.
- Conclusiones.

**Aclaración:** Se penalizará medio punto por cada entrega tardía. Dado la cantidad de tiempo disponible para el desarrollo del trabajo, se recomienda aprovechar el espacio de consultas del día 4/11 teniendo el trabajo lo más avanzado posible.
