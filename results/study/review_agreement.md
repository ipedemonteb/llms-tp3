# Acuerdo de la revisión manual

20 scripts de dev (10 de base, 10 de rerank), muestra estratificada por (éxito del nivel 1 × veredicto de gpt-4o-mini), revisada a ciegas (metricas.md §3.8). IC 95% por bootstrap sobre ítems; con 10 ítems por versión son muy anchos.

## Preservación de la tarea (Answer Relevance (b))

| Par | Grupo | n | Acuerdo exacto | A un nivel | Kappa de Cohen |
|---|---|---|---|---|---|
| gpt-4o-mini vs humano | total | 20 | 0.45 [0.25, 0.65] | 0.85 | 0.26 [0.07, 0.50] |
| gpt-4o-mini vs humano | base | 10 | 0.40 [0.10, 0.70] | 0.80 | 0.17 [0.00, 0.48] |
| gpt-4o-mini vs humano | rerank | 10 | 0.50 [0.20, 0.80] | 0.90 | 0.33 [0.00, 0.70] |
| gpt-4o-mini vs opus | total | 20 | 0.40 [0.20, 0.60] | 0.85 | 0.21 [0.04, 0.43] |
| gpt-4o-mini vs opus | base | 10 | 0.30 [0.00, 0.60] | 0.80 | 0.08 [-0.05, 0.33] |
| gpt-4o-mini vs opus | rerank | 10 | 0.50 [0.20, 0.80] | 0.90 | 0.33 [0.00, 0.70] |
| opus vs humano | total | 20 | 0.95 [0.85, 1.00] | 1.00 | 0.88 [0.63, 1.00] |
| opus vs humano | base | 10 | 0.90 [0.70, 1.00] | 1.00 | 0.71 [0.41, 1.00] |
| opus vs humano | rerank | 10 | 1.00 [1.00, 1.00] | 1.00 | 1.00 [1.00, 1.00] |

Puntaje medio de cada evaluador (si = 1, parcial = 0,5, no = 0):

| Evaluador | total | base | rerank |
|---|---|---|---|
| humano | 0.17 | 0.10 | 0.25 |
| gpt-4o-mini | 0.53 | 0.50 | 0.55 |
| opus | 0.20 | 0.15 | 0.25 |

Matriz de confusión, total (filas: gpt-4o-mini; columnas: humano):

| | si | parcial | no |
|---|---|---|---|
| si | 2 | 0 | 3 |
| parcial | 0 | 3 | 8 |
| no | 0 | 0 | 4 |

Matriz de confusión, total (filas: gpt-4o-mini; columnas: opus):

| | si | parcial | no |
|---|---|---|---|
| si | 2 | 0 | 3 |
| parcial | 1 | 2 | 8 |
| no | 0 | 0 | 4 |

Matriz de confusión, total (filas: opus; columnas: humano):

| | si | parcial | no |
|---|---|---|---|
| si | 2 | 1 | 0 |
| parcial | 0 | 2 | 0 |
| no | 0 | 0 | 15 |

## Faithfulness (heurística) contra la persona

| Grupo | Ítems | Filas | Errores | Falsos sí | Falsos no | Acierto | Ítems con algún error |
|---|---|---|---|---|---|---|---|
| total | 20 | 93 | 5 | 5 | 0 | 0.95 | R02, R03, R12 |
| base | 10 | 50 | 3 | 3 | 0 | 0.94 | R02, R03 |
| rerank | 10 | 43 | 2 | 2 | 0 | 0.95 | R12 |

## Detalle por ítem

| R | Versión | Ítem | humano | gpt-4o-mini | opus | Errores heurística |
|---|---|---|---|---|---|---|
| R01 | base | syn-scipy-01 | parcial | parcial | parcial | ninguno |
| R02 | base | dep-jose-validate-01 | no | si | no | 5 |
| R03 | base | syn-scrapy-01 | no | parcial | no | 7, 8 |
| R04 | rerank | syn-pydantic-02 | no | no | no | ninguno |
| R05 | base | syn-pymupdf-02 | no | parcial | no | ninguno |
| R06 | rerank | syn-lxml-02 | si | si | si | ninguno |
| R07 | base | syn-typer-02 | parcial | parcial | si | ninguno |
| R08 | base | syn-transformers-03 | no | no | no | ninguno |
| R09 | rerank | syn-textual-02 | parcial | parcial | parcial | ninguno |
| R10 | base | dep-oauth2client-sa-01 | no | si | no | ninguno |
| R11 | rerank | dep-nose-generators-01 | no | si | no | ninguno |
| R12 | rerank | neg-oauth2client-storage-01 | no | parcial | no | 4, 8 |
| R13 | rerank | syn-matplotlib-03 | no | parcial | no | ninguno |
| R14 | rerank | dep-nose-asserts-01 | no | parcial | no | ninguno |
| R15 | base | syn-scikit-learn-02 | no | no | no | ninguno |
| R16 | base | syn-pydantic-01 | no | parcial | no | ninguno |
| R17 | rerank | dep-fuzzywuzzy-suggest-01 | no | parcial | no | ninguno |
| R18 | base | syn-matplotlib-01 | no | parcial | no | ninguno |
| R19 | rerank | syn-wand-01 | si | si | si | ninguno |
| R20 | rerank | syn-celery-02 | no | no | no | ninguno |
