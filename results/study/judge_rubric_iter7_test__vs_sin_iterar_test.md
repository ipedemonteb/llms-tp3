# Preservación de la tarea con rúbrica: `iter7_test`

`s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4.1-mini__test__iter7`: scripts cambiados que pasan el umbral, juzgados a ciegas por tres agentes de Opus 5.5 con la rúbrica común. Anclas (las de la revisión de sin iterar e `__iter6`): los agentes nuevos coinciden con la revisión anterior en 6 de 6. IC 95% por bootstrap sobre ítems.

| Grupo | n | Opus con rúbrica | si / parcial / no |
|---|---|---|---|
| total | 54 | 0.50 | 24 / 6 / 24 |
| exitosos | 42 | 0.57 | 22 / 4 / 16 |
| no exitosos | 12 | 0.25 | 2 / 2 / 8 |

| gpt-4o-mini (juez de siempre) vs. Opus con rúbrica | n | gpt | Opus | gpt − Opus | Acuerdo exacto | Kappa |
|---|---|---|---|---|---|---|
| total | 54 | 0.72 | 0.50 | +0.22 [+0.09, +0.35] | 0.37 [0.24, 0.50] | 0.15 [0.04, 0.27] |
| exitosos | 42 | 0.71 | 0.57 | +0.14 [+0.00, +0.30] | 0.38 [0.24, 0.52] | 0.14 [0.02, 0.28] |
| no exitosos | 12 | 0.75 | 0.25 | +0.50 [+0.29, +0.71] | 0.33 [0.08, 0.58] | 0.20 [0.03, 0.41] |

Pareado contra `sin_iterar_test` (Opus con rúbrica, los 40 scripts que cambiaron las dos corridas): iter7_test − sin_iterar_test = +0.40 [+0.26, +0.54]; iter7_test preserva más en 19, igual en 21, menos en 0.
