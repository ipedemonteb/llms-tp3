# Preservación de la tarea con rúbrica: `sin_iterar_test`

`s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4o-mini__test`: scripts cambiados que pasan el umbral, juzgados a ciegas por tres agentes de Opus 5.5 con la rúbrica común. Anclas (las de la revisión de sin iterar e `__iter6`): los agentes nuevos coinciden con la revisión anterior en 6 de 6. IC 95% por bootstrap sobre ítems.

| Grupo | n | Opus con rúbrica | si / parcial / no |
|---|---|---|---|
| total | 58 | 0.09 | 4 / 3 / 51 |
| exitosos | 22 | 0.23 | 4 / 2 / 16 |
| no exitosos | 36 | 0.01 | 0 / 1 / 35 |

| gpt-4o-mini (juez de siempre) vs. Opus con rúbrica | n | gpt | Opus | gpt − Opus | Acuerdo exacto | Kappa |
|---|---|---|---|---|---|---|
| total | 58 | 0.57 | 0.09 | +0.47 [+0.39, +0.56] | 0.24 [0.14, 0.36] | 0.10 [0.03, 0.19] |
| exitosos | 22 | 0.73 | 0.23 | +0.50 [+0.34, +0.66] | 0.27 [0.09, 0.45] | 0.16 [0.05, 0.30] |
| no exitosos | 36 | 0.47 | 0.01 | +0.46 [+0.36, +0.56] | 0.22 [0.08, 0.36] | 0.02 [0.00, 0.07] |

Sin scripts en común con `__iter6` (otro split): no hay comparación pareada.
