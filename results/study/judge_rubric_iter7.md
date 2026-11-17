# Preservación de la tarea con rúbrica: `iter7`

`s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4.1-mini__dev__iter7`: scripts cambiados que pasan el umbral, juzgados a ciegas por tres agentes de Opus 5.5 con la rúbrica común. Anclas (las de la revisión de sin iterar e `__iter6`): los agentes nuevos coinciden con la revisión anterior en 6 de 6. IC 95% por bootstrap sobre ítems.

| Grupo | n | Opus con rúbrica | si / parcial / no |
|---|---|---|---|
| total | 83 | 0.67 | 51 / 9 / 23 |
| exitosos | 75 | 0.70 | 49 / 7 / 19 |
| no exitosos | 8 | 0.38 | 2 / 2 / 4 |

| gpt-4o-mini (juez de siempre) vs. Opus con rúbrica | n | gpt | Opus | gpt − Opus | Acuerdo exacto | Kappa |
|---|---|---|---|---|---|---|
| total | 83 | 0.80 | 0.67 | +0.13 [+0.03, +0.22] | 0.52 [0.41, 0.63] | 0.19 [0.06, 0.31] |
| exitosos | 75 | 0.81 | 0.70 | +0.11 [+0.01, +0.21] | 0.53 [0.43, 0.64] | 0.17 [0.04, 0.31] |
| no exitosos | 8 | 0.69 | 0.38 | +0.31 [+0.00, +0.69] | 0.38 [0.12, 0.75] | 0.17 [-0.08, 0.50] |

Pareado contra `__iter6` (Opus con rúbrica, los 80 scripts que cambiaron las dos corridas): iter7 − iter6 = +0.20 [+0.10, +0.30]; iter7 preserva más en 25, igual en 51, menos en 4.
