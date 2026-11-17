# Preservación de la tarea: Opus 5.5 con rúbrica común como referencia

Scripts cambiados que pasan el umbral elegido, juzgados a ciegas por tres agentes de Opus 5.5 con la rúbrica común (tprag.eval.judge.PRESERVE_RUBRIC), con las dos corridas intercaladas. gpt-4o-mini: el juez de siempre (sin rúbrica) y el mismo prompt con la rúbrica. IC 95% por bootstrap sobre ítems.

Anclas (juzgadas por los tres agentes): coinciden en 6 de 6.

## `rerank` (s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4o-mini__dev)

| Grupo | n | Opus con rúbrica | si / parcial / no |
|---|---|---|---|
| total | 77 | 0.23 | 15 / 5 / 57 |
| exitosos | 33 | 0.52 | 15 / 4 / 14 |
| no exitosos | 44 | 0.01 | 0 / 1 / 43 |

| Juez gpt-4o-mini vs. Opus con rúbrica | Grupo | n | gpt | Opus | gpt − Opus | Acuerdo exacto | Kappa |
|---|---|---|---|---|---|---|---|
| sin rúbrica | total | 77 | 0.58 | 0.23 | +0.36 [+0.29, +0.42] | 0.34 [0.23, 0.44] | 0.20 [0.11, 0.30] |
| sin rúbrica | exitosos | 33 | 0.77 | 0.52 | +0.26 [+0.15, +0.36] | 0.55 [0.36, 0.73] | 0.33 [0.17, 0.50] |
| sin rúbrica | no exitosos | 44 | 0.44 | 0.01 | +0.43 [+0.36, +0.50] | 0.18 [0.07, 0.30] | 0.01 [0.00, 0.04] |
| con rúbrica | total | 77 | 0.56 | 0.23 | +0.33 [+0.26, +0.40] | 0.43 [0.31, 0.55] | 0.25 [0.15, 0.36] |
| con rúbrica | exitosos | 33 | 0.76 | 0.52 | +0.24 [+0.14, +0.36] | 0.61 [0.42, 0.76] | 0.39 [0.21, 0.59] |
| con rúbrica | no exitosos | 44 | 0.41 | 0.01 | +0.40 [+0.31, +0.49] | 0.30 [0.16, 0.43] | 0.02 [0.00, 0.07] |

## `iter6` (s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4o-mini__dev__iter6)

| Grupo | n | Opus con rúbrica | si / parcial / no |
|---|---|---|---|
| total | 94 | 0.41 | 33 / 12 / 49 |
| exitosos | 76 | 0.49 | 32 / 10 / 34 |
| no exitosos | 18 | 0.11 | 1 / 2 / 15 |

| Juez gpt-4o-mini vs. Opus con rúbrica | Grupo | n | gpt | Opus | gpt − Opus | Acuerdo exacto | Kappa |
|---|---|---|---|---|---|---|---|
| sin rúbrica | total | 94 | 0.78 | 0.41 | +0.37 [+0.28, +0.45] | 0.38 [0.29, 0.48] | 0.17 [0.09, 0.26] |
| sin rúbrica | exitosos | 76 | 0.82 | 0.49 | +0.34 [+0.24, +0.43] | 0.45 [0.34, 0.55] | 0.19 [0.09, 0.30] |
| sin rúbrica | no exitosos | 18 | 0.61 | 0.11 | +0.50 [+0.39, +0.61] | 0.11 [0.00, 0.28] | 0.01 [-0.12, 0.15] |
| con rúbrica | total | 94 | 0.81 | 0.41 | +0.39 [+0.31, +0.47] | 0.41 [0.32, 0.51] | 0.19 [0.11, 0.27] |
| con rúbrica | exitosos | 76 | 0.86 | 0.49 | +0.37 [+0.28, +0.46] | 0.47 [0.36, 0.59] | 0.20 [0.10, 0.30] |
| con rúbrica | no exitosos | 18 | 0.61 | 0.11 | +0.50 [+0.36, +0.64] | 0.17 [0.00, 0.33] | 0.04 [-0.09, 0.18] |
