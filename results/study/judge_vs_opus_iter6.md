# gpt-4o-mini vs. Claude Opus 5.5 como jueces de preservación

Los scripts cambiados de `s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4o-mini__dev__iter6` que juzgó gpt-4o-mini y que pasan el umbral elegido (-4.51562), juzgados también por Opus 5.5 con el mismo prompt y el mismo esquema. Exitosos / no exitosos: éxito del nivel 1 del script. IC 95% por bootstrap sobre ítems. Sesgo a declarar: los scripts del dataset los escribió Claude.

| Grupo | n | gpt-4o-mini | Opus 5.5 | Diferencia (gpt − Opus) | gpt más blando / igual / más estricto | Acuerdo exacto | A un nivel | Kappa |
|---|---|---|---|---|---|---|---|---|
| total | 94 | 0.78 | 0.55 | +0.23 [+0.16, +0.32] | 36 / 53 / 5 | 0.56 [0.46, 0.66] | 0.86 | 0.32 [0.20, 0.45] |
| exitosos | 76 | 0.82 | 0.63 | +0.19 [+0.10, +0.28] | 23 / 48 / 5 | 0.63 [0.51, 0.74] | 0.86 | 0.37 [0.20, 0.52] |
| no exitosos | 18 | 0.61 | 0.19 | +0.42 [+0.28, +0.56] | 13 / 5 / 0 | 0.28 [0.11, 0.50] | 0.89 | 0.06 [-0.10, 0.25] |

Veredictos de cada juez:

| Grupo | Juez | si | parcial | no |
|---|---|---|---|---|
| total | gpt-4o-mini | 53 | 41 | 0 |
| total | opus | 37 | 29 | 28 |
| exitosos | gpt-4o-mini | 49 | 27 | 0 |
| exitosos | opus | 36 | 24 | 16 |
| no exitosos | gpt-4o-mini | 4 | 14 | 0 |
| no exitosos | opus | 1 | 5 | 12 |

Matriz de confusión, total (filas: gpt-4o-mini; columnas: Opus 5.5):

| | si | parcial | no |
|---|---|---|---|
| si | 32 | 8 | 13 |
| parcial | 5 | 21 | 15 |
| no | 0 | 0 | 0 |
