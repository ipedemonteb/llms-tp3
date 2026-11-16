# gpt-4o-mini vs. Claude Opus 5.5 como jueces de preservación

Todos los scripts cambiados que juzgó gpt-4o-mini en las corridas permisivas de dev, juzgados también por Opus 5.5 con el mismo prompt y el mismo esquema (metricas.md §3.8). Sin estratificar: los promedios son de la población juzgada. IC 95% por bootstrap sobre ítems. Sesgo a declarar: los scripts del dataset los escribió Claude.

| Grupo | n | gpt-4o-mini | Opus 5.5 | Diferencia (gpt − Opus) | gpt más blando / igual / más estricto | Acuerdo exacto | A un nivel | Kappa |
|---|---|---|---|---|---|---|---|---|
| total | 146 | 0.59 | 0.30 | +0.28 [+0.23, +0.34] | 79 / 63 / 4 | 0.43 [0.35, 0.51] | 0.95 | 0.24 [0.16, 0.33] |
| base | 67 | 0.59 | 0.23 | +0.36 [+0.29, +0.43] | 45 / 21 / 1 | 0.31 [0.21, 0.42] | 0.94 | 0.16 [0.06, 0.26] |
| rerank | 79 | 0.58 | 0.36 | +0.22 [+0.15, +0.30] | 34 / 42 / 3 | 0.53 [0.42, 0.65] | 0.95 | 0.34 [0.21, 0.47] |

Veredictos de cada juez:

| Grupo | Juez | si | parcial | no |
|---|---|---|---|---|
| total | gpt-4o-mini | 36 | 99 | 11 |
| total | opus | 27 | 34 | 85 |
| base | gpt-4o-mini | 15 | 49 | 3 |
| base | opus | 10 | 11 | 46 |
| rerank | gpt-4o-mini | 21 | 50 | 8 |
| rerank | opus | 17 | 23 | 39 |

Matriz de confusión, total (filas: gpt-4o-mini; columnas: Opus 5.5):

| | si | parcial | no |
|---|---|---|---|
| si | 24 | 4 | 8 |
| parcial | 3 | 29 | 67 |
| no | 0 | 1 | 10 |
