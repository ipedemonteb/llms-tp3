# Faithfulness: heurística vs. Claude Opus 5.5

`s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4o-mini__dev`: los scripts con Faithfulness definida sobre el umbral elegido, juzgados a ciegas por tres agentes de Opus 5.5 con la rúbrica de docs/agents/prompt_opus_faithfulness.md (¿el uso de cada nombre está respaldado por los fragmentos?). Unidad: nombre distinto por script. IC 95% por bootstrap sobre scripts.

Anclas: 4 scripts, 21 nombres; los tres agentes coinciden en 21.

| Scripts | Nombres | Acierto de la heurística | Kappa |
|---|---|---|---|
| 71 | 360 | 0.92 [0.87, 0.97] | 0.55 |

| | Opus: respaldado | Opus: no respaldado |
|---|---|---|
| Heurística: fiel | 312 | 28 (falso fiel) |
| Heurística: no fiel | 0 (falso no fiel) | 20 |

Faithfulness promedio por script: heurística 0.95, Opus 0.88.
