# Éxito confirmado por import (test, reranker)

Umbral elegido: -4.515625. Confirmado = éxito automático en un script que Opus 5.5 con rúbrica marca `si` (el veredicto es del script entero y vale para sus imports). Imports sobre los positivos del split.

| Paso | Éxito automático | Éxito confirmado | Opus en los exitosos (sí / parcial / no) | Scripts mixtos |
|---|---|---|---|---|
| Sin iterar | 0.180 (22/122) | 0.033 (4/122) | 4 / 2 / 16 | — |
| Variante 7 | 0.361 (44/122) | 0.180 (22/122) | 22 / 4 / 18 | — |

Scripts mixtos: juzgados con imports exitosos y no exitosos a la vez; el import exitoso cuenta como confirmado solo si Opus marcó `si` el script.
