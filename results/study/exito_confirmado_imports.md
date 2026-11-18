# Éxito confirmado por import (dev, reranker)

Umbral elegido: -4.515625. Confirmado = éxito automático en un script que Opus 5.5 con rúbrica marca `si` (el veredicto es del script entero y vale para sus imports). Imports sobre los positivos de dev.

| Paso | Éxito automático | Éxito confirmado | Opus en los exitosos (sí / parcial / no) | Scripts mixtos |
|---|---|---|---|---|
| Sin iterar | 0.310 (35/113) | 0.142 (16/113) | 16 / 4 / 15 | — |
| Variante 6 | 0.735 (83/113) | 0.301 (34/113) | 34 / 10 / 39 | — |
| Variante 7 | 0.726 (82/113) | 0.451 (51/113) | 51 / 8 / 23 | syn-flask-02, syn-playwright-01, syn-playwright-02 |

Scripts mixtos: juzgados con imports exitosos y no exitosos a la vez; el import exitoso cuenta como confirmado solo si Opus marcó `si` el script.
