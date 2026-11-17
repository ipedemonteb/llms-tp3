# Diagnóstico del generador: `s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4.1-mini__test__iter7`

Umbral (fijo, calibrado en dev; no se elige): -4.5156. Positivos: 122; exitosos: 44; no exitosos: 78.

## Por qué fallan los imports no exitosos (una causa por import: la primera en este orden)

| Causa | Imports |
|---|---|
| Debajo del umbral (no llega al LLM) | 40 |
| Abstención por la post-validación (después del reintento) | 10 |
| El LLM se abstiene, con la correcta en el contexto | 13 |
| El LLM se abstiene, sin la correcta en el contexto | 4 |
| Migra a medias (queda el import viejo o nombres sin definir) | 1 |
| Elige mal la librería, con la correcta en el contexto | 3 |
| Elige mal la librería, sin la correcta en el contexto (retrieval) | 7 |
| **Total** | **78** |

## Negativos cambiados (cambios indebidos)

- `neg-toml-dump-01` (toml → dynaconf)

## Cuánto reescribe (scripts entregados cambiados; líneas del original cambiadas)

54 scripts: mediana 7, máximo 15, más de 10 líneas: 8.

## Comparación pareada: esta corrida (A) vs `s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4o-mini__test` (B)

IC 95% bootstrap por script (10.000 remuestreos, semilla 2026); empate si el intervalo de la diferencia incluye el cero.

| Métrica | A | B | A − B | IC 95% | Veredicto |
|---|---|---|---|---|---|
| Éxito en positivos | 0.361 | 0.180 | +0.180 | [+0.089; +0.272] | mejor |
| Precisión | 0.786 | 0.367 | +0.419 | [+0.271; +0.564] | mejor |
| Negative Rejection | 0.900 | 0.900 | +0.000 | [+0.000; +0.000] | empate |

Positivos que A corrige y B no: 27; que B corrige y A no: 5.
