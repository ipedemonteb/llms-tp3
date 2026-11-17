# Diagnóstico del generador: `s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4.1-mini__dev__iter7`

Umbral elegido (pasa ≥ 95% de los positivos (regla de retrieval)): -4.5156. Positivos: 113; exitosos: 82; no exitosos: 31.

## Por qué fallan los imports no exitosos (una causa por import: la primera en este orden)

| Causa | Imports |
|---|---|
| Debajo del umbral (no llega al LLM) | 5 |
| Abstención por la post-validación (después del reintento) | 5 |
| El LLM se abstiene, con la correcta en el contexto | 11 |
| El LLM se abstiene, sin la correcta en el contexto | 5 |
| Elige mal la librería, con la correcta en el contexto | 2 |
| Elige mal la librería, sin la correcta en el contexto (retrieval) | 3 |
| **Total** | **31** |

## Negativos cambiados (cambios indebidos)

Ninguno.

## Cuánto reescribe (scripts entregados cambiados; líneas del original cambiadas)

86 scripts: mediana 5.5, máximo 18, más de 10 líneas: 13.

## Comparación pareada: esta corrida (A) vs `s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4o-mini__dev` (B)

IC 95% bootstrap por script (10.000 remuestreos, semilla 2026); empate si el intervalo de la diferencia incluye el cero.

| Métrica | A | B | A − B | IC 95% | Veredicto |
|---|---|---|---|---|---|
| Éxito en positivos | 0.726 | 0.310 | +0.416 | [+0.310; +0.522] | mejor |
| Precisión | 0.911 | 0.417 | +0.494 | [+0.375; +0.610] | mejor |
| Negative Rejection | 1.000 | 0.889 | +0.111 | [+0.000; +0.375] | empate |

Positivos que A corrige y B no: 51; que B corrige y A no: 4.

## Comparación pareada: esta corrida (A) vs `s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4o-mini__dev__iter6` (B)

IC 95% bootstrap por script (10.000 remuestreos, semilla 2026); empate si el intervalo de la diferencia incluye el cero.

| Métrica | A | B | A − B | IC 95% | Veredicto |
|---|---|---|---|---|---|
| Éxito en positivos | 0.726 | 0.735 | -0.009 | [-0.080; +0.062] | empate |
| Precisión | 0.911 | 0.822 | +0.089 | [+0.016; +0.170] | mejor |
| Negative Rejection | 1.000 | 0.778 | +0.222 | [+0.000; +0.556] | empate |

Positivos que A corrige y B no: 8; que B corrige y A no: 9.
