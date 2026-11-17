# Diagnóstico del generador: `s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4o-mini__dev__iter6`

Umbral elegido (pasa ≥ 95% de los positivos (regla de retrieval)): -4.5156. Positivos: 113; exitosos: 83; no exitosos: 30.

## Por qué fallan los imports no exitosos (una causa por import: la primera en este orden)

| Causa | Imports |
|---|---|
| Debajo del umbral (no llega al LLM) | 5 |
| Respuesta cortada por el tope de salida | 1 |
| Abstención por la post-validación (después del reintento) | 1 |
| El LLM se abstiene, con la correcta en el contexto | 4 |
| El LLM se abstiene, sin la correcta en el contexto | 3 |
| Elige mal la librería, con la correcta en el contexto | 11 |
| Elige mal la librería, sin la correcta en el contexto (retrieval) | 5 |
| **Total** | **30** |

## Negativos cambiados (cambios indebidos)

- `neg-oauth2client-storage-01` (oauth2client.file → google-auth-oauthlib)
- `neg-zipcrypt-01` (zipcryptor → pycryptodome)

## Cuánto reescribe (scripts entregados cambiados; líneas del original cambiadas)

98 scripts: mediana 6, máximo 16, más de 10 líneas: 15.

## Comparación pareada: esta corrida (A) vs `s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4o-mini__dev` (B)

IC 95% bootstrap por script (10.000 remuestreos, semilla 2026); empate si el intervalo de la diferencia incluye el cero.

| Métrica | A | B | A − B | IC 95% | Veredicto |
|---|---|---|---|---|---|
| Éxito en positivos | 0.735 | 0.310 | +0.425 | [+0.323; +0.527] | mejor |
| Precisión | 0.822 | 0.417 | +0.405 | [+0.285; +0.523] | mejor |
| Negative Rejection | 0.778 | 0.889 | -0.111 | [-0.375; +0.000] | empate |

Positivos que A corrige y B no: 50; que B corrige y A no: 2.

## Comparación pareada: esta corrida (A) vs `s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4o-mini__dev__iter3` (B)

IC 95% bootstrap por script (10.000 remuestreos, semilla 2026); empate si el intervalo de la diferencia incluye el cero.

| Métrica | A | B | A − B | IC 95% | Veredicto |
|---|---|---|---|---|---|
| Éxito en positivos | 0.735 | 0.699 | +0.035 | [-0.017; +0.093] | empate |
| Precisión | 0.822 | 0.798 | +0.024 | [-0.024; +0.073] | empate |
| Negative Rejection | 0.778 | 0.778 | +0.000 | [+0.000; +0.000] | empate |

Positivos que A corrige y B no: 6; que B corrige y A no: 2.
