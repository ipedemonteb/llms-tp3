# Diagnóstico del generador: `s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4o-mini__test`

Umbral (fijo, calibrado en dev; no se elige): -4.5156. Positivos: 122; exitosos: 22; no exitosos: 100.

## Por qué fallan los imports no exitosos (una causa por import: la primera en este orden)

| Causa | Imports |
|---|---|
| Debajo del umbral (no llega al LLM) | 40 |
| El LLM se abstiene, con la correcta en el contexto | 5 |
| No aplica los cambios (el "antes" no está en el script) | 18 |
| El código corregido no parsea | 2 |
| Migra a medias (queda el import viejo o nombres sin definir) | 21 |
| Elige mal la librería, con la correcta en el contexto | 6 |
| Elige mal la librería, sin la correcta en el contexto (retrieval) | 8 |
| **Total** | **100** |

## Negativos cambiados (cambios indebidos)

- `neg-toml-dump-01` (toml → dynaconf)
