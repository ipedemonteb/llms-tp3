# Diagnóstico del generador: `s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4o-mini__dev`

Umbral elegido (pasa ≥ 95% de los positivos (regla de retrieval)): -4.5156. Positivos: 113; exitosos: 35; no exitosos: 78.

## Por qué fallan los imports no exitosos (una causa por import: la primera en este orden)

| Causa | Imports |
|---|---|
| Debajo del umbral (no llega al LLM) | 5 |
| El LLM se abstiene, con la correcta en el contexto | 2 |
| El LLM se abstiene, sin la correcta en el contexto | 1 |
| No aplica los cambios (el "antes" no está en el script) | 22 |
| El código corregido no parsea | 6 |
| Migra a medias (queda el import viejo o nombres sin definir) | 26 |
| Elige mal la librería, con la correcta en el contexto | 10 |
| Elige mal la librería, sin la correcta en el contexto (retrieval) | 6 |
| **Total** | **78** |

## Negativos cambiados (cambios indebidos)

- `neg-oauth2client-storage-01` (oauth2client.file → google-auth-oauthlib)
