# Costo computacional y uso de tokens

Generado por `src/scripts/analyze_cost.py` el 2026-11-15. Sin API: reportes de indexado, corridas permisivas de dev del generador (todos los scripts con imports problemáticos llegan al LLM) y manifiesto del estudio del reranker. Precios de gpt-4o-mini: 0,15 / 0,60 USD por millón de tokens de entrada / salida.

## 1. Indexado (una vez por configuración)

| Colección | Embedder | Dónde | Casos de uso | Tokens del corpus | Embebidos ahora | Del caché | USD | Segundos | Textos/s | USD del corpus completo |
|---|---|---|---|---|---|---|---|---|---|---|
| `s0-1024__e1__r1__cap150` | text-embedding-3-small | API | 6.460 | 5.361.047 | 4.882 | 1.578 | 0,0887 | 262 | 18,6 | 0,1072 |
| `s1-1024__e1__r1__cap150` | text-embedding-3-small | API | 6.576 | 4.975.413 | 6.255 | 321 | 0,0954 | 279 | 22,4 | 0,0995 |
| `s2-1024__e1__r1__cap150` | text-embedding-3-small | API | 8.667 | 3.193.551 | 885 | 7.782 | 0,0119 | 32 | 27,4 | 0,0639 |
| `s3-1024__e1__r1__cap150` | text-embedding-3-small | API | 8.633 | 3.242.739 | 8.438 | 195 | 0,0639 | 138 | 61,0 | 0,0649 |
| `s3-1024__e4__r1__cap150` | jinaai/jina-embeddings-v2-base-code | local | 8.633 | 3.242.739 | 4.281 | 4.352 | 0,0000 | 772 | 5,5 | 0,0000 |

Con la API (E1, text-embedding-3-small) indexar las 101 librerías cuesta ~0,065 USD; con el embedder local elegido (E4, jina-code) cuesta 0 USD y lo que se paga es tiempo de máquina.

## 2. Generador: tokens por llamada a gpt-4o-mini

Entrada contada con `o200k_base` sobre el mensaje reconstruido + 482 tokens fijos (esquema JSON y formato), calibrado con 27 llamadas del caché (rango del sobrecosto: 482-482). Salida reconstruida × 1,13 (rango real/reconstruida 0,98-1,55; error medio de la salida estimada sobre esas llamadas: 9%). Las llamadas del caché usan el uso real.

| Versión | Llamadas | Exactas | Entrada media | p95 | Salida media | p95 | USD / llamada | p95 | Salida / USD | USD de la corrida |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 115 | 25 | 3.355 | 5.098 | 255 | 467 | 0,00066 | 0,00094 | 23% | 0,0755 |
| rerank | 115 | 2 | 3.376 | 4.850 | 283 | 519 | 0,00068 | 0,00101 | 25% | 0,0778 |

De qué está hecha la entrada (tokens medios por llamada):

| Versión | Instrucciones | Esquema y formato | Script | Fragmentos recuperados |
|---|---|---|---|---|
| base | 505 | 482 | 331 | 2.037 |
| rerank | 505 | 482 | 331 | 2.058 |

## 3. Costo por cada 1.000 scripts con imports problemáticos

Los scripts sin imports problemáticos no llaman al LLM ni al reranker (solo el validador, gratis). Con el umbral, solo llegan al LLM los imports con un fragmento suficientemente parecido.

| Versión | Umbral | Valor | Llamadas al LLM | USD / 1.000 scripts |
|---|---|---|---|---|
| base | elegido | 0,7491 | 2/115 (2%) | 0,011 |
| base | permisivo_95 | 0,4547 | 105/115 (91%) | 0,599 |
| rerank | elegido | 6,2500 | 10/115 (9%) | 0,059 |
| rerank | permisivo_95 | -4,5156 | 103/115 (90%) | 0,606 |

## 4. Latencia del reranker (por consulta = por import problemático)

Medido en el estudio (`rerank_full_dev_estudio_manifest.json`), macOS-26.6.2-arm64-arm-64bit, float16 en mps, de a un par. Un par por candidato reordenado.

| Reranker | K | s / par | s / consulta | Memoria pico (MB) |
|---|---|---|---|---|
| Qwen3-Reranker-0.6B | 10 | 0,537 | 5,4 | 2.289 |
| Qwen3-Reranker-0.6B | 20 | 0,537 | 10,7 | 2.289 |
| Qwen3-Reranker-0.6B ⭐ | 30 | 0,537 | 16,1 | 2.289 |
| Qwen3-Reranker-0.6B | 50 | 0,537 | 26,9 | 2.289 |
| gte-reranker-modernbert-base | 10 | 0,123 | 1,2 | 1.050 |
| gte-reranker-modernbert-base | 20 | 0,123 | 2,5 | 1.050 |
| gte-reranker-modernbert-base | 30 | 0,123 | 3,7 | 1.050 |
| gte-reranker-modernbert-base | 50 | 0,123 | 6,1 | 1.050 |

## 5. Gasto de API del TP

| Concepto | Llamadas / textos | Tokens | USD |
|---|---|---|---|
| Embeddings con la API (todos los índices E1) | 20.647 | 13.040.294 | 0,2608 |
| Generador gpt-4o-mini (caché local) | 443 | 1.612.721 | 0,2948 |
| **Total registrado** | | | **0,5556** de 10 USD |

El caché local tiene las llamadas hechas en esta máquina; las corridas de otros integrantes y el juez (caché propio, `judge.sqlite`) se suman con sus cachés. Las llamadas repetidas no se pagan dos veces (caché por prompt).

## 6. Qué no está medido

- Latencia del LLM: las corridas salen del caché (el tiempo guardado es el de leerlo).
- Retrieval (embedding de la consulta con E4 + búsqueda en Chroma): no se cronometró por consulta.
- Las cifras del reranker son de una Mac M3 (MPS); en CPU sin aceleración serían más altas.
