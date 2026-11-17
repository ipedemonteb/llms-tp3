# Análisis del generador: `s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4o-mini__dev__iter6`

Colección `s3-1024__e4__r1__cap150`, reranker k1-rr30, gpt-4o-mini, split dev, umbral de la corrida -1000.0000 (permisiva). Targets: 113 positivos, 9 negativos.

## Umbral elegido (pasa ≥ 95% de los positivos (regla de retrieval)): -4.5156

| Buenas | Malas | Perdidas | Cambios indebidos | Abst. correctas | Reporte engañoso | Éxito positivos | Precisión | Negative Rejection | Faithfulness | AR (b) | AR (a) | Llamadas LLM | Puntaje (λ = 2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 83 | 16 | 14 | 2 | 7 | 0 | 0.735 | 0.822 | 0.778 | 0.920 (n=94) | - (n=0) | - (n=0) | 103 | 47.0 |

AR = Answer Relevance: (b) juez directo (principal), (a) técnica de clase. Sin juicio todavía (judge_generation.py).

Umbral de máximo puntaje con cada λ: λ = 1 → -4.6719, λ = 2 → -1.5000, λ = 3 → 1.4141

## Comparación de criterios

Umbral que elige cada criterio sobre el mismo barrido. El principal es la regla de retrieval (generador.md §2.3: la de los umbrales calibrados en el estudio del retrieval). El puntaje con λ = 2 (metricas.md §4.2) fue el principal hasta el 2026-11-16: con la precisión actual del generador elige umbrales que corrigen muy pocos imports, así que queda como comparación, igual que los otros dos.

| Criterio | Umbral | Buenas | Malas | Indebidos | Éxito positivos | Precisión | Negative Rejection | Positivos que pasan | Llamadas LLM | Puntaje (λ = 2) |
|---|---|---|---|---|---|---|---|---|---|---|
| puntaje con λ = 2 | -1.5000 | 70 | 10 | 0 | 0.619 | 0.875 | 1.000 | 0.752 | 80 | 50.0 |
| máximas buenas con precisión ≥ 0.8 | -4.6719 | 84 | 16 | 2 | 0.743 | 0.824 | 0.778 | 0.965 | 104 | 48.0 |
| máximas buenas con Negative Rejection ≥ 0.85 | -2.1719 | 74 | 11 | 1 | 0.655 | 0.860 | 0.889 | 0.796 | 84 | 50.0 |
| pasa ≥ 95% de los positivos (regla de retrieval) ⭐ | -4.5156 | 83 | 16 | 2 | 0.735 | 0.822 | 0.778 | 0.956 | 103 | 47.0 |

## Por subconjunto (umbral elegido)

| Subconjunto | Positivos | Negativos | Éxito positivos | Malas | Negative Rejection | Cambios indebidos |
|---|---|---|---|---|---|---|
| manual_deprecated | 15 | 0 | 0.867 | 1 | - | 0 |
| negative | 0 | 9 | - | 0 | 0.778 | 2 |
| synthetic | 98 | 0 | 0.714 | 15 | - | 0 |

## Noise Robustness (umbral elegido; positivos con un paquete aceptable en el contexto)

| Grupo | Targets | Éxito |
|---|---|---|
| limpio | 32 | 0.969 |
| ruido_aceptable_primero | 51 | 0.863 |
| ruido_aceptable_despues | 17 | 0.471 |
| noisy | 68 | 0.765 |
| sin_aceptable_en_contexto | 8 | 0.000 |

Noise Robustness (éxito con ruido / éxito limpio): 0.789

## Barrido del umbral

| Umbral | Buenas | Malas | Perdidas | Indebidos | Abst. ok | Engañoso | Éxito + | Precisión | Neg. Rej. | Faithf. | AR (b) | AR (a) | Llamadas | U(λ=1) | U(λ=2) | U(λ=3) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| −∞ | 84 | 16 | 13 | 5 | 4 | 0 | 0.743 | 0.800 | 0.444 | 0.909 | - | - | 115 | 63.0 | 42.0 | 21.0 |
| -6.9805 | 84 | 16 | 13 | 5 | 4 | 0 | 0.743 | 0.800 | 0.444 | 0.909 | - | - | 112 | 63.0 | 42.0 | 21.0 |
| -6.0469 | 84 | 16 | 13 | 3 | 6 | 0 | 0.743 | 0.816 | 0.667 | 0.918 | - | - | 108 | 65.0 | 46.0 | 27.0 |
| -4.6719 | 84 | 16 | 13 | 2 | 7 | 0 | 0.743 | 0.824 | 0.778 | 0.918 | - | - | 104 | 66.0 | 48.0 | 30.0 |
| -4.5156 ⭐ | 83 | 16 | 14 | 2 | 7 | 0 | 0.735 | 0.822 | 0.778 | 0.920 | - | - | 103 | 65.0 | 47.0 | 29.0 |
| -3.8438 | 81 | 16 | 16 | 2 | 7 | 0 | 0.717 | 0.818 | 0.778 | 0.921 | - | - | 100 | 63.0 | 45.0 | 27.0 |
| -3.2656 | 79 | 14 | 20 | 2 | 7 | 0 | 0.699 | 0.832 | 0.778 | 0.924 | - | - | 96 | 63.0 | 47.0 | 31.0 |
| -2.9766 | 78 | 13 | 22 | 2 | 7 | 0 | 0.690 | 0.839 | 0.778 | 0.922 | - | - | 92 | 63.0 | 48.0 | 33.0 |
| -2.3203 | 75 | 12 | 26 | 2 | 7 | 0 | 0.664 | 0.843 | 0.778 | 0.927 | - | - | 88 | 61.0 | 47.0 | 33.0 |
| -2.1719 | 74 | 11 | 28 | 1 | 8 | 0 | 0.655 | 0.860 | 0.889 | 0.925 | - | - | 84 | 62.0 | 50.0 | 38.0 |
| -1.8281 | 70 | 11 | 32 | 1 | 8 | 0 | 0.619 | 0.854 | 0.889 | 0.923 | - | - | 82 | 58.0 | 46.0 | 34.0 |
| -1.3750 | 68 | 10 | 35 | 0 | 9 | 0 | 0.602 | 0.872 | 1.000 | 0.937 | - | - | 78 | 58.0 | 48.0 | 38.0 |
| -1.0000 | 66 | 9 | 38 | 0 | 9 | 0 | 0.584 | 0.880 | 1.000 | 0.941 | - | - | 74 | 57.0 | 48.0 | 39.0 |
| -0.5469 | 63 | 9 | 41 | 0 | 9 | 0 | 0.558 | 0.875 | 1.000 | 0.938 | - | - | 70 | 54.0 | 45.0 | 36.0 |
| 0.0625 | 59 | 9 | 45 | 0 | 9 | 0 | 0.522 | 0.868 | 1.000 | 0.937 | - | - | 66 | 50.0 | 41.0 | 32.0 |
| 0.2969 | 58 | 6 | 49 | 0 | 9 | 0 | 0.513 | 0.906 | 1.000 | 0.940 | - | - | 62 | 52.0 | 46.0 | 40.0 |
| 0.8594 | 56 | 4 | 53 | 0 | 9 | 0 | 0.496 | 0.933 | 1.000 | 0.941 | - | - | 58 | 52.0 | 48.0 | 44.0 |
| 1.4141 | 53 | 3 | 57 | 0 | 9 | 0 | 0.469 | 0.946 | 1.000 | 0.938 | - | - | 54 | 50.0 | 47.0 | 44.0 |
| 1.7500 | 48 | 3 | 62 | 0 | 9 | 0 | 0.425 | 0.941 | 1.000 | 0.951 | - | - | 49 | 45.0 | 42.0 | 39.0 |
| 2.1875 | 44 | 3 | 66 | 0 | 9 | 0 | 0.389 | 0.936 | 1.000 | 0.965 | - | - | 46 | 41.0 | 38.0 | 35.0 |
| 2.6016 | 41 | 2 | 70 | 0 | 9 | 0 | 0.363 | 0.953 | 1.000 | 0.962 | - | - | 43 | 39.0 | 37.0 | 35.0 |
| 3.2734 | 37 | 2 | 74 | 0 | 9 | 0 | 0.327 | 0.949 | 1.000 | 0.958 | - | - | 39 | 35.0 | 33.0 | 31.0 |
| 3.9531 | 34 | 2 | 77 | 0 | 9 | 0 | 0.301 | 0.944 | 1.000 | 0.962 | - | - | 35 | 32.0 | 30.0 | 28.0 |
| 4.2109 | 31 | 1 | 81 | 0 | 9 | 0 | 0.274 | 0.969 | 1.000 | 0.971 | - | - | 31 | 30.0 | 29.0 | 28.0 |
| 4.3594 | 27 | 1 | 85 | 0 | 9 | 0 | 0.239 | 0.964 | 1.000 | 0.984 | - | - | 28 | 26.0 | 25.0 | 24.0 |
| 4.4922 | 23 | 1 | 89 | 0 | 9 | 0 | 0.204 | 0.958 | 1.000 | 0.981 | - | - | 24 | 22.0 | 21.0 | 20.0 |
| 5.6562 | 19 | 1 | 93 | 0 | 9 | 0 | 0.168 | 0.950 | 1.000 | 0.977 | - | - | 20 | 18.0 | 17.0 | 16.0 |
| 5.8672 | 15 | 1 | 97 | 0 | 9 | 0 | 0.133 | 0.938 | 1.000 | 1.000 | - | - | 16 | 14.0 | 13.0 | 12.0 |
| 6.2031 | 12 | 0 | 101 | 0 | 9 | 0 | 0.106 | 1.000 | 1.000 | 1.000 | - | - | 12 | 12.0 | 12.0 | 12.0 |
| 6.4766 | 8 | 0 | 105 | 0 | 9 | 0 | 0.071 | 1.000 | 1.000 | 1.000 | - | - | 9 | 8.0 | 8.0 | 8.0 |
| 6.9531 | 5 | 0 | 108 | 0 | 9 | 0 | 0.044 | 1.000 | 1.000 | 1.000 | - | - | 5 | 5.0 | 5.0 | 5.0 |
| 7.9844 | 1 | 0 | 112 | 0 | 9 | 0 | 0.009 | 1.000 | 1.000 | 1.000 | - | - | 1 | 1.0 | 1.0 | 1.0 |
| +∞ | 0 | 0 | 113 | 0 | 9 | 0 | 0.000 | - | 1.000 | - | - | - | 0 | 0.0 | 0.0 | 0.0 |

(Se muestran 33 de 122 umbrales; todos están en el JSON.)
