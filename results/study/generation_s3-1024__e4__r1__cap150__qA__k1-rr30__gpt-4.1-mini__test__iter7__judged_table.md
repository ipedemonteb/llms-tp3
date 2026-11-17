# Análisis del generador: `s3-1024__e4__r1__cap150__qA__k1-rr30__gpt-4.1-mini__test__iter7`

Colección `s3-1024__e4__r1__cap150`, reranker k1-rr30, gpt-4.1-mini, split test, umbral de la corrida -4.5156 (NO permisiva). Targets: 122 positivos, 10 negativos.

## Umbral elegido (umbral fijo, calibrado en dev; no se elige): -4.5156

| Buenas | Malas | Perdidas | Cambios indebidos | Abst. correctas | Reporte engañoso | Éxito positivos | Precisión | Negative Rejection | Faithfulness | AR (b) | AR (a) | Llamadas LLM | Puntaje (λ = 2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 44 | 11 | 67 | 1 | 9 | 0 | 0.361 | 0.786 | 0.900 | 0.911 (n=54) | 0.722 (n=54) | 0.531 (n=54) | 82 | 20.0 |

AR = Answer Relevance: (b) juez directo (principal), (a) técnica de clase. Juzgados 54 scripts.

Umbral de máximo puntaje con cada λ: λ = 1 → -4.1991, λ = 2 → -3.6417, λ = 3 → -3.6417

## Comparación de criterios

Umbral que elige cada criterio sobre el mismo barrido. El principal es la regla de retrieval (generador.md §2.3: la de los umbrales calibrados en el estudio del retrieval). El puntaje con λ = 2 (metricas.md §4.2) fue el principal hasta el 2026-11-16: con la precisión actual del generador elige umbrales que corrigen muy pocos imports, así que queda como comparación, igual que los otros dos.

| Criterio | Umbral | Buenas | Malas | Indebidos | Éxito positivos | Precisión | Negative Rejection | Positivos que pasan | Llamadas LLM | Puntaje (λ = 2) |
|---|---|---|---|---|---|---|---|---|---|---|
| puntaje con λ = 2 | -3.6417 | 37 | 8 | 0 | 0.303 | 0.822 | 1.000 | 0.541 | 67 | 21.0 |
| máximas buenas con precisión ≥ 0.8 | -3.6417 | 37 | 8 | 0 | 0.303 | 0.822 | 1.000 | 0.541 | 67 | 21.0 |
| máximas buenas con Negative Rejection ≥ 0.85 | -4.2489 | 44 | 11 | 1 | 0.361 | 0.786 | 0.900 | 0.672 | 82 | 20.0 |
| pasa ≥ 95% de los positivos (regla de retrieval) | -4.5156 | 44 | 11 | 1 | 0.361 | 0.786 | 0.900 | 0.672 | 82 | 20.0 |

## Por subconjunto (umbral elegido)

| Subconjunto | Positivos | Negativos | Éxito positivos | Malas | Negative Rejection | Cambios indebidos |
|---|---|---|---|---|---|---|
| manual_deprecated | 16 | 0 | 0.500 | 2 | - | 0 |
| negative | 0 | 10 | - | 0 | 0.900 | 1 |
| synthetic | 106 | 0 | 0.340 | 9 | - | 0 |

## Noise Robustness (umbral elegido; positivos con un paquete aceptable en el contexto)

| Grupo | Targets | Éxito |
|---|---|---|
| limpio | 17 | 0.706 |
| ruido_aceptable_primero | 31 | 0.677 |
| ruido_aceptable_despues | 21 | 0.476 |
| noisy | 52 | 0.596 |
| sin_aceptable_en_contexto | 13 | 0.077 |

Noise Robustness (éxito con ruido / éxito limpio): 0.845

## Barrido del umbral

| Umbral | Buenas | Malas | Perdidas | Indebidos | Abst. ok | Engañoso | Éxito + | Precisión | Neg. Rej. | Faithf. | AR (b) | AR (a) | Llamadas | U(λ=1) | U(λ=2) | U(λ=3) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| -4.5156 ⭐ | 44 | 11 | 67 | 1 | 9 | 0 | 0.361 | 0.786 | 0.900 | 0.911 | 0.722 | 0.531 | 82 | 32.0 | 20.0 | 8.0 |
| -4.1991 | 43 | 11 | 68 | 0 | 10 | 0 | 0.352 | 0.796 | 1.000 | 0.911 | 0.731 | 0.534 | 80 | 32.0 | 21.0 | 10.0 |
| -4.0416 | 42 | 11 | 69 | 0 | 10 | 0 | 0.344 | 0.792 | 1.000 | 0.909 | 0.735 | 0.532 | 77 | 31.0 | 20.0 | 9.0 |
| -3.9330 | 42 | 11 | 69 | 0 | 10 | 0 | 0.344 | 0.792 | 1.000 | 0.909 | 0.735 | 0.532 | 75 | 31.0 | 20.0 | 9.0 |
| -3.8410 | 40 | 11 | 71 | 0 | 10 | 0 | 0.328 | 0.784 | 1.000 | 0.908 | 0.745 | 0.533 | 72 | 29.0 | 18.0 | 7.0 |
| -3.7819 | 38 | 10 | 74 | 0 | 10 | 0 | 0.311 | 0.792 | 1.000 | 0.904 | 0.734 | 0.534 | 70 | 28.0 | 18.0 | 8.0 |
| -3.6417 | 37 | 8 | 77 | 0 | 10 | 0 | 0.303 | 0.822 | 1.000 | 0.904 | 0.739 | 0.534 | 67 | 29.0 | 21.0 | 13.0 |
| -3.4552 | 35 | 8 | 79 | 0 | 10 | 0 | 0.287 | 0.814 | 1.000 | 0.901 | 0.738 | 0.539 | 64 | 27.0 | 19.0 | 11.0 |
| -3.2949 | 32 | 8 | 82 | 0 | 10 | 0 | 0.262 | 0.800 | 1.000 | 0.900 | 0.718 | 0.539 | 61 | 24.0 | 16.0 | 8.0 |
| -2.9514 | 30 | 8 | 84 | 0 | 10 | 0 | 0.246 | 0.789 | 1.000 | 0.901 | 0.730 | 0.539 | 58 | 22.0 | 14.0 | 6.0 |
| -2.8025 | 30 | 8 | 84 | 0 | 10 | 0 | 0.246 | 0.789 | 1.000 | 0.901 | 0.730 | 0.539 | 55 | 22.0 | 14.0 | 6.0 |
| -2.7378 | 28 | 8 | 86 | 0 | 10 | 0 | 0.230 | 0.778 | 1.000 | 0.908 | 0.729 | 0.539 | 52 | 20.0 | 12.0 | 4.0 |
| -2.3732 | 27 | 8 | 87 | 0 | 10 | 0 | 0.221 | 0.771 | 1.000 | 0.906 | 0.721 | 0.539 | 49 | 19.0 | 11.0 | 3.0 |
| -2.1679 | 26 | 6 | 90 | 0 | 10 | 0 | 0.213 | 0.812 | 1.000 | 0.918 | 0.726 | 0.538 | 46 | 20.0 | 14.0 | 8.0 |
| -2.0690 | 25 | 6 | 91 | 0 | 10 | 0 | 0.205 | 0.806 | 1.000 | 0.915 | 0.733 | 0.539 | 43 | 19.0 | 13.0 | 7.0 |
| -1.4116 | 23 | 6 | 93 | 0 | 10 | 0 | 0.189 | 0.793 | 1.000 | 0.921 | 0.714 | 0.544 | 40 | 17.0 | 11.0 | 5.0 |
| -0.9745 | 20 | 6 | 96 | 0 | 10 | 0 | 0.164 | 0.769 | 1.000 | 0.942 | 0.720 | 0.548 | 37 | 14.0 | 8.0 | 2.0 |
| -0.7906 | 19 | 6 | 97 | 0 | 10 | 0 | 0.156 | 0.760 | 1.000 | 0.942 | 0.720 | 0.548 | 35 | 13.0 | 7.0 | 1.0 |
| -0.4983 | 17 | 5 | 100 | 0 | 10 | 0 | 0.139 | 0.773 | 1.000 | 0.944 | 0.750 | 0.542 | 32 | 12.0 | 7.0 | 2.0 |
| 0.2460 | 15 | 4 | 103 | 0 | 10 | 0 | 0.123 | 0.789 | 1.000 | 0.953 | 0.763 | 0.545 | 29 | 11.0 | 7.0 | 3.0 |
| 0.4982 | 13 | 4 | 105 | 0 | 10 | 0 | 0.107 | 0.765 | 1.000 | 0.954 | 0.765 | 0.544 | 26 | 9.0 | 5.0 | 1.0 |
| 1.4306 | 12 | 4 | 106 | 0 | 10 | 0 | 0.098 | 0.750 | 1.000 | 0.951 | 0.781 | 0.553 | 23 | 8.0 | 4.0 | 0.0 |
| 1.8626 | 11 | 4 | 107 | 0 | 10 | 0 | 0.090 | 0.733 | 1.000 | 0.947 | 0.800 | 0.548 | 20 | 7.0 | 3.0 | -1.0 |
| 2.5715 | 10 | 2 | 110 | 0 | 10 | 0 | 0.082 | 0.833 | 1.000 | 0.952 | 0.792 | 0.543 | 17 | 8.0 | 6.0 | 4.0 |
| 2.6678 | 8 | 2 | 112 | 0 | 10 | 0 | 0.066 | 0.800 | 1.000 | 0.943 | 0.750 | 0.548 | 14 | 6.0 | 4.0 | 2.0 |
| 3.1148 | 7 | 0 | 115 | 0 | 10 | 0 | 0.057 | 1.000 | 1.000 | 0.982 | 0.786 | 0.575 | 11 | 7.0 | 7.0 | 7.0 |
| 3.8465 | 5 | 0 | 117 | 0 | 10 | 0 | 0.041 | 1.000 | 1.000 | 1.000 | 0.800 | 0.584 | 8 | 5.0 | 5.0 | 5.0 |
| 4.8267 | 2 | 0 | 120 | 0 | 10 | 0 | 0.016 | 1.000 | 1.000 | 1.000 | 0.750 | 0.567 | 5 | 2.0 | 2.0 | 2.0 |
| 6.4972 | 1 | 0 | 121 | 0 | 10 | 0 | 0.008 | 1.000 | 1.000 | 1.000 | 0.500 | 0.638 | 2 | 1.0 | 1.0 | 1.0 |
| +∞ | 0 | 0 | 122 | 0 | 10 | 0 | 0.000 | - | 1.000 | - | - | - | 0 | 0.0 | 0.0 | 0.0 |

(Se muestran 30 de 87 umbrales; todos están en el JSON.)
