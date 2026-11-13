Reranker sobre s3-1024__e4__r1__cap150, dev: 113 positivos, 9 negativos, 115 ítems; 50 candidatos puntuados; IC 95% bootstrap sobre ítems (10000 remuestreos)

| Variante | Pkg Hit@1 | Span@1000 | **Span@2000** | Span@4000 | nDCG@10 | Ef. contexto@2000 | AUROC coseno | AUROC reranker | s / consulta | Span rescata / rompe | Pkg rescata / rompe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base · sin reranker (S3 · 1024 · E4 · R1 · tope 150) | 0.593 [0.500, 0.684] | 0.345 [0.259, 0.438] | **0.460 [0.366, 0.559]** | 0.602 [0.509, 0.696] | 0.357 [0.284, 0.436] | 0.031 [0.023, 0.040] | 0.842 [0.625, 0.979] | - | - | - | - |
| k1-rr10 ⭐ · Qwen3-Reranker-0.6B · reranker · top-10 | 0.717 [0.634, 0.798] | 0.425 [0.333, 0.518] | **0.522 [0.425, 0.620]** | 0.611 [0.518, 0.704] | 0.423 [0.341, 0.509] | 0.037 [0.028, 0.048] | 0.804 [0.581, 0.959] | 0.932 [0.838, 0.997] | 5.37 | +9 / −2 | +18 / −4 |
| k1-rr20 · Qwen3-Reranker-0.6B · reranker · top-20 | 0.735 [0.654, 0.814] | 0.496 [0.400, 0.590] | **0.602 [0.509, 0.692]** | 0.664 [0.573, 0.752] | 0.513 [0.433, 0.593] | 0.043 [0.034, 0.054] | 0.838 [0.612, 0.986] | 0.947 [0.871, 0.999] | 10.74 | +19 / −3 | +21 / −5 |
| k1-rr30 · Qwen3-Reranker-0.6B · reranker · top-30 | 0.743 [0.664, 0.821] | 0.504 [0.411, 0.597] | **0.637 [0.545, 0.725]** | 0.699 [0.612, 0.783] | 0.525 [0.447, 0.604] | 0.045 [0.036, 0.056] | 0.919 [0.817, 0.994] | 0.946 [0.866, 1.000] | 16.12 | +23 / −3 | +22 / −5 |
| k1-rr50 · Qwen3-Reranker-0.6B · reranker · top-50 | 0.770 [0.693, 0.844] | 0.513 [0.417, 0.610] | **0.655 [0.562, 0.741]** | 0.717 [0.628, 0.800] | 0.534 [0.456, 0.613] | 0.046 [0.037, 0.057] | 0.921 [0.840, 0.985] | 0.951 [0.870, 1.000] | 26.86 | +25 / −3 | +24 / −4 |
| k1-rrf10 · Qwen3-Reranker-0.6B · fusión RRF · top-10 | 0.673 [0.584, 0.759] | 0.416 [0.324, 0.513] | **0.522 [0.425, 0.620]** | 0.611 [0.518, 0.704] | 0.399 [0.320, 0.482] | 0.036 [0.027, 0.046] | 0.839 [0.620, 0.978] | 0.932 [0.838, 0.997] | 5.37 | +7 / −0 | +11 / −2 |
| k1-rrf20 · Qwen3-Reranker-0.6B · fusión RRF · top-20 | 0.673 [0.586, 0.756] | 0.460 [0.366, 0.556] | **0.584 [0.487, 0.679]** | 0.655 [0.565, 0.743] | 0.445 [0.371, 0.521] | 0.041 [0.032, 0.052] | 0.830 [0.610, 0.972] | 0.947 [0.871, 0.999] | 10.74 | +15 / −1 | +12 / −3 |
| k1-rrf30 · Qwen3-Reranker-0.6B · fusión RRF · top-30 | 0.673 [0.586, 0.756] | 0.460 [0.366, 0.556] | **0.593 [0.500, 0.688]** | 0.681 [0.591, 0.768] | 0.457 [0.383, 0.532] | 0.042 [0.032, 0.053] | 0.830 [0.610, 0.972] | 0.946 [0.866, 1.000] | 16.12 | +16 / −1 | +12 / −3 |
| k1-rrf50 · Qwen3-Reranker-0.6B · fusión RRF · top-50 | 0.664 [0.576, 0.748] | 0.460 [0.366, 0.556] | **0.593 [0.496, 0.686]** | 0.699 [0.609, 0.786] | 0.461 [0.388, 0.536] | 0.042 [0.033, 0.053] | 0.826 [0.607, 0.971] | 0.951 [0.870, 1.000] | 26.86 | +16 / −1 | +11 / −3 |

Diferencia contra la base (pareada; empate = el IC de la diferencia incluye 0):

- **k1-rr10**: pkg_hit@1 +0.124 [+0.045, +0.209] mejor; span_hit@2000 +0.062 [+0.009, +0.120] mejor; ctx_eff@2000 +0.006 [+0.002, +0.011] mejor; ndcg@10 +0.065 [+0.022, +0.111] mejor
- **k1-rr20**: pkg_hit@1 +0.142 [+0.056, +0.230] mejor; span_hit@2000 +0.142 [+0.061, +0.223] mejor; ctx_eff@2000 +0.012 [+0.006, +0.019] mejor; ndcg@10 +0.156 [+0.089, +0.226] mejor
- **k1-rr30**: pkg_hit@1 +0.150 [+0.063, +0.239] mejor; span_hit@2000 +0.177 [+0.089, +0.265] mejor; ctx_eff@2000 +0.015 [+0.007, +0.022] mejor; ndcg@10 +0.167 [+0.093, +0.241] mejor
- **k1-rr50**: pkg_hit@1 +0.177 [+0.090, +0.267] mejor; span_hit@2000 +0.195 [+0.104, +0.283] mejor; ctx_eff@2000 +0.015 [+0.009, +0.023] mejor; ndcg@10 +0.177 [+0.102, +0.252] mejor
- **k1-rrf10**: pkg_hit@1 +0.080 [+0.018, +0.140] mejor; span_hit@2000 +0.062 [+0.020, +0.109] mejor; ctx_eff@2000 +0.005 [+0.002, +0.009] mejor; ndcg@10 +0.042 [+0.022, +0.064] mejor
- **k1-rrf20**: pkg_hit@1 +0.080 [+0.017, +0.144] mejor; span_hit@2000 +0.124 [+0.054, +0.197] mejor; ctx_eff@2000 +0.011 [+0.004, +0.017] mejor; ndcg@10 +0.087 [+0.057, +0.119] mejor
- **k1-rrf30**: pkg_hit@1 +0.080 [+0.017, +0.144] mejor; span_hit@2000 +0.133 [+0.062, +0.207] mejor; ctx_eff@2000 +0.011 [+0.005, +0.018] mejor; ndcg@10 +0.100 [+0.067, +0.133] mejor
- **k1-rrf50**: pkg_hit@1 +0.071 [+0.009, +0.135] mejor; span_hit@2000 +0.133 [+0.061, +0.208] mejor; ctx_eff@2000 +0.012 [+0.005, +0.018] mejor; ndcg@10 +0.103 [+0.068, +0.139] mejor

Fallos de la base en Span@2000 y cuántos rescata cada variante, por tipo (gold fuera del presupuesto: 19, otra librería en el top-1: 34, gold no recuperado: 8):

- **k1-rr10**: gold fuera del presupuesto 6/19, otra librería en el top-1 3/34, gold no recuperado 0/8
- **k1-rr20**: gold fuera del presupuesto 11/19, otra librería en el top-1 8/34, gold no recuperado 0/8
- **k1-rr30**: gold fuera del presupuesto 14/19, otra librería en el top-1 9/34, gold no recuperado 0/8
- **k1-rr50**: gold fuera del presupuesto 16/19, otra librería en el top-1 9/34, gold no recuperado 0/8
- **k1-rrf10**: gold fuera del presupuesto 4/19, otra librería en el top-1 3/34, gold no recuperado 0/8
- **k1-rrf20**: gold fuera del presupuesto 10/19, otra librería en el top-1 5/34, gold no recuperado 0/8
- **k1-rrf30**: gold fuera del presupuesto 10/19, otra librería en el top-1 6/34, gold no recuperado 0/8
- **k1-rrf50**: gold fuera del presupuesto 9/19, otra librería en el top-1 7/34, gold no recuperado 0/8

Umbral al 95% de los positivos (coseno del top-1 en la base, máximo del reranker en las variantes):

- **base**: 0.4547 -> pasan 108/113 positivos y 4/9 negativos
- **k1-rr10**: -5.4297 -> pasan 108/113 positivos y 2/9 negativos
- **k1-rr20**: -5.2266 -> pasan 108/113 positivos y 2/9 negativos
- **k1-rr30**: -4.5156 -> pasan 108/113 positivos y 2/9 negativos
- **k1-rr50**: -4.1875 -> pasan 108/113 positivos y 2/9 negativos
- **k1-rrf10**: -5.4297 -> pasan 108/113 positivos y 2/9 negativos
- **k1-rrf20**: -5.2266 -> pasan 108/113 positivos y 2/9 negativos
- **k1-rrf30**: -4.5156 -> pasan 108/113 positivos y 2/9 negativos
- **k1-rrf50**: -4.1875 -> pasan 108/113 positivos y 2/9 negativos

Costo por reranker:

- **k1** Qwen/Qwen3-Reranker-0.6B: 0.537 s/par, 1615.3 s en total, GPU máx. 2289 MB, truncados 0.0016

Decisión (regla fijada antes de correr): **k1-rr10** reemplaza a la base.
