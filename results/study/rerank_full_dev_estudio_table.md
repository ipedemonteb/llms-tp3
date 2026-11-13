Reranker sobre s3-1024__e4__r1__cap150, dev: 113 positivos, 9 negativos, 115 ítems; 50 candidatos puntuados; IC 95% bootstrap sobre ítems (10000 remuestreos)

| Variante | Pkg Hit@1 | Span@1000 | **Span@2000** | Span@4000 | nDCG@10 | Ef. contexto@2000 | AUROC coseno | AUROC reranker | s / consulta | Span rescata / rompe | Pkg rescata / rompe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base · sin reranker (S3 · 1024 · E4 · R1 · tope 150) | 0.593 [0.500, 0.684] | 0.345 [0.259, 0.438] | **0.460 [0.366, 0.559]** | 0.602 [0.509, 0.696] | 0.357 [0.284, 0.436] | 0.031 [0.023, 0.040] | 0.842 [0.625, 0.979] | - | - | - | - |
| k1-rr10 · Qwen3-Reranker-0.6B · reranker · top-10 | 0.717 [0.634, 0.798] | 0.425 [0.333, 0.518] | **0.522 [0.425, 0.620]** | 0.611 [0.518, 0.704] | 0.423 [0.341, 0.509] | 0.037 [0.028, 0.048] | 0.804 [0.581, 0.959] | 0.932 [0.838, 0.997] | 5.37 | +9 / −2 | +18 / −4 |
| k1-rr20 · Qwen3-Reranker-0.6B · reranker · top-20 | 0.735 [0.654, 0.814] | 0.496 [0.400, 0.590] | **0.602 [0.509, 0.692]** | 0.664 [0.573, 0.752] | 0.513 [0.433, 0.593] | 0.043 [0.034, 0.054] | 0.838 [0.612, 0.986] | 0.947 [0.871, 0.999] | 10.74 | +19 / −3 | +21 / −5 |
| k1-rr30 · Qwen3-Reranker-0.6B · reranker · top-30 | 0.743 [0.664, 0.821] | 0.504 [0.411, 0.597] | **0.637 [0.545, 0.725]** | 0.699 [0.612, 0.783] | 0.525 [0.447, 0.604] | 0.045 [0.036, 0.056] | 0.919 [0.817, 0.994] | 0.946 [0.866, 1.000] | 16.12 | +23 / −3 | +22 / −5 |
| k1-rr50 · Qwen3-Reranker-0.6B · reranker · top-50 | 0.770 [0.693, 0.844] | 0.513 [0.417, 0.610] | **0.655 [0.562, 0.741]** | 0.717 [0.628, 0.800] | 0.534 [0.456, 0.613] | 0.046 [0.037, 0.057] | 0.921 [0.840, 0.985] | 0.951 [0.870, 1.000] | 26.86 | +25 / −3 | +24 / −4 |
| k1-rrf10 · Qwen3-Reranker-0.6B · fusión RRF · top-10 | 0.673 [0.584, 0.759] | 0.416 [0.324, 0.513] | **0.522 [0.425, 0.620]** | 0.611 [0.518, 0.704] | 0.399 [0.320, 0.482] | 0.036 [0.027, 0.046] | 0.839 [0.620, 0.978] | 0.932 [0.838, 0.997] | 5.37 | +7 / −0 | +11 / −2 |
| k1-rrf20 · Qwen3-Reranker-0.6B · fusión RRF · top-20 | 0.673 [0.586, 0.756] | 0.460 [0.366, 0.556] | **0.584 [0.487, 0.679]** | 0.655 [0.565, 0.743] | 0.445 [0.371, 0.521] | 0.041 [0.032, 0.052] | 0.830 [0.610, 0.972] | 0.947 [0.871, 0.999] | 10.74 | +15 / −1 | +12 / −3 |
| k1-rrf30 · Qwen3-Reranker-0.6B · fusión RRF · top-30 | 0.673 [0.586, 0.756] | 0.460 [0.366, 0.556] | **0.593 [0.500, 0.688]** | 0.681 [0.591, 0.768] | 0.457 [0.383, 0.532] | 0.042 [0.032, 0.053] | 0.830 [0.610, 0.972] | 0.946 [0.866, 1.000] | 16.12 | +16 / −1 | +12 / −3 |
| k1-rrf50 · Qwen3-Reranker-0.6B · fusión RRF · top-50 | 0.664 [0.576, 0.748] | 0.460 [0.366, 0.556] | **0.593 [0.496, 0.686]** | 0.699 [0.609, 0.786] | 0.461 [0.388, 0.536] | 0.042 [0.033, 0.053] | 0.826 [0.607, 0.971] | 0.951 [0.870, 1.000] | 26.86 | +16 / −1 | +11 / −3 |
| k2-rr10 · gte-reranker-modernbert-base · reranker · top-10 | 0.681 [0.596, 0.767] | 0.381 [0.288, 0.478] | **0.496 [0.400, 0.595]** | 0.602 [0.509, 0.696] | 0.404 [0.322, 0.491] | 0.032 [0.024, 0.041] | 0.872 [0.688, 0.988] | 0.696 [0.570, 0.811] | 1.23 | +8 / −4 | +19 / −9 |
| k2-rr20 · gte-reranker-modernbert-base · reranker · top-20 | 0.708 [0.624, 0.790] | 0.389 [0.295, 0.486] | **0.478 [0.383, 0.573]** | 0.620 [0.527, 0.713] | 0.447 [0.366, 0.530] | 0.035 [0.026, 0.045] | 0.872 [0.686, 0.983] | 0.696 [0.552, 0.827] | 2.45 | +15 / −13 | +22 / −9 |
| k2-rr30 · gte-reranker-modernbert-base · reranker · top-30 | 0.673 [0.585, 0.759] | 0.345 [0.257, 0.438] | **0.469 [0.375, 0.563]** | 0.575 [0.483, 0.667] | 0.419 [0.342, 0.498] | 0.037 [0.028, 0.049] | 0.898 [0.767, 0.992] | 0.679 [0.531, 0.821] | 3.68 | +17 / −16 | +21 / −12 |
| k2-rr50 · gte-reranker-modernbert-base · reranker · top-50 | 0.673 [0.583, 0.761] | 0.336 [0.248, 0.431] | **0.443 [0.348, 0.536]** | 0.513 [0.420, 0.607] | 0.407 [0.328, 0.488] | 0.042 [0.026, 0.065] | 0.900 [0.766, 0.993] | 0.548 [0.367, 0.709] | 6.13 | +16 / −18 | +21 / −12 |
| k2-rrf10 · gte-reranker-modernbert-base · fusión RRF · top-10 | 0.620 [0.531, 0.709] | 0.398 [0.307, 0.491] | **0.496 [0.400, 0.593]** | 0.611 [0.518, 0.704] | 0.400 [0.321, 0.484] | 0.034 [0.025, 0.044] | 0.841 [0.616, 0.980] | 0.696 [0.570, 0.811] | 1.23 | +6 / −2 | +9 / −6 |
| k2-rrf20 ⭐ · gte-reranker-modernbert-base · fusión RRF · top-20 | 0.655 [0.564, 0.743] | 0.407 [0.317, 0.504] | **0.557 [0.462, 0.654]** | 0.628 [0.536, 0.721] | 0.435 [0.361, 0.514] | 0.039 [0.030, 0.050] | 0.828 [0.602, 0.978] | 0.696 [0.552, 0.827] | 2.45 | +14 / −3 | +16 / −9 |
| k2-rrf30 · gte-reranker-modernbert-base · fusión RRF · top-30 | 0.655 [0.566, 0.741] | 0.398 [0.307, 0.491] | **0.549 [0.453, 0.645]** | 0.637 [0.544, 0.729] | 0.435 [0.362, 0.511] | 0.039 [0.030, 0.051] | 0.853 [0.688, 0.977] | 0.679 [0.531, 0.821] | 3.68 | +14 / −4 | +16 / −9 |
| k2-rrf50 · gte-reranker-modernbert-base · fusión RRF · top-50 | 0.655 [0.565, 0.741] | 0.398 [0.307, 0.491] | **0.531 [0.435, 0.627]** | 0.628 [0.537, 0.719] | 0.439 [0.366, 0.515] | 0.040 [0.030, 0.052] | 0.845 [0.680, 0.972] | 0.548 [0.367, 0.709] | 6.13 | +16 / −8 | +17 / −10 |

Diferencia contra la base (pareada; empate = el IC de la diferencia incluye 0):

- **k1-rr10**: pkg_hit@1 +0.124 [+0.045, +0.209] mejor; span_hit@2000 +0.062 [+0.009, +0.120] mejor; ctx_eff@2000 +0.006 [+0.002, +0.011] mejor; ndcg@10 +0.065 [+0.022, +0.111] mejor
- **k1-rr20**: pkg_hit@1 +0.142 [+0.056, +0.230] mejor; span_hit@2000 +0.142 [+0.061, +0.223] mejor; ctx_eff@2000 +0.012 [+0.006, +0.019] mejor; ndcg@10 +0.156 [+0.089, +0.226] mejor
- **k1-rr30**: pkg_hit@1 +0.150 [+0.063, +0.239] mejor; span_hit@2000 +0.177 [+0.089, +0.265] mejor; ctx_eff@2000 +0.015 [+0.007, +0.022] mejor; ndcg@10 +0.167 [+0.093, +0.241] mejor
- **k1-rr50**: pkg_hit@1 +0.177 [+0.090, +0.267] mejor; span_hit@2000 +0.195 [+0.104, +0.283] mejor; ctx_eff@2000 +0.015 [+0.009, +0.023] mejor; ndcg@10 +0.177 [+0.102, +0.252] mejor
- **k1-rrf10**: pkg_hit@1 +0.080 [+0.018, +0.140] mejor; span_hit@2000 +0.062 [+0.020, +0.109] mejor; ctx_eff@2000 +0.005 [+0.002, +0.009] mejor; ndcg@10 +0.042 [+0.022, +0.064] mejor
- **k1-rrf20**: pkg_hit@1 +0.080 [+0.017, +0.144] mejor; span_hit@2000 +0.124 [+0.054, +0.197] mejor; ctx_eff@2000 +0.011 [+0.004, +0.017] mejor; ndcg@10 +0.087 [+0.057, +0.119] mejor
- **k1-rrf30**: pkg_hit@1 +0.080 [+0.017, +0.144] mejor; span_hit@2000 +0.133 [+0.062, +0.207] mejor; ctx_eff@2000 +0.011 [+0.005, +0.018] mejor; ndcg@10 +0.100 [+0.067, +0.133] mejor
- **k1-rrf50**: pkg_hit@1 +0.071 [+0.009, +0.135] mejor; span_hit@2000 +0.133 [+0.061, +0.208] mejor; ctx_eff@2000 +0.012 [+0.005, +0.018] mejor; ndcg@10 +0.103 [+0.068, +0.139] mejor
- **k2-rr10**: pkg_hit@1 +0.088 [+0.000, +0.183] empate; span_hit@2000 +0.035 [-0.026, +0.097] empate; ctx_eff@2000 +0.001 [-0.005, +0.007] empate; ndcg@10 +0.047 [+0.004, +0.092] mejor
- **k2-rr20**: pkg_hit@1 +0.115 [+0.018, +0.210] mejor; span_hit@2000 +0.018 [-0.081, +0.111] empate; ctx_eff@2000 +0.004 [-0.005, +0.013] empate; ndcg@10 +0.090 [+0.014, +0.166] mejor
- **k2-rr30**: pkg_hit@1 +0.080 [-0.019, +0.179] empate; span_hit@2000 +0.009 [-0.097, +0.111] empate; ctx_eff@2000 +0.007 [-0.003, +0.016] empate; ndcg@10 +0.062 [-0.017, +0.139] empate
- **k2-rr50**: pkg_hit@1 +0.080 [-0.026, +0.180] empate; span_hit@2000 -0.018 [-0.127, +0.091] empate; ctx_eff@2000 +0.011 [-0.003, +0.029] empate; ndcg@10 +0.050 [-0.036, +0.135] empate
- **k2-rrf10**: pkg_hit@1 +0.026 [-0.037, +0.095] empate; span_hit@2000 +0.035 [-0.009, +0.087] empate; ctx_eff@2000 +0.003 [-0.000, +0.007] empate; ndcg@10 +0.043 [+0.018, +0.069] mejor
- **k2-rrf20**: pkg_hit@1 +0.062 [-0.019, +0.148] empate; span_hit@2000 +0.097 [+0.026, +0.171] mejor; ctx_eff@2000 +0.008 [+0.004, +0.013] mejor; ndcg@10 +0.078 [+0.043, +0.113] mejor
- **k2-rrf30**: pkg_hit@1 +0.062 [-0.019, +0.149] empate; span_hit@2000 +0.088 [+0.017, +0.164] mejor; ctx_eff@2000 +0.009 [+0.004, +0.014] mejor; ndcg@10 +0.078 [+0.037, +0.117] mejor
- **k2-rrf50**: pkg_hit@1 +0.062 [-0.026, +0.152] empate; span_hit@2000 +0.071 [-0.018, +0.158] empate; ctx_eff@2000 +0.009 [+0.002, +0.017] mejor; ndcg@10 +0.082 [+0.035, +0.126] mejor

Fallos de la base en Span@2000 y cuántos rescata cada variante, por tipo (gold fuera del presupuesto: 19, otra librería en el top-1: 34, gold no recuperado: 8):

- **k1-rr10**: gold fuera del presupuesto 6/19, otra librería en el top-1 3/34, gold no recuperado 0/8
- **k1-rr20**: gold fuera del presupuesto 11/19, otra librería en el top-1 8/34, gold no recuperado 0/8
- **k1-rr30**: gold fuera del presupuesto 14/19, otra librería en el top-1 9/34, gold no recuperado 0/8
- **k1-rr50**: gold fuera del presupuesto 16/19, otra librería en el top-1 9/34, gold no recuperado 0/8
- **k1-rrf10**: gold fuera del presupuesto 4/19, otra librería en el top-1 3/34, gold no recuperado 0/8
- **k1-rrf20**: gold fuera del presupuesto 10/19, otra librería en el top-1 5/34, gold no recuperado 0/8
- **k1-rrf30**: gold fuera del presupuesto 10/19, otra librería en el top-1 6/34, gold no recuperado 0/8
- **k1-rrf50**: gold fuera del presupuesto 9/19, otra librería en el top-1 7/34, gold no recuperado 0/8
- **k2-rr10**: gold fuera del presupuesto 5/19, otra librería en el top-1 3/34, gold no recuperado 0/8
- **k2-rr20**: gold fuera del presupuesto 8/19, otra librería en el top-1 7/34, gold no recuperado 0/8
- **k2-rr30**: gold fuera del presupuesto 10/19, otra librería en el top-1 7/34, gold no recuperado 0/8
- **k2-rr50**: gold fuera del presupuesto 10/19, otra librería en el top-1 6/34, gold no recuperado 0/8
- **k2-rrf10**: gold fuera del presupuesto 3/19, otra librería en el top-1 3/34, gold no recuperado 0/8
- **k2-rrf20**: gold fuera del presupuesto 8/19, otra librería en el top-1 6/34, gold no recuperado 0/8
- **k2-rrf30**: gold fuera del presupuesto 8/19, otra librería en el top-1 6/34, gold no recuperado 0/8
- **k2-rrf50**: gold fuera del presupuesto 9/19, otra librería en el top-1 7/34, gold no recuperado 0/8

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
- **k2-rr10**: 0.6694 -> pasan 108/113 positivos y 9/9 negativos
- **k2-rr20**: 0.7134 -> pasan 108/113 positivos y 9/9 negativos
- **k2-rr30**: 0.7158 -> pasan 108/113 positivos y 9/9 negativos
- **k2-rr50**: 0.7295 -> pasan 108/113 positivos y 9/9 negativos
- **k2-rrf10**: 0.6694 -> pasan 108/113 positivos y 9/9 negativos
- **k2-rrf20**: 0.7134 -> pasan 108/113 positivos y 9/9 negativos
- **k2-rrf30**: 0.7158 -> pasan 108/113 positivos y 9/9 negativos
- **k2-rrf50**: 0.7295 -> pasan 108/113 positivos y 9/9 negativos

Comparaciones pareadas entre variantes (A − B; empate = el IC incluye 0):

- **k1-rr30 vs k2-rrf10**: pkg_hit@1 +0.124 [+0.037, +0.209] mejor; span_hit@2000 +0.142 [+0.062, +0.223] mejor; span_hit@1000 +0.106 [+0.009, +0.200] mejor; ndcg@10 +0.124 [+0.051, +0.199] mejor
- **k1-rr20 vs k2-rrf10**: pkg_hit@1 +0.115 [+0.026, +0.202] mejor; span_hit@2000 +0.106 [+0.035, +0.180] mejor; span_hit@1000 +0.097 [+0.009, +0.189] mejor; ndcg@10 +0.113 [+0.047, +0.182] mejor
- **k1-rr50 vs k1-rr30**: pkg_hit@1 +0.026 [+0.000, +0.061] empate; span_hit@2000 +0.018 [-0.018, +0.053] empate; span_hit@1000 +0.009 [-0.018, +0.038] empate; ndcg@10 +0.010 [-0.011, +0.035] empate
- **k1-rr30 vs k1-rr20**: pkg_hit@1 +0.009 [+0.000, +0.028] empate; span_hit@2000 +0.035 [+0.009, +0.072] mejor; span_hit@1000 +0.009 [-0.028, +0.048] empate; ndcg@10 +0.011 [-0.017, +0.045] empate

Por subconjunto (Pkg Hit@1 / Span@2000):

| Variante | manual_deprecated (15) | synthetic (98) |
|---|---|---|
| base | 0.867 / 0.733 | 0.551 / 0.418 |
| k1-rr10 | 0.867 / 0.667 | 0.694 / 0.500 |
| k1-rr20 | 0.867 / 0.600 | 0.714 / 0.602 |
| k1-rr30 | 0.867 / 0.600 | 0.725 / 0.643 |
| k1-rr50 | 0.867 / 0.600 | 0.755 / 0.663 |
| k1-rrf10 | 0.867 / 0.733 | 0.643 / 0.490 |
| k1-rrf20 | 0.867 / 0.667 | 0.643 / 0.571 |
| k1-rrf30 | 0.867 / 0.667 | 0.643 / 0.582 |
| k1-rrf50 | 0.867 / 0.667 | 0.633 / 0.582 |
| k2-rr10 | 0.800 / 0.667 | 0.663 / 0.469 |
| k2-rr20 | 0.867 / 0.467 | 0.684 / 0.480 |
| k2-rr30 | 0.867 / 0.467 | 0.643 / 0.469 |
| k2-rr50 | 0.867 / 0.467 | 0.643 / 0.439 |
| k2-rrf10 | 0.867 / 0.600 | 0.582 / 0.480 |
| k2-rrf20 | 0.867 / 0.600 | 0.622 / 0.551 |
| k2-rrf30 | 0.867 / 0.600 | 0.622 / 0.541 |
| k2-rrf50 | 0.867 / 0.533 | 0.622 / 0.531 |

Contra results/study/rerank_analysis.json (s3-1024__e4__r1__cap150__sub-437d70f2; Pkg Hit@1 / Span@2000, otro → este):

- **base**: 0.766 → 0.593 / 0.561 → 0.460
- **k1-rr10**: 0.851 → 0.717 / 0.626 → 0.522
- **k1-rr20**: 0.897 → 0.735 / 0.692 → 0.602
- **k1-rr30**: 0.906 → 0.743 / 0.729 → 0.637
- **k1-rr50**: 0.897 → 0.770 / 0.748 → 0.655
- **k1-rrf10**: 0.860 → 0.673 / 0.626 → 0.522
- **k1-rrf20**: 0.851 → 0.673 / 0.673 → 0.584
- **k1-rrf30**: 0.860 → 0.673 / 0.682 → 0.593
- **k1-rrf50**: 0.851 → 0.664 / 0.710 → 0.593
- **k2-rr10**: 0.813 → 0.681 / 0.598 → 0.496
- **k2-rr20**: 0.804 → 0.708 / 0.533 → 0.478
- **k2-rr30**: 0.785 → 0.673 / 0.542 → 0.469
- **k2-rr50**: 0.748 → 0.673 / 0.467 → 0.443
- **k2-rrf10**: 0.822 → 0.620 / 0.635 → 0.496
- **k2-rrf20**: 0.794 → 0.655 / 0.635 → 0.557
- **k2-rrf30**: 0.813 → 0.655 / 0.626 → 0.549
- **k2-rrf50**: 0.813 → 0.655 / 0.608 → 0.531

Costo por reranker:

- **k1** Qwen/Qwen3-Reranker-0.6B: 0.537 s/par, 1615.3 s en total, GPU máx. 2289 MB, truncados 0.0016
- **k2** Alibaba-NLP/gte-reranker-modernbert-base: 0.123 s/par, 368.5 s en total, GPU máx. 1050 MB, truncados 0.0018

Decisión (regla fijada antes de correr): **k2-rrf20** reemplaza a la base.
