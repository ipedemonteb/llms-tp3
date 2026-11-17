Reranker sobre s3-1024__e4__r1__cap150, test: 122 positivos, 10 negativos, 126 ítems; 50 candidatos puntuados; IC 95% bootstrap sobre ítems (10000 remuestreos)

| Variante | Pkg Hit@1 | Span@1000 | **Span@2000** | Span@4000 | nDCG@10 | Ef. contexto@2000 | AUROC coseno | AUROC reranker | s / consulta | Span rescata / rompe | Pkg rescata / rompe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base · sin reranker (S3 · 1024 · E4 · R1 · tope 150) | 0.467 [0.377, 0.558] | 0.295 [0.217, 0.381] | **0.377 [0.291, 0.468]** | 0.500 [0.410, 0.592] | 0.281 [0.218, 0.348] | 0.024 [0.017, 0.032] | 0.690 [0.485, 0.885] | - | - | - | - |
| k1-rr10 · Qwen3-Reranker-0.6B · reranker · top-10 | 0.475 [0.385, 0.566] | 0.344 [0.261, 0.433] | **0.393 [0.306, 0.487]** | 0.500 [0.409, 0.592] | 0.312 [0.243, 0.385] | 0.027 [0.019, 0.036] | 0.670 [0.457, 0.875] | 0.760 [0.547, 0.942] | 23.68 | +5 / −3 | +13 / −12 |
| k1-rr20 ⭐ · Qwen3-Reranker-0.6B · reranker · top-20 | 0.500 [0.410, 0.590] | 0.344 [0.262, 0.431] | **0.451 [0.362, 0.542]** | 0.533 [0.441, 0.625] | 0.336 [0.266, 0.407] | 0.030 [0.021, 0.040] | 0.654 [0.425, 0.870] | 0.750 [0.551, 0.918] | 47.37 | +14 / −5 | +17 / −13 |
| k1-rr30 · Qwen3-Reranker-0.6B · reranker · top-30 | 0.500 [0.410, 0.590] | 0.369 [0.283, 0.458] | **0.484 [0.395, 0.577]** | 0.533 [0.443, 0.625] | 0.348 [0.280, 0.420] | 0.032 [0.024, 0.042] | 0.714 [0.478, 0.922] | 0.760 [0.562, 0.928] | 71.05 | +18 / −5 | +17 / −13 |
| k1-rr50 · Qwen3-Reranker-0.6B · reranker · top-50 | 0.484 [0.393, 0.575] | 0.336 [0.252, 0.424] | **0.459 [0.370, 0.552]** | 0.549 [0.460, 0.641] | 0.334 [0.265, 0.406] | 0.032 [0.023, 0.043] | 0.765 [0.527, 0.958] | 0.734 [0.528, 0.921] | 118.42 | +17 / −7 | +15 / −13 |
| k1-rrf10 · Qwen3-Reranker-0.6B · fusión RRF · top-10 | 0.500 [0.408, 0.591] | 0.320 [0.240, 0.407] | **0.393 [0.306, 0.487]** | 0.508 [0.418, 0.600] | 0.304 [0.236, 0.375] | 0.023 [0.017, 0.031] | 0.725 [0.523, 0.907] | 0.760 [0.547, 0.942] | 23.68 | +4 / −2 | +10 / −6 |
| k1-rrf20 · Qwen3-Reranker-0.6B · fusión RRF · top-20 | 0.508 [0.418, 0.600] | 0.352 [0.268, 0.441] | **0.418 [0.331, 0.508]** | 0.516 [0.426, 0.610] | 0.332 [0.265, 0.403] | 0.025 [0.018, 0.032] | 0.693 [0.492, 0.882] | 0.750 [0.551, 0.918] | 47.37 | +9 / −4 | +12 / −7 |
| k1-rrf30 · Qwen3-Reranker-0.6B · fusión RRF · top-30 | 0.516 [0.426, 0.608] | 0.369 [0.282, 0.458] | **0.434 [0.347, 0.525]** | 0.549 [0.460, 0.642] | 0.339 [0.272, 0.409] | 0.025 [0.019, 0.032] | 0.711 [0.505, 0.897] | 0.760 [0.562, 0.928] | 71.05 | +11 / −4 | +13 / −7 |
| k1-rrf50 · Qwen3-Reranker-0.6B · fusión RRF · top-50 | 0.508 [0.419, 0.597] | 0.377 [0.291, 0.467] | **0.443 [0.355, 0.533]** | 0.566 [0.475, 0.658] | 0.337 [0.272, 0.407] | 0.026 [0.019, 0.033] | 0.702 [0.496, 0.891] | 0.734 [0.528, 0.921] | 118.42 | +12 / −4 | +14 / −9 |

Diferencia contra la base (pareada; empate = el IC de la diferencia incluye 0):

- **k1-rr10**: pkg_hit@1 +0.008 [-0.073, +0.090] empate; span_hit@2000 +0.016 [-0.026, +0.063] empate; ctx_eff@2000 +0.003 [-0.003, +0.011] empate; ndcg@10 +0.032 [-0.007, +0.072] empate
- **k1-rr20**: pkg_hit@1 +0.033 [-0.056, +0.121] empate; span_hit@2000 +0.074 [+0.008, +0.144] mejor; ctx_eff@2000 +0.006 [-0.002, +0.015] empate; ndcg@10 +0.055 [-0.003, +0.115] empate
- **k1-rr30**: pkg_hit@1 +0.033 [-0.056, +0.121] empate; span_hit@2000 +0.107 [+0.033, +0.183] mejor; ctx_eff@2000 +0.008 [+0.000, +0.018] mejor; ndcg@10 +0.068 [+0.004, +0.133] mejor
- **k1-rr50**: pkg_hit@1 +0.016 [-0.067, +0.101] empate; span_hit@2000 +0.082 [+0.008, +0.160] mejor; ctx_eff@2000 +0.009 [+0.000, +0.018] mejor; ndcg@10 +0.054 [-0.011, +0.121] empate
- **k1-rrf10**: pkg_hit@1 +0.033 [-0.032, +0.099] empate; span_hit@2000 +0.016 [-0.024, +0.057] empate; ctx_eff@2000 -0.000 [-0.005, +0.004] empate; ndcg@10 +0.024 [+0.004, +0.044] mejor
- **k1-rrf20**: pkg_hit@1 +0.041 [-0.026, +0.114] empate; span_hit@2000 +0.041 [-0.016, +0.099] empate; ctx_eff@2000 +0.001 [-0.004, +0.006] empate; ndcg@10 +0.052 [+0.022, +0.082] mejor
- **k1-rrf30**: pkg_hit@1 +0.049 [-0.024, +0.124] empate; span_hit@2000 +0.057 [+0.000, +0.120] empate; ctx_eff@2000 +0.002 [-0.004, +0.006] empate; ndcg@10 +0.058 [+0.024, +0.093] mejor
- **k1-rrf50**: pkg_hit@1 +0.041 [-0.034, +0.120] empate; span_hit@2000 +0.066 [+0.000, +0.130] empate; ctx_eff@2000 +0.002 [-0.004, +0.007] empate; ndcg@10 +0.057 [+0.017, +0.097] mejor

Fallos de la base en Span@2000 y cuántos rescata cada variante, por tipo (gold fuera del presupuesto: 17, otra librería en el top-1: 52, gold no recuperado: 7):

- **k1-rr10**: gold fuera del presupuesto 2/17, otra librería en el top-1 3/52, gold no recuperado 0/7
- **k1-rr20**: gold fuera del presupuesto 6/17, otra librería en el top-1 8/52, gold no recuperado 0/7
- **k1-rr30**: gold fuera del presupuesto 8/17, otra librería en el top-1 10/52, gold no recuperado 0/7
- **k1-rr50**: gold fuera del presupuesto 7/17, otra librería en el top-1 10/52, gold no recuperado 0/7
- **k1-rrf10**: gold fuera del presupuesto 1/17, otra librería en el top-1 3/52, gold no recuperado 0/7
- **k1-rrf20**: gold fuera del presupuesto 2/17, otra librería en el top-1 7/52, gold no recuperado 0/7
- **k1-rrf30**: gold fuera del presupuesto 3/17, otra librería en el top-1 8/52, gold no recuperado 0/7
- **k1-rrf50**: gold fuera del presupuesto 3/17, otra librería en el top-1 9/52, gold no recuperado 0/7

Umbral al 95% de los positivos (coseno del top-1 en la base, máximo del reranker en las variantes):

- **base**: 0.4225 -> pasan 116/122 positivos y 6/10 negativos
- **k1-rr10**: -7.5256 -> pasan 116/122 positivos y 6/10 negativos
- **k1-rr20**: -7.2467 -> pasan 116/122 positivos y 7/10 negativos
- **k1-rr30**: -6.9907 -> pasan 116/122 positivos y 7/10 negativos
- **k1-rr50**: -6.6678 -> pasan 116/122 positivos y 6/10 negativos
- **k1-rrf10**: -7.5256 -> pasan 116/122 positivos y 6/10 negativos
- **k1-rrf20**: -7.2467 -> pasan 116/122 positivos y 7/10 negativos
- **k1-rrf30**: -6.9907 -> pasan 116/122 positivos y 7/10 negativos
- **k1-rrf50**: -6.6678 -> pasan 116/122 positivos y 6/10 negativos

Comparaciones pareadas entre variantes (A − B; empate = el IC incluye 0):

- **k1-rr50 vs k1-rr30**: pkg_hit@1 -0.016 [-0.057, +0.024] empate; span_hit@2000 -0.025 [-0.061, +0.008] empate; span_hit@1000 -0.033 [-0.074, +0.000] empate; ndcg@10 -0.014 [-0.030, +0.003] empate
- **k1-rr30 vs k1-rr20**: pkg_hit@1 +0.000 [-0.025, +0.024] empate; span_hit@2000 +0.033 [+0.000, +0.074] empate; span_hit@1000 +0.025 [-0.008, +0.064] empate; ndcg@10 +0.012 [-0.013, +0.042] empate

Por subconjunto (Pkg Hit@1 / Span@2000):

| Variante | manual_deprecated (16) | synthetic (106) |
|---|---|---|
| base | 0.625 / 0.688 | 0.443 / 0.330 |
| k1-rr10 | 0.438 / 0.688 | 0.481 / 0.349 |
| k1-rr20 | 0.375 / 0.750 | 0.519 / 0.406 |
| k1-rr30 | 0.375 / 0.750 | 0.519 / 0.443 |
| k1-rr50 | 0.375 / 0.625 | 0.500 / 0.434 |
| k1-rrf10 | 0.562 / 0.688 | 0.491 / 0.349 |
| k1-rrf20 | 0.562 / 0.688 | 0.500 / 0.377 |
| k1-rrf30 | 0.562 / 0.688 | 0.509 / 0.396 |
| k1-rrf50 | 0.562 / 0.688 | 0.500 / 0.406 |

Contra results/study/rerank_analysis.json (s3-1024__e4__r1__cap150__sub-437d70f2; Pkg Hit@1 / Span@2000, otro → este):

- **base**: 0.766 → 0.467 / 0.561 → 0.377
- **k1-rr10**: 0.851 → 0.475 / 0.626 → 0.393
- **k1-rr20**: 0.897 → 0.500 / 0.692 → 0.451
- **k1-rr30**: 0.906 → 0.500 / 0.729 → 0.484
- **k1-rr50**: 0.897 → 0.484 / 0.748 → 0.459
- **k1-rrf10**: 0.860 → 0.500 / 0.626 → 0.393
- **k1-rrf20**: 0.851 → 0.508 / 0.673 → 0.418
- **k1-rrf30**: 0.860 → 0.516 / 0.682 → 0.434
- **k1-rrf50**: 0.851 → 0.508 / 0.710 → 0.443

Costo por reranker:

- **k1** Qwen/Qwen3-Reranker-0.6B: 2.368 s/par, 15615.2 s en total, GPU máx. 0 MB, truncados 0.0017

Decisión (regla fijada antes de correr): **k1-rr20** reemplaza a la base.
