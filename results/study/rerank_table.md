Reranker sobre s3-1024__e4__r1__cap150__sub-437d70f2, dev: 107 positivos, 9 negativos, 109 ítems; 50 candidatos puntuados; IC 95% bootstrap sobre ítems (10000 remuestreos)

| Variante | Pkg Hit@1 | Span@1000 | **Span@2000** | Span@4000 | nDCG@10 | Ef. contexto@2000 | AUROC coseno | AUROC reranker | s / consulta | Span rescata / rompe | Pkg rescata / rompe |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base · sin reranker (S3 · 1024 · E4 · R1 · tope 150) | 0.766 [0.676, 0.848] | 0.449 [0.350, 0.545] | **0.561 [0.462, 0.658]** | 0.682 [0.589, 0.771] | 0.476 [0.396, 0.555] | 0.040 [0.030, 0.051] | 0.822 [0.602, 0.967] | - | - | - | - |
| k1-rr10 · Qwen3-Reranker-0.6B · reranker · top-10 | 0.851 [0.779, 0.915] | 0.523 [0.426, 0.620] | **0.626 [0.529, 0.718]** | 0.682 [0.589, 0.771] | 0.538 [0.453, 0.622] | 0.045 [0.035, 0.057] | 0.802 [0.579, 0.955] | 0.952 [0.877, 0.996] | 5.18 | +10 / −3 | +11 / −2 |
| k1-rr20 · Qwen3-Reranker-0.6B · reranker · top-20 | 0.897 [0.835, 0.952] | 0.561 [0.465, 0.654] | **0.692 [0.602, 0.777]** | 0.729 [0.644, 0.809] | 0.590 [0.511, 0.667] | 0.051 [0.041, 0.062] | 0.809 [0.585, 0.961] | 0.962 [0.900, 0.998] | 10.36 | +17 / −3 | +16 / −2 |
| k1-rr30 · Qwen3-Reranker-0.6B · reranker · top-30 | 0.906 [0.848, 0.956] | 0.579 [0.481, 0.676] | **0.729 [0.641, 0.809]** | 0.766 [0.682, 0.841] | 0.612 [0.534, 0.685] | 0.053 [0.043, 0.064] | 0.865 [0.756, 0.954] | 0.962 [0.895, 0.999] | 15.53 | +22 / −4 | +17 / −2 |
| k1-rr50 · Qwen3-Reranker-0.6B · reranker · top-50 | 0.897 [0.836, 0.952] | 0.589 [0.486, 0.685] | **0.748 [0.660, 0.827]** | 0.804 [0.725, 0.875] | 0.622 [0.544, 0.694] | 0.053 [0.044, 0.064] | 0.872 [0.763, 0.960] | 0.965 [0.898, 1.000] | 25.89 | +24 / −4 | +18 / −4 |
| k1-rrf10 · Qwen3-Reranker-0.6B · fusión RRF · top-10 | 0.860 [0.788, 0.923] | 0.533 [0.434, 0.630] | **0.626 [0.529, 0.719]** | 0.682 [0.589, 0.771] | 0.520 [0.436, 0.603] | 0.045 [0.035, 0.057] | 0.805 [0.588, 0.954] | 0.952 [0.877, 0.996] | 5.18 | +7 / −0 | +12 / −2 |
| k1-rrf20 · Qwen3-Reranker-0.6B · fusión RRF · top-20 | 0.851 [0.778, 0.916] | 0.579 [0.481, 0.673] | **0.673 [0.579, 0.761]** | 0.729 [0.639, 0.813] | 0.559 [0.480, 0.637] | 0.050 [0.039, 0.062] | 0.795 [0.577, 0.950] | 0.962 [0.900, 0.998] | 10.36 | +13 / −1 | +11 / −2 |
| k1-rrf30 · Qwen3-Reranker-0.6B · fusión RRF · top-30 | 0.860 [0.788, 0.924] | 0.589 [0.490, 0.684] | **0.682 [0.587, 0.771]** | 0.766 [0.680, 0.846] | 0.577 [0.500, 0.651] | 0.051 [0.040, 0.062] | 0.792 [0.574, 0.949] | 0.962 [0.895, 0.999] | 15.53 | +14 / −1 | +12 / −2 |
| k1-rrf50 · Qwen3-Reranker-0.6B · fusión RRF · top-50 | 0.851 [0.778, 0.916] | 0.598 [0.500, 0.692] | **0.710 [0.618, 0.796]** | 0.804 [0.726, 0.875] | 0.584 [0.509, 0.657] | 0.051 [0.041, 0.062] | 0.801 [0.583, 0.950] | 0.965 [0.898, 1.000] | 25.89 | +17 / −1 | +11 / −2 |
| k2-rr10 · gte-reranker-modernbert-base · reranker · top-10 | 0.813 [0.735, 0.884] | 0.467 [0.367, 0.565] | **0.598 [0.500, 0.692]** | 0.664 [0.570, 0.752] | 0.508 [0.424, 0.591] | 0.040 [0.031, 0.049] | 0.835 [0.651, 0.970] | 0.721 [0.596, 0.829] | 1.04 | +10 / −6 | +11 / −6 |
| k2-rr20 · gte-reranker-modernbert-base · reranker · top-20 | 0.804 [0.726, 0.875] | 0.402 [0.306, 0.495] | **0.533 [0.435, 0.627]** | 0.664 [0.573, 0.748] | 0.490 [0.410, 0.568] | 0.039 [0.030, 0.048] | 0.893 [0.757, 0.989] | 0.705 [0.554, 0.837] | 2.08 | +14 / −17 | +12 / −8 |
| k2-rr30 · gte-reranker-modernbert-base · reranker · top-30 | 0.785 [0.705, 0.861] | 0.402 [0.305, 0.500] | **0.542 [0.441, 0.641]** | 0.617 [0.519, 0.712] | 0.493 [0.412, 0.574] | 0.050 [0.033, 0.073] | 0.885 [0.755, 0.982] | 0.709 [0.568, 0.836] | 3.12 | +17 / −19 | +11 / −9 |
| k2-rr50 · gte-reranker-modernbert-base · reranker · top-50 | 0.748 [0.660, 0.827] | 0.355 [0.261, 0.449] | **0.467 [0.370, 0.565]** | 0.533 [0.435, 0.627] | 0.447 [0.367, 0.527] | 0.046 [0.029, 0.070] | 0.835 [0.697, 0.950] | 0.678 [0.555, 0.794] | 5.20 | +14 / −24 | +11 / −13 |
| k2-rrf10 ⭐ · gte-reranker-modernbert-base · fusión RRF · top-10 | 0.822 [0.745, 0.892] | 0.449 [0.349, 0.545] | **0.635 [0.538, 0.728]** | 0.682 [0.589, 0.771] | 0.508 [0.425, 0.588] | 0.047 [0.036, 0.059] | 0.818 [0.590, 0.978] | 0.721 [0.596, 0.829] | 1.04 | +10 / −2 | +9 / −3 |
| k2-rrf20 · gte-reranker-modernbert-base · fusión RRF · top-20 | 0.794 [0.714, 0.869] | 0.486 [0.385, 0.583] | **0.635 [0.540, 0.726]** | 0.682 [0.589, 0.771] | 0.537 [0.455, 0.616] | 0.046 [0.036, 0.058] | 0.812 [0.584, 0.974] | 0.705 [0.554, 0.837] | 2.08 | +12 / −4 | +10 / −7 |
| k2-rrf30 · gte-reranker-modernbert-base · fusión RRF · top-30 | 0.813 [0.735, 0.883] | 0.486 [0.385, 0.582] | **0.626 [0.528, 0.717]** | 0.682 [0.591, 0.767] | 0.550 [0.469, 0.630] | 0.045 [0.035, 0.056] | 0.828 [0.653, 0.967] | 0.709 [0.568, 0.836] | 3.12 | +14 / −7 | +12 / −7 |
| k2-rrf50 · gte-reranker-modernbert-base · fusión RRF · top-50 | 0.813 [0.735, 0.883] | 0.514 [0.414, 0.610] | **0.608 [0.509, 0.700]** | 0.682 [0.591, 0.768] | 0.541 [0.458, 0.621] | 0.044 [0.034, 0.056] | 0.818 [0.641, 0.961] | 0.678 [0.555, 0.794] | 5.20 | +14 / −9 | +12 / −7 |

Diferencia contra la base (pareada; empate = el IC de la diferencia incluye 0):

- **k1-rr10**: pkg_hit@1 +0.084 [+0.018, +0.157] mejor; span_hit@2000 +0.065 [+0.000, +0.136] empate; ctx_eff@2000 +0.005 [+0.000, +0.011] empate; ndcg@10 +0.062 [+0.014, +0.113] mejor
- **k1-rr20**: pkg_hit@1 +0.131 [+0.055, +0.211] mejor; span_hit@2000 +0.131 [+0.047, +0.217] mejor; ctx_eff@2000 +0.011 [+0.003, +0.019] mejor; ndcg@10 +0.114 [+0.047, +0.182] mejor
- **k1-rr30**: pkg_hit@1 +0.140 [+0.063, +0.223] mejor; span_hit@2000 +0.168 [+0.075, +0.262] mejor; ctx_eff@2000 +0.013 [+0.005, +0.021] mejor; ndcg@10 +0.136 [+0.063, +0.209] mejor
- **k1-rr50**: pkg_hit@1 +0.131 [+0.047, +0.221] mejor; span_hit@2000 +0.187 [+0.094, +0.281] mejor; ctx_eff@2000 +0.013 [+0.005, +0.021] mejor; ndcg@10 +0.146 [+0.071, +0.221] mejor
- **k1-rrf10**: pkg_hit@1 +0.093 [+0.028, +0.168] mejor; span_hit@2000 +0.065 [+0.018, +0.121] mejor; ctx_eff@2000 +0.005 [+0.000, +0.011] mejor; ndcg@10 +0.044 [+0.022, +0.069] mejor
- **k1-rrf20**: pkg_hit@1 +0.084 [+0.019, +0.157] mejor; span_hit@2000 +0.112 [+0.046, +0.183] mejor; ctx_eff@2000 +0.010 [+0.003, +0.017] mejor; ndcg@10 +0.083 [+0.054, +0.114] mejor
- **k1-rrf30**: pkg_hit@1 +0.093 [+0.028, +0.168] mejor; span_hit@2000 +0.121 [+0.054, +0.194] mejor; ctx_eff@2000 +0.010 [+0.004, +0.018] mejor; ndcg@10 +0.101 [+0.069, +0.133] mejor
- **k1-rrf50**: pkg_hit@1 +0.084 [+0.019, +0.157] mejor; span_hit@2000 +0.149 [+0.076, +0.229] mejor; ctx_eff@2000 +0.011 [+0.004, +0.018] mejor; ndcg@10 +0.108 [+0.073, +0.144] mejor
- **k2-rr10**: pkg_hit@1 +0.047 [-0.029, +0.128] empate; span_hit@2000 +0.037 [-0.038, +0.115] empate; ctx_eff@2000 -0.000 [-0.009, +0.008] empate; ndcg@10 +0.032 [-0.022, +0.088] empate
- **k2-rr20**: pkg_hit@1 +0.037 [-0.046, +0.125] empate; span_hit@2000 -0.028 [-0.134, +0.075] empate; ctx_eff@2000 -0.002 [-0.012, +0.009] empate; ndcg@10 +0.014 [-0.061, +0.086] empate
- **k2-rr30**: pkg_hit@1 +0.019 [-0.065, +0.107] empate; span_hit@2000 -0.019 [-0.136, +0.097] empate; ctx_eff@2000 +0.010 [-0.007, +0.028] empate; ndcg@10 +0.017 [-0.067, +0.100] empate
- **k2-rr50**: pkg_hit@1 -0.019 [-0.110, +0.076] empate; span_hit@2000 -0.093 [-0.210, +0.019] empate; ctx_eff@2000 +0.006 [-0.010, +0.025] empate; ndcg@10 -0.029 [-0.115, +0.055] empate
- **k2-rrf10**: pkg_hit@1 +0.056 [-0.009, +0.127] empate; span_hit@2000 +0.075 [+0.010, +0.143] mejor; ctx_eff@2000 +0.007 [+0.001, +0.013] mejor; ndcg@10 +0.032 [-0.000, +0.064] empate
- **k2-rrf20**: pkg_hit@1 +0.028 [-0.048, +0.111] empate; span_hit@2000 +0.075 [+0.000, +0.151] empate; ctx_eff@2000 +0.006 [+0.000, +0.013] empate; ndcg@10 +0.061 [+0.018, +0.105] mejor
- **k2-rrf30**: pkg_hit@1 +0.047 [-0.036, +0.133] empate; span_hit@2000 +0.065 [-0.019, +0.152] empate; ctx_eff@2000 +0.005 [-0.005, +0.014] empate; ndcg@10 +0.075 [+0.023, +0.126] mejor
- **k2-rrf50**: pkg_hit@1 +0.047 [-0.036, +0.133] empate; span_hit@2000 +0.047 [-0.045, +0.136] empate; ctx_eff@2000 +0.004 [-0.006, +0.013] empate; ndcg@10 +0.065 [+0.005, +0.123] mejor

Fallos de la base en Span@2000 y cuántos rescata cada variante, por tipo (gold fuera del presupuesto: 22, otra librería en el top-1: 17, gold no recuperado: 8):

- **k1-rr10**: gold fuera del presupuesto 8/22, otra librería en el top-1 2/17, gold no recuperado 0/8
- **k1-rr20**: gold fuera del presupuesto 12/22, otra librería en el top-1 5/17, gold no recuperado 0/8
- **k1-rr30**: gold fuera del presupuesto 17/22, otra librería en el top-1 5/17, gold no recuperado 0/8
- **k1-rr50**: gold fuera del presupuesto 18/22, otra librería en el top-1 6/17, gold no recuperado 0/8
- **k1-rrf10**: gold fuera del presupuesto 5/22, otra librería en el top-1 2/17, gold no recuperado 0/8
- **k1-rrf20**: gold fuera del presupuesto 8/22, otra librería en el top-1 5/17, gold no recuperado 0/8
- **k1-rrf30**: gold fuera del presupuesto 9/22, otra librería en el top-1 5/17, gold no recuperado 0/8
- **k1-rrf50**: gold fuera del presupuesto 11/22, otra librería en el top-1 6/17, gold no recuperado 0/8
- **k2-rr10**: gold fuera del presupuesto 9/22, otra librería en el top-1 1/17, gold no recuperado 0/8
- **k2-rr20**: gold fuera del presupuesto 11/22, otra librería en el top-1 3/17, gold no recuperado 0/8
- **k2-rr30**: gold fuera del presupuesto 14/22, otra librería en el top-1 3/17, gold no recuperado 0/8
- **k2-rr50**: gold fuera del presupuesto 11/22, otra librería en el top-1 3/17, gold no recuperado 0/8
- **k2-rrf10**: gold fuera del presupuesto 9/22, otra librería en el top-1 1/17, gold no recuperado 0/8
- **k2-rrf20**: gold fuera del presupuesto 9/22, otra librería en el top-1 3/17, gold no recuperado 0/8
- **k2-rrf30**: gold fuera del presupuesto 11/22, otra librería en el top-1 3/17, gold no recuperado 0/8
- **k2-rrf50**: gold fuera del presupuesto 11/22, otra librería en el top-1 3/17, gold no recuperado 0/8

Costo por reranker:

- **k1** Qwen/Qwen3-Reranker-0.6B: 0.518 s/par, 2950.0 s en total, GPU máx. 2418 MB, truncados 0.0019
- **k2** Alibaba-NLP/gte-reranker-modernbert-base: 0.104 s/par, 592.5 s en total, GPU máx. 1050 MB, truncados 0.0021

Decisión (regla fijada antes de correr): **k2-rrf10** reemplaza a la base.
