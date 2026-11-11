Etapa 2, s3, embedders, dev: 107 positivos, 9 negativos, 109 ítems; IC 95% bootstrap sobre ítems (10000 remuestreos)

| Modelo | Pkg Hit@1 | Span@1000 | **Span@2000** | Span@4000 | AUROC | Chunks | Mediana tok | Bloques Py no enteros | Min indexado (ver nota) | Truncados |
|---|---|---|---|---|---|---|---|---|---|---|
| e4 | 0.766 [0.676, 0.848] | 0.449 [0.350, 0.545] | **0.561 [0.462, 0.658]** | 0.682 [0.589, 0.771] | 0.822 [0.602, 0.967] | 4345 | 308 | 0.0% | 4.4 | 0.0025 |
| e3 | 0.654 [0.559, 0.744] | 0.383 [0.287, 0.478] | **0.449 [0.346, 0.547]** | 0.505 [0.402, 0.604] | 0.658 [0.430, 0.859] | 4345 | 308 | 0.0% | 24.5 | 0.0041 |
| e5 | 0.701 [0.608, 0.789] | 0.336 [0.243, 0.433] | **0.421 [0.321, 0.520]** | 0.495 [0.395, 0.594] | 0.762 [0.529, 0.933] | 4345 | 308 | 0.0% | 12.2 | 0.0041 |

Diferencia por pares contra el mejor de cada métrica (empate = el IC de la diferencia incluye 0):

- **pkg_hit@1** (mejor: e4): e3 -0.112 [-0.217, -0.009] peor; e5 -0.065 [-0.174, +0.041] empate
- **span_hit@1000** (mejor: e4): e3 -0.065 [-0.164, +0.030] empate; e5 -0.112 [-0.206, -0.019] peor
- **span_hit@2000** (mejor: e4): e3 -0.112 [-0.223, -0.009] peor; e5 -0.140 [-0.236, -0.048] peor
- **span_hit@4000** (mejor: e4): e3 -0.178 [-0.284, -0.073] peor; e5 -0.187 [-0.294, -0.084] peor
- **auroc** (mejor: e4): e3 -0.164 [-0.335, +0.000] empate; e5 -0.060 [-0.167, +0.056] empate

Min indexado depende del caché (un splitter que comparte textos con otro sale casi gratis): para el costo de cada modelo usar embedders_manifest.json (segundos y memoria del paso) y embedders_smoke.json.
