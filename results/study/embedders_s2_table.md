Etapa 2, s2, embedders, dev: 107 positivos, 9 negativos, 109 ítems; IC 95% bootstrap sobre ítems (10000 remuestreos)

| Modelo | Pkg Hit@1 | Span@1000 | **Span@2000** | Span@4000 | AUROC | Chunks | Mediana tok | Bloques Py no enteros | Min indexado (ver nota) | Truncados |
|---|---|---|---|---|---|---|---|---|---|---|
| e4 | 0.776 [0.691, 0.853] | 0.430 [0.333, 0.526] | **0.551 [0.453, 0.649]** | 0.682 [0.589, 0.771] | 0.825 [0.605, 0.967] | 4357 | 303 | 3.7% | 1.8 | 0.0 |
| e3 | 0.654 [0.558, 0.745] | 0.383 [0.287, 0.478] | **0.439 [0.339, 0.539]** | 0.514 [0.412, 0.613] | 0.665 [0.435, 0.864] | 4357 | 303 | 3.7% | 6.1 | 0.0007 |
| e5 | 0.692 [0.596, 0.778] | 0.346 [0.252, 0.443] | **0.430 [0.330, 0.529]** | 0.505 [0.404, 0.604] | 0.765 [0.532, 0.939] | 4357 | 303 | 3.7% | 2.7 | 0.0014 |

Diferencia por pares contra el mejor de cada métrica (empate = el IC de la diferencia incluye 0):

- **pkg_hit@1** (mejor: e4): e3 -0.121 [-0.221, -0.027] peor; e5 -0.084 [-0.188, +0.018] empate
- **span_hit@1000** (mejor: e4): e3 -0.047 [-0.140, +0.049] empate; e5 -0.084 [-0.174, +0.000] empate
- **span_hit@2000** (mejor: e4): e3 -0.112 [-0.220, -0.009] peor; e5 -0.121 [-0.215, -0.036] peor
- **span_hit@4000** (mejor: e4): e3 -0.168 [-0.274, -0.065] peor; e5 -0.178 [-0.282, -0.075] peor
- **auroc** (mejor: e4): e3 -0.160 [-0.328, +0.002] empate; e5 -0.059 [-0.168, +0.054] empate

Min indexado depende del caché (un splitter que comparte textos con otro sale casi gratis): para el costo de cada modelo usar embedders_manifest.json (segundos y memoria del paso) y embedders_smoke.json.
