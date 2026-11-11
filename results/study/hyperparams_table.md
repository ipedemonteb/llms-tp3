Etapa 3, hiperparámetros de s3 + e4, dev: 107 positivos, 9 negativos, 109 ítems; IC 95% bootstrap sobre ítems (10000 remuestreos)

| Variante | Pkg Hit@1 | Span@1000 | **Span@2000** | Span@4000 | Ef. contexto@2000 | AUROC | Espec.@95 | Chunks | Tokens (M) | Mediana tok | Py partidos | Truncados | Gen. pos / neg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base ⭐ · S3 · 1024 · R1 · tope 150 | 0.766 [0.676, 0.848] | 0.449 [0.350, 0.545] | **0.561 [0.462, 0.658]** | 0.682 [0.589, 0.771] | 0.040 [0.030, 0.051] | 0.822 [0.602, 0.967] | 0.44 | 4345 | 1.69 | 308 | 0.0% | 0.0025 | - |
| cap75 ⭐ · tope 75 | 0.766 [0.685, 0.845] | 0.355 [0.262, 0.450] | **0.421 [0.324, 0.518]** | 0.523 [0.423, 0.624] | 0.029 [0.020, 0.039] | 0.838 [0.673, 0.967] | 0.56 | 2240 | 0.88 | 314 | 0.0% | 0.0031 | - |

Diferencia contra la base (pareada; empate = el IC de la diferencia incluye 0):

- **cap75**: pkg_hit@1 +0.000 [-0.073, +0.074] empate; span_hit@2000 -0.140 [-0.221, -0.064] peor; ctx_eff@2000 -0.011 [-0.020, -0.004] peor; ndcg@10 -0.106 [-0.175, -0.041] peor

Dentro de cada factor (contra el mejor nivel de ese factor):

- **Tope por librería · pkg_hit@1** (mejor: 75): 150 +0.000 [-0.074, +0.073] empate
- **Tope por librería · span_hit@2000** (mejor: 150): 75 -0.140 [-0.221, -0.064] peor

Finalistas para la evaluación con generación (⭐): base, cap75. Correr: `uv run --group local python src/scripts/run_hyperparam_study.py --generation`.
