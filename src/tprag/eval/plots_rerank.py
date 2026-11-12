"""Gráficos del estudio del reranker. Los arma `src/scripts/analyze_rerank.py` a partir de
`results/study/rerank_analysis.json`: no leen nada más. Mismas convenciones que `plots.py` (colores, fondo, sin título:
va en la diapositiva).

1. `1_candidatos`: Package Hit@1 y Span Hit@2000 contra la cantidad de candidatos K, una línea por reranker y modo
   (azul = K1, naranja = K2; continua = reranker, punteada = fusión RRF). La base es la banda gris (su IC).
2. `2_presupuesto`: curva de Span Hit contra tokens de contexto: la base y la mejor variante de cada reranker.
3. `3_antes_despues`: por variante, aciertos de Span@2000 que rescata (derecha) y que rompe (izquierda) respecto de la
   base. Es lo que muestra si el reranker arregla casos o solo los mueve de lugar.
"""

from pathlib import Path

from .plots import ACCENT, GRAY, INK, INK2, MUTED, ORANGE, _canvas, _frame, _save

SPAN, PKG = "span_hit@2000", "pkg_hit@1"
COLORS = {"k1": ACCENT, "k2": ORANGE}
DASH = {"rr": "-", "rrf": "--"}


def _series(result: dict) -> dict[tuple[str, str], list[str]]:
    """(reranker, modo) → ids de variantes ordenados por K."""
    out: dict[tuple[str, str], list[str]] = {}
    for i, v in result["variants"].items():
        out.setdefault((v["reranker"], v["mode"]), []).append(i)
    return {s: sorted(ids, key=lambda i: result["variants"][i]["k"]) for s, ids in out.items()}


def fig_candidates(result: dict, out_dir: Path) -> Path:
    fig, axes = _canvas(10.0, 4.4, cols=2, left=0.75, right=0.3, top=0.45, bottom=0.75, wspace=0.25)
    series = _series(result)
    ks = result["ks"]
    for ax, (metric, name) in zip(axes[0], ((PKG, "Package Hit@1"), (SPAN, "Span Hit@2000"))):
        b = result["base"]["metrics"][metric]
        ax.axhspan(*b["ci"], color=GRAY, alpha=0.15, linewidth=0, zorder=1)
        ax.axhline(b["value"], color=GRAY, linewidth=1.4, zorder=2)
        ax.text(ks[-1] + 4, b["value"], "base", va="center", fontsize=9.5, color=MUTED)
        low = min([b["ci"][0]] + [v["metrics"][metric]["ci"][0] for v in result["variants"].values()])
        handles, labels = [], []
        for n, ((r, mode), ids) in enumerate(sorted(series.items())):
            xs = [result["variants"][i]["k"] + (n - 1.5) * 0.5 for i in ids]  # corridos para que no se pisen los IC
            ys = [result["variants"][i]["metrics"][metric]["value"] for i in ids]
            for x, i in zip(xs, ids):
                ax.plot([x, x], result["variants"][i]["metrics"][metric]["ci"], color=COLORS.get(r, INK2),
                        linewidth=1.0, alpha=0.45, zorder=2)
            (line,) = ax.plot(xs, ys, color=COLORS.get(r, INK2), linestyle=DASH[mode], linewidth=2.0, marker="o",
                              markersize=4.5, zorder=3)
            handles.append(line)
            labels.append(f"{result['rerankers'][r]['label']} · {result['modes'][mode]}")
        ax.set_title(name, fontsize=10.5, loc="left", color=INK, pad=8)
        ax.set_xticks(ks)
        ax.set_xlim(ks[0] - 4, ks[-1] + 10)
        ax.set_ylim(min(0.3, int(low * 10) / 10), 1.0)  # baja si alguna variante cae por debajo de 0,3
        ax.set_xlabel("candidatos que reordena el reranker (K)", fontsize=10, color=INK2)
        _frame(ax, grid="y")
    axes[0][1].legend(handles, labels, loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
    return _save(fig, out_dir, "1_candidatos")


def fig_budget(result: dict, out_dir: Path) -> Path:
    budgets = result["budgets"]
    fig, axes = _canvas(7.0, 4.4, left=0.75, right=0.3, top=0.3, bottom=0.75)
    ax = axes[0][0]
    ys = [result["base"]["metrics"][f"span_hit@{b}"]["value"] for b in budgets]
    handles = [ax.plot(budgets, ys, color=GRAY, linewidth=2.4, marker="o", markersize=4.5, zorder=2)[0]]
    labels = ["base (sin reranker)"]
    for r in result["rerankers"]:
        ids = [i for i, v in result["variants"].items() if v["reranker"] == r]
        best = max(ids, key=lambda i: result["variants"][i]["metrics"][SPAN]["value"])
        ys = [result["variants"][best]["metrics"][f"span_hit@{b}"]["value"] for b in budgets]
        handles.append(ax.plot(budgets, ys, color=COLORS.get(r, INK2), linewidth=2.2, marker="o", markersize=4.5,
                               zorder=3)[0])
        labels.append(result["variants"][best]["label"])
    ax.set_xticks(budgets)
    ax.set_xlim(budgets[0] - 100, budgets[-1] + 100)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("tokens de contexto", fontsize=10, color=INK2)
    ax.set_ylabel("Span Hit", fontsize=10, color=INK2)
    ax.legend(handles, labels, loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
    _frame(ax, grid="y")
    return _save(fig, out_dir, "2_presupuesto")


def fig_moves(result: dict, out_dir: Path) -> Path:
    ids = list(result["variants"])
    fig, axes = _canvas(8.0, 0.36 * len(ids) + 1.2, left=3.6, right=0.5, top=0.3, bottom=0.65)
    ax = axes[0][0]
    top = max(max(v["transitions"]["span"]["rescued"], v["transitions"]["span"]["broken"])
              for v in result["variants"].values()) or 1
    ax.set_xlim(-top * 1.3, top * 1.3)
    ax.set_ylim(len(ids) - 0.5, -0.5)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([result["variants"][i]["label"] for i in ids], color=INK2, fontsize=9)
    for n, i in enumerate(ids):
        tr = result["variants"][i]["transitions"]["span"]
        ax.barh(n, tr["rescued"], 0.6, color=ACCENT, zorder=2)
        ax.barh(n, -tr["broken"], 0.6, color=ORANGE, zorder=2)
        ax.text(tr["rescued"] + top * 0.03, n, f"+{tr['rescued']}", va="center", fontsize=9, color=INK)
        ax.text(-tr["broken"] - top * 0.03, n, f"−{tr['broken']}" if tr["broken"] else "0", va="center", ha="right",
                fontsize=9, color=INK)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("Span@2000 contra la base: aciertos rotos (−) y rescatados (+)", fontsize=9.5, color=INK2)
    _frame(ax, grid="x")
    return _save(fig, out_dir, "3_antes_despues")


def rerank_figures(result: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [fig_candidates(result, out_dir), fig_budget(result, out_dir), fig_moves(result, out_dir)]
