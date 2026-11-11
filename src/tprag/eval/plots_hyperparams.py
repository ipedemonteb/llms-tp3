"""Gráficos del estudio de hiperparámetros (etapa 3). Los arma `src/scripts/analyze_hyperparams.py` a partir de
`results/study/hyperparams_analysis.json`: no leen nada más. Mismas convenciones que `plots.py` (colores, fondo, sin
título: va en la diapositiva).

Cuatro figuras, todas simples:
1. `1_factores`: barras con IC por nivel de cada factor (tamaño, mínimo, tope, representación), en Pkg Hit@1 y
   Span Hit@2000. Azul = el mejor nivel del factor y los que empatan con él (bootstrap pareado); gris = peores.
2. `2_presupuesto`: curva de Span Hit contra tokens de contexto, por tamaño de chunk y por representación. Es la que
   separa el efecto del tamaño del sesgo de "un chunk grande acierta más fácil".
3. `3_costo`: Span Hit@2000 contra tokens a embeber (costo). Azul = base, naranja = finalistas, gris = el resto.
4. `4_eficiencia`: eficiencia de contexto@2000 de cada variante (cuánto del contexto es respuesta).
"""

from pathlib import Path

from .plots import ACCENT, BASELINE, GRAY, INK, INK2, MUTED, ORANGE, _canvas, _fmt, _frame, _save

SPAN, PKG, EFF = "span_hit@2000", "pkg_hit@1", "ctx_eff@2000"


def _winners(within: dict, metric: str) -> set[str]:
    p = within["metrics"][metric]
    return {p["reference"]} | {i for i, d in p["vs"].items() if d["verdict"] == "empate"}


def fig_factors(result: dict, out_dir: Path) -> Path:
    factors = list(result["within_factor"].items())
    n = len(factors)
    widths = [max(len(w["levels"]), 2) for _, w in factors]
    fig, axes = _canvas(2.2 * sum(widths) / 2 + 1.2, 6.4, cols=n, rows=2, left=0.75, right=0.2, top=0.55, bottom=0.7,
                        wspace=0.28, hspace=0.55)
    for c, (_, w) in enumerate(factors):
        for r, (metric, name) in enumerate(((PKG, "Package Hit@1"), (SPAN, "Span Hit@2000"))):
            ax = axes[r][c]
            win = _winners(w, metric)
            for x, i in enumerate(w["levels"]):
                v = result["variants"][i]["metrics"][metric]
                ax.bar(x, v["value"], 0.6, color=ACCENT if i in win else GRAY, zorder=2)
                ax.plot([x, x], v["ci"], color=INK2, linewidth=1.1, zorder=3)
                for e in v["ci"]:
                    ax.plot([x - 0.08, x + 0.08], [e, e], color=INK2, linewidth=1.1, zorder=3)
                ax.text(x, v["ci"][1] + 0.02, _fmt(v["value"]), ha="center", fontsize=9.5, color=INK, fontweight="semibold")
            ax.set_xticks(range(len(w["levels"])))
            ax.set_xticklabels([w["labels"][i] + ("\n(base)" if i == "base" else "") for i in w["levels"]],
                               color=INK2, fontsize=9.5)
            ax.set_xlim(-0.6, len(w["levels"]) - 0.4)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            if c:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(name, fontsize=10, color=INK2)
            if r == 0:
                ax.set_title(w["title"], fontsize=10.5, loc="left", color=INK, pad=8)
            _frame(ax, grid="y")
    return _save(fig, out_dir, "1_factores")


def fig_budget(result: dict, out_dir: Path) -> Path:
    budgets = result["budgets"]
    fig, axes = _canvas(9.5, 4.2, cols=2, left=0.75, right=1.0, top=0.45, bottom=0.7, wspace=0.55)
    panels = [("Tamaño de chunk (tokens)", "size"), ("Representación", "representation")]
    for ax, (title, factor) in zip(axes[0], panels):
        w = result["within_factor"].get(factor)
        ax.set_title(title, fontsize=10.5, loc="left", color=INK, pad=8)
        if w:
            for k, i in enumerate(w["levels"]):
                ys = [result["variants"][i]["metrics"][f"span_hit@{b}"]["value"] for b in budgets]
                base = i == "base"
                color = ACCENT if base else (ORANGE if factor == "representation" else [GRAY, MUTED, INK2][k % 3])
                ax.plot(budgets, ys, color=color, linewidth=2.4 if base else 1.6, marker="o",
                        markersize=4.5 if base else 3.5, zorder=3 if base else 2)
                ax.text(budgets[-1] + 80, ys[-1], w["labels"][i], va="center", fontsize=9.5, color=color)
        ax.set_xticks(budgets)
        ax.set_xlim(budgets[0] - 100, budgets[-1] + 100)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("tokens de contexto", fontsize=10, color=INK2)
        _frame(ax, grid="y")
    axes[0][0].set_ylabel("Span Hit", fontsize=10, color=INK2)
    return _save(fig, out_dir, "2_presupuesto")


def fig_cost(result: dict, out_dir: Path) -> Path:
    fig, axes = _canvas(8.5, 5.0, left=0.8, right=0.5, top=0.45, bottom=0.8)
    ax = axes[0][0]
    for i, v in result["variants"].items():
        if not v["tokens"]:
            continue
        x, m = v["tokens"] / 1e6, v["metrics"][SPAN]
        color = ACCENT if i == "base" else (ORANGE if i in result["finalists"] else GRAY)
        ax.plot([x, x], m["ci"], color=color, linewidth=1.1, alpha=0.6, zorder=2)
        ax.plot([x], [m["value"]], marker="o", markersize=8, markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=1.2, linestyle="none", zorder=4)
        ax.annotate(i, (x, m["value"]), xytext=(8, -3), textcoords="offset points", fontsize=9.5, color=INK2)
    ax.set_xlabel("tokens a embeber (millones, cl100k)", fontsize=10, color=INK2)
    ax.set_ylabel("Span Hit@2000", fontsize=10, color=INK2)
    ax.set_ylim(0, 1.0)
    _frame(ax, grid="y")
    handles = [ax.plot([], [], "o", color=c, markersize=7)[0] for c in (ACCENT, ORANGE, GRAY)]
    ax.legend(handles, ["base", "finalistas", "resto"], loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK2)
    return _save(fig, out_dir, "3_costo")


def fig_efficiency(result: dict, out_dir: Path) -> Path:
    ids = list(result["variants"])
    fig, axes = _canvas(8.0, 0.42 * len(ids) + 1.2, left=2.4, right=0.7, top=0.35, bottom=0.65)
    ax = axes[0][0]
    ax.set_ylim(len(ids) - 0.5, -0.5)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([result["variants"][i]["label"] for i in ids], color=INK2, fontsize=10)
    top = max(result["variants"][i]["metrics"][EFF]["ci"][1] for i in ids)
    ax.set_xlim(0, top * 1.25)
    for k, i in enumerate(ids):
        m = result["variants"][i]["metrics"][EFF]
        ax.barh(k, m["value"], 0.55, color=ACCENT if i == "base" else GRAY, zorder=2)
        ax.plot(m["ci"], [k, k], color=INK2, linewidth=1.1, zorder=3)
        ax.text(m["ci"][1] + top * 0.02, k, _fmt(m["value"]), va="center", fontsize=10, color=INK, fontweight="semibold")
    ax.set_xlabel("parte del contexto (2000 tokens) que es bloque gold", fontsize=10, color=INK2)
    _frame(ax, grid="x")
    ax.spines["bottom"].set_color(BASELINE)
    return _save(fig, out_dir, "4_eficiencia")


def hyperparam_figures(result: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [fig_factors(result, out_dir), fig_budget(result, out_dir), fig_cost(result, out_dir),
            fig_efficiency(result, out_dir)]
