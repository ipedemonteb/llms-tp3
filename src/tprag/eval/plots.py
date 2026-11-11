"""Gráficos del estudio de splitters (etapa 1). Los arma `src/scripts/analyze_splitters.py` a partir del resultado del
análisis (`results/study/splitters_analysis.json`): no leen nada más.

Convención de color (skill dataviz, énfasis): azul = el mejor y los que empatan con él en Span Hit@2000 (según el
bootstrap pareado); gris = significativamente peores. Se calcula del resultado, no está fijo. Solo la figura de
manual vs sintético usa dos colores categóricos (azul y naranja, ranuras 1 y 2 de la paleta de referencia).

Pensadas para pegar en diapositivas: sin título ni descripción (eso va en la diapositiva), solo el gráfico. Cada una
se guarda en PNG con fondo claro. La tabla equivalente está en `splitters_table.md`.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402
from matplotlib.transforms import IdentityTransform  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
ACCENT, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
DPI = 200
PX = DPI / 96  # píxeles de la figura por píxel CSS (para los espesores de la especificación)
NAMES = {"s0": "S0 · por tamaño", "s1": "S1 · tamaño + Markdown", "s2": "S2 · una sección",
         "s3": "S3 · casos de uso", "char": "Char · párrafos", "tok": "Tok · ventanas de tokens"}
SHORT = {"s0": "S0", "s1": "S1", "s2": "S2", "s3": "S3", "char": "Char", "tok": "Tok"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASELINE, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.unicode_minus": True,
})


# --- Datos ----------------------------------------------------------------------------------------------------

def _order(result: dict, metric: str = "span_hit@2000") -> list[str]:
    return sorted(result["splitters"], key=lambda s: -result["splitters"][s][metric]["value"])


def _top(result: dict) -> set[str]:
    """El mejor en Span Hit@2000 y los que empatan con él."""
    paired = result["paired"]["span_hit@2000"]
    return {paired["reference"]} | {s for s, d in paired["vs"].items() if d["verdict"] == "empate"}


# --- Piezas ---------------------------------------------------------------------------------------------------

def _canvas(width, height, cols=1, rows=1, left=1.9, right=0.4, top=0.95, bottom=0.7, wspace=0.25, hspace=0.45):
    """Figura con márgenes fijos en pulgadas: las barras se dibujan en píxeles y no admiten un layout automático."""
    fig, axes = plt.subplots(rows, cols, figsize=(width, height), dpi=DPI, squeeze=False,
                             gridspec_kw={"wspace": wspace, "hspace": hspace})
    fig.subplots_adjust(left=left / width, right=1 - right / width, top=1 - top / height, bottom=bottom / height)
    return fig, axes


def _frame(ax, grid="x"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE if grid == "y" else SURFACE)
    ax.tick_params(length=0, labelsize=9.5)
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color=GRID, linewidth=0.6)


def _bar(ax, y, x1, height, color, x0=0.0, radius_px=4):
    """Barra horizontal: cuadrada en la base y con la punta redondeada de 4 px (la figura tiene límites ya fijos)."""
    if x1 <= x0:
        return
    (a, y_a), (c, y_c) = ax.transData.transform([(x0, y - height / 2), (x1, y + height / 2)])
    b, d = min(y_a, y_c), max(y_a, y_c)  # el eje y va invertido (primera fila arriba)
    r = radius_px * PX
    patch = FancyBboxPatch((a - r, b), c - a + r, d - b, boxstyle=f"round,pad=0,rounding_size={min(r, (d - b) / 2)}",
                           transform=IdentityTransform(), facecolor=color, edgecolor="none", linewidth=0)
    ax.add_patch(patch)
    patch.set_clip_path(Rectangle((a, b - 1), c - a + r + 1, d - b + 2, transform=IdentityTransform()))


def _whisker(ax, y, lo, hi, color=INK2, cap=0.12):
    ax.plot([lo, hi], [y, y], color=color, linewidth=1.1 * PX / 2, solid_capstyle="butt", zorder=3)
    for x in (lo, hi):
        ax.plot([x, x], [y - cap / 2, y + cap / 2], color=color, linewidth=1.1 * PX / 2, zorder=3)


def _dot(ax, x, y, color):
    ax.plot([x], [y], marker="o", markersize=7, markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.5,
            linestyle="none", zorder=4)


def _rows(ax, labels: list[str], show=True):
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels if show else [""] * len(labels), color=INK2, fontsize=10)


def _legend(fig, handles, labels, ncol=3):
    w, h = fig.get_size_inches()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.3 / w, 0.05 / h), ncol=ncol, frameon=False,
               fontsize=9.5, handlelength=1.2, columnspacing=1.6, labelcolor=INK2)


def _color(s, top):
    return ACCENT if s in top else GRAY


def _fmt(v):
    return f"{v:.2f}"


def _save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# --- Figuras --------------------------------------------------------------------------------------------------

def fig_span_hit(result: dict, out_dir: Path) -> Path:
    order, top = _order(result), _top(result)
    n = len(order)
    fig, ((ax,),) = _canvas(7, 0.5 * n + 1.0, left=2.4, right=0.5, top=0.2, bottom=0.65)
    ax.set_xlim(0, 0.8)
    _rows(ax, [NAMES[s] for s in order])
    for i, s in enumerate(order):
        m = result["splitters"][s]["span_hit@2000"]
        _bar(ax, i, m["value"], 0.5, _color(s, top))
        _whisker(ax, i, *m["ci"])
        ax.text(m["ci"][1] + 0.015, i, _fmt(m["value"]), va="center", fontsize=10.5, color=INK, fontweight="semibold")
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    _frame(ax)
    ax.set_xlabel("Span Hit@2000", fontsize=10)
    return _save(fig, out_dir, "1_span_hit_2000")


def fig_ties(result: dict, out_dir: Path) -> Path:
    order = _order(result)
    n = len(order)
    fig, axes = _canvas(8.5, 0.5 * n + 1.1, cols=2, left=2.4, right=0.4, top=0.45, bottom=0.55, wspace=0.2)
    panels = [("pkg_hit@1", "Package Hit@1", (0.5, 1.0), [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], None),
              ("auroc", "AUROC", (0.4, 1.0), [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 0.5)]
    for k, (ax, (metric, title, lim, ticks, ref)) in enumerate(zip(axes[0], panels)):
        ax.set_xlim(*lim)
        _rows(ax, [NAMES[s] for s in order], show=k == 0)
        for i, s in enumerate(order):
            m = result["splitters"][s][metric]
            _whisker(ax, i, *m["ci"], color=BASELINE)
            _dot(ax, m["value"], i, ACCENT)
            ax.text(m["value"], i - 0.3, _fmt(m["value"]), ha="center", va="bottom", fontsize=9.5, color=INK)
        if ref is not None:
            ax.axvline(ref, color=INK2, linewidth=PX / 2)
            ax.text(ref + 0.01, n - 0.55, "azar", fontsize=9, color=INK2, va="bottom")
        ax.set_xticks(ticks)
        ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
        _frame(ax)
    return _save(fig, out_dir, "2_pkg_hit_y_auroc")


def fig_budget(result: dict, out_dir: Path) -> Path:
    order, top = _order(result), _top(result)
    budgets = result["budgets"]
    fig, axes = _canvas(9, 4.9, cols=3, rows=2, left=0.65, right=0.3, top=0.3, bottom=0.75, wspace=0.12, hspace=0.5)
    for idx, (ax, s) in enumerate(zip(axes.flat, order)):
        ax.set_xlim(-0.4, len(budgets) - 0.6)
        ax.set_ylim(0, 0.85)
        color = _color(s, top)
        values = [result["splitters"][s][f"span_hit@{b}"] for b in budgets]
        for i, m in enumerate(values):
            ax.plot([i, i], m["ci"], color=color, linewidth=PX / 2 * 1.1, alpha=0.55, zorder=3)
        ax.plot(range(len(budgets)), [m["value"] for m in values], color=color, linewidth=PX, zorder=4,
                solid_capstyle="round", solid_joinstyle="round")
        for i, m in enumerate(values):
            _dot(ax, i, m["value"], color)
        two = budgets.index(2000)
        ax.text(two - 0.1, values[two]["value"] + 0.035, _fmt(values[two]["value"]), ha="right", va="bottom",
                fontsize=10.5, color=INK, fontweight="semibold")
        ax.set_xticks(range(len(budgets)))
        ax.set_xticklabels([f"{b:,}" for b in budgets])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8"] if idx % 3 == 0 else [""] * 5)
        ax.set_title(NAMES[s], fontsize=10.5, loc="left", color=INK, pad=6)
        _frame(ax, grid="y")
    for ax in axes[1]:
        ax.set_xlabel("presupuesto (tokens)", fontsize=10)
    return _save(fig, out_dir, "3_sensibilidad_presupuesto")


def fig_paired(result: dict, out_dir: Path) -> Path:
    fig, axes = _canvas(9, 3.4, cols=2, left=0.5, right=0.3, top=0.5, bottom=0.6, wspace=0.12)
    panels = [("span_hit@2000", "Span Hit@2000"), ("pkg_hit@1", "Package Hit@1")]
    for ax, (metric, title) in zip(axes[0], panels):
        p = result["paired"][metric]
        rows = sorted(p["vs"].items(), key=lambda kv: -kv[1]["diff"])
        lo, hi = min(d["ci"][0] for _, d in rows), max(d["ci"][1] for _, d in rows)
        span = max(hi, 0) - lo
        ax.set_xlim(lo - 0.04 * span - 0.02, max(hi, 0) + 0.7 * span)
        _rows(ax, [SHORT[s] for s, _ in rows])
        ax.axvline(0, color=INK2, linewidth=PX / 2)
        for i, (s, d) in enumerate(rows):
            color = ACCENT if d["verdict"] == "empate" else GRAY
            _whisker(ax, i, *d["ci"], color=color)
            _dot(ax, d["diff"], i, color)
            ax.text(1.0, i, f"{d['diff']:+.2f} · {d['verdict']}", transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=10, color=INK, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2})
        ax.set_title(f"{title} (contra {SHORT[p['reference']]})", fontsize=11, loc="left", color=INK, pad=8)
        _frame(ax)
        ax.tick_params(axis="y", labelsize=10.5)
    return _save(fig, out_dir, "4_diferencias_por_pares")


def fig_structure(result: dict, out_dir: Path) -> Path:
    order, top = _order(result), _top(result)
    n = len(order)
    fig, axes = _canvas(10, 0.5 * n + 1.1, cols=3, left=2.2, right=0.3, top=0.45, bottom=0.55, wspace=0.12)
    panels = [("median_tokens", "Mediana de tokens por chunk", 1.0, (0, 1300), [0, 500, 1000], lambda v: f"{v:,.0f}"),
              ("indexed_tokens", "Tokens a embeber (millones)", 1e-6, (0, 4.4), [0, 1, 2, 3, 4],
               lambda v: f"{v:.1f}"),
              ("python_blocks_not_intact", "Bloques de Python partidos (%)", 100.0, (0, 6.8), [0, 2, 4, 6],
               lambda v: f"{v:.1f}")]
    for k, (ax, (key, title, scale, lim, ticks, fmt)) in enumerate(zip(axes[0], panels)):
        ax.set_xlim(*lim)
        _rows(ax, [NAMES[s] for s in order], show=k == 0)
        for i, s in enumerate(order):
            v = result["splitters"][s][key] * scale
            _bar(ax, i, v, 0.5, _color(s, top))
            ax.text(v + lim[1] * 0.015, i, fmt(v), va="center", fontsize=10, color=INK, fontweight="semibold")
        ax.set_xticks(ticks)
        ax.set_title(title, fontsize=10.5, loc="left", color=INK, pad=8)
        _frame(ax)
    return _save(fig, out_dir, "5_costo_y_estructura")


def fig_subsets(result: dict, out_dir: Path) -> Path:
    order = _order(result)
    n = len(order)
    subsets = [("manual_deprecated", "Manual", ACCENT), ("synthetic", "Sintético", ORANGE)]
    sizes = {k: result["splitters"][order[0]]["by_subset"][k]["targets"] for k, _, _ in subsets}
    fig, ((ax,),) = _canvas(7, 0.85 * n + 1.35, left=2.4, right=0.5, top=0.2, bottom=0.95)
    ax.set_xlim(0, 1.0)
    _rows(ax, [NAMES[s] for s in order])
    h, gap = 0.3, 0.035
    for i, s in enumerate(order):
        for j, (key, _, color) in enumerate(subsets):
            y = i + (j - 0.5) * (h + gap)
            v = result["splitters"][s]["by_subset"][key]["span_hit@2000"]
            _bar(ax, y, v, h, color)
            ax.text(v + 0.015, y, _fmt(v), va="center", fontsize=9.5, color=INK)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _frame(ax)
    ax.set_xlabel("Span Hit@2000", fontsize=10)
    _legend(fig, [Patch(color=c) for _, _, c in subsets],
            [f"{label} (n = {sizes[k]})" for k, label, _ in subsets], ncol=2)
    return _save(fig, out_dir, "6_manual_vs_sintetico")


def splitter_figures(result: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [fig(result, out_dir) for fig in (fig_span_hit, fig_ties, fig_budget, fig_paired, fig_structure,
                                             fig_subsets)]
