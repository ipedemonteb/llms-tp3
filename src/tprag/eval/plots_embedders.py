"""Gráficos del estudio de embedders (etapa 2). Los arma `src/scripts/plot_embedders.py` a partir de los análisis de
`analyze_splitters.py --vary model` (uno por splitter), el manifiesto de la corrida y los reportes de índice.

Mismas convenciones que `plots.py` (se reusan sus piezas): azul = el mejor en Span Hit@2000 y los que empatan con él
según el bootstrap pareado, gris = significativamente peores; el color se calcula del resultado. Cuando una figura
compara dos splitters, el splitter es la variable categórica (azul = S3, naranja = S2). Sin título: va en la diapositiva.
"""

from pathlib import Path

from matplotlib.patches import Patch

from .plots import (ACCENT, GRAY, INK, INK2, ORANGE, PX, SURFACE, _bar, _canvas, _color, _dot, _fmt, _frame, _legend,
                    _rows, _save, _whisker)

NAMES = {"e3": "E3 · bge-m3", "e4": "E4 · jina-code", "e5": "E5 · nomic"}
SHORT = {"e3": "E3", "e4": "E4", "e5": "E5"}
SPLITTER_COLOR = {"s3": ACCENT, "s2": ORANGE}


def _order(results: dict, base: str = "s3") -> list[str]:
    r = results[base]["splitters"]
    return sorted(r, key=lambda m: -r[m]["span_hit@2000"]["value"])


def _top(result: dict) -> set[str]:
    paired = result["paired"]["span_hit@2000"]
    return {paired["reference"]} | {m for m, d in paired["vs"].items() if d["verdict"] == "empate"}


def seconds_per_text(index: dict) -> float | None:
    """Segundos por texto realmente calculado (no por texto del índice: el caché hace que los splitters no se puedan
    comparar por tiempo total)."""
    computed = index["use_cases"] - index["cache_hits"]
    return index["seconds"] / computed if computed > 50 else None


def costs(results: dict, manifest: dict, indexes: dict, base: str = "s3") -> dict[str, dict]:
    """Por modelo: segundos por 100 textos y memoria máxima del paso de índice (None si no se registró)."""
    out = {}
    for m, name in results[base]["collections"].items():
        spt = seconds_per_text(indexes[name])
        attempts = manifest.get("steps", {}).get(name, {}).get("index", {})
        out[m] = {"s_per_100": spt * 100 if spt else None, "peak_gb": (attempts.get("peak_memory_mb") or 0) / 1024 or None}
    return out


# --- Figuras --------------------------------------------------------------------------------------------------

def fig_span_hit(results: dict, out_dir: Path) -> Path:
    order = _order(results)
    n = len(order)
    fig, axes = _canvas(9, 0.5 * n + 1.1, cols=2, left=2.0, right=0.4, top=0.45, bottom=0.6, wspace=0.12)
    for k, (ax, s) in enumerate(zip(axes[0], ("s3", "s2"))):
        top = _top(results[s])
        ax.set_xlim(0, 0.8)
        _rows(ax, [NAMES[m] for m in order], show=k == 0)
        for i, m in enumerate(order):
            v = results[s]["splitters"][m]["span_hit@2000"]
            _bar(ax, i, v["value"], 0.5, _color(m, top))
            _whisker(ax, i, *v["ci"])
            ax.text(v["ci"][1] + 0.015, i, _fmt(v["value"]), va="center", fontsize=10.5, color=INK, fontweight="semibold")
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_title(f"Span Hit@2000 · {s.upper()}", fontsize=11, loc="left", color=INK, pad=8)
        _frame(ax)
    return _save(fig, out_dir, "1_span_hit_2000")


def fig_ties(results: dict, out_dir: Path) -> Path:
    order = _order(results)
    n = len(order)
    fig, axes = _canvas(9, 0.62 * n + 1.5, cols=2, left=2.0, right=0.4, top=0.45, bottom=0.75, wspace=0.12)
    panels = [("pkg_hit@1", "Package Hit@1", (0.4, 1.0), [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], None),
              ("auroc", "AUROC", (0.3, 1.0), [0.3, 0.5, 0.7, 0.9], 0.5)]
    for k, (ax, (metric, title, lim, ticks, ref)) in enumerate(zip(axes[0], panels)):
        ax.set_xlim(*lim)
        _rows(ax, [NAMES[m] for m in order], show=k == 0)
        for i, m in enumerate(order):
            for s, dy in (("s3", -0.17), ("s2", 0.17)):
                v = results[s]["splitters"][m][metric]
                _whisker(ax, i + dy, *v["ci"], color=SPLITTER_COLOR[s], cap=0.1)
                _dot(ax, v["value"], i + dy, SPLITTER_COLOR[s])
        if ref is not None:
            ax.axvline(ref, color=INK2, linewidth=PX / 2)
            ax.text(ref + 0.01, n - 0.55, "azar", fontsize=9, color=INK2, va="bottom")
        ax.set_xticks(ticks)
        ax.set_title(title, fontsize=11, loc="left", color=INK, pad=8)
        _frame(ax)
    _legend(fig, [Patch(color=SPLITTER_COLOR[s]) for s in ("s3", "s2")], ["S3", "S2"], ncol=2)
    return _save(fig, out_dir, "2_pkg_hit_y_auroc")


def fig_budget(results: dict, out_dir: Path) -> Path:
    order = _order(results)
    budgets = results["s3"]["budgets"]
    fig, axes = _canvas(9, 4.9, cols=len(order), rows=2, left=0.65, right=0.3, top=0.3, bottom=0.75, wspace=0.12,
                        hspace=0.5)
    for r, s in enumerate(("s3", "s2")):
        top = _top(results[s])
        for c, m in enumerate(order):
            ax = axes[r][c]
            ax.set_xlim(-0.4, len(budgets) - 0.6)
            ax.set_ylim(0, 0.85)
            color = _color(m, top)
            values = [results[s]["splitters"][m][f"span_hit@{b}"] for b in budgets]
            for i, v in enumerate(values):
                ax.plot([i, i], v["ci"], color=color, linewidth=PX / 2 * 1.1, alpha=0.55, zorder=3)
            ax.plot(range(len(budgets)), [v["value"] for v in values], color=color, linewidth=PX, zorder=4,
                    solid_capstyle="round", solid_joinstyle="round")
            for i, v in enumerate(values):
                _dot(ax, i, v["value"], color)
            two = budgets.index(2000)
            ax.text(two - 0.1, values[two]["value"] + 0.035, _fmt(values[two]["value"]), ha="right", va="bottom",
                    fontsize=10.5, color=INK, fontweight="semibold")
            ax.set_xticks(range(len(budgets)))
            ax.set_xticklabels([f"{b:,}" for b in budgets])
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
            ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8"] if c == 0 else [""] * 5)
            ax.set_title(f"{NAMES[m]} · {s.upper()}", fontsize=10.5, loc="left", color=INK, pad=6)
            _frame(ax, grid="y")
    for ax in axes[1]:
        ax.set_xlabel("presupuesto (tokens)", fontsize=10)
    return _save(fig, out_dir, "3_sensibilidad_presupuesto")


def fig_paired(results: dict, out_dir: Path) -> Path:
    panels = [("span_hit@2000", "Span Hit@2000"), ("pkg_hit@1", "Package Hit@1")]
    order = _order(results)
    rows = [(s, m) for s in ("s3", "s2") for m in order]  # el que es el mejor de una métrica sale como "el mejor"
    fig, axes = _canvas(9.5, 0.42 * len(rows) + 1.2, cols=2, left=1.9, right=0.3, top=0.5, bottom=0.55, wspace=0.12)
    for k, (ax, (metric, title)) in enumerate(zip(axes[0], panels)):
        data = [(s, m, results[s]["paired"][metric]["reference"], results[s]["paired"][metric]["vs"].get(m))
                for s, m in rows]
        diffs = [d for *_, d in data if d]
        lo, hi = min(d["ci"][0] for d in diffs), max(d["ci"][1] for d in diffs)
        span = max(hi, 0) - lo
        ax.set_xlim(lo - 0.04 * span - 0.02, max(hi, 0) + 0.8 * span)
        _rows(ax, [f"{SHORT[m]} · {s.upper()}" for s, m, _, _ in data], show=k == 0)
        ax.axvline(0, color=INK2, linewidth=PX / 2)
        for i, (_, _, _, d) in enumerate(data):
            if d is None:  # este modelo es el mejor en esta métrica
                ax.text(1.0, i, "es el mejor", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=10,
                        color=INK2)
                continue
            color = ACCENT if d["verdict"] == "empate" else GRAY
            _whisker(ax, i, *d["ci"], color=color)
            _dot(ax, d["diff"], i, color)
            ax.text(1.0, i, f"{d['diff']:+.2f} · {d['verdict']}", transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=10, color=INK, bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2})
        ax.set_title(f"{title}: diferencia contra el mejor", fontsize=11, loc="left", color=INK, pad=8)
        _frame(ax)
    return _save(fig, out_dir, "4_diferencias_por_pares")


def fig_cost(results: dict, cost: dict, out_dir: Path) -> Path:
    order = _order(results)
    n = len(order)
    fig, axes = _canvas(9, 0.5 * n + 1.1, cols=2, left=2.0, right=0.4, top=0.45, bottom=0.55, wspace=0.12)
    top = _top(results["s3"])
    panels = [("s_per_100", "Segundos por 100 textos", lambda v: f"{v:.0f}"),
              ("peak_gb", "Memoria máxima del proceso (GB)", lambda v: f"{v:.1f}")]
    for k, (ax, (key, title, fmt)) in enumerate(zip(axes[0], panels)):
        known = [cost[m][key] for m in order if cost[m][key]]
        lim = max(known) * 1.2
        ax.set_xlim(0, lim)
        _rows(ax, [NAMES[m] for m in order], show=k == 0)
        for i, m in enumerate(order):
            v = cost[m][key]
            if v:
                _bar(ax, i, v, 0.5, _color(m, top))
                ax.text(v + lim * 0.015, i, fmt(v), va="center", fontsize=10, color=INK, fontweight="semibold")
            else:
                ax.text(lim * 0.015, i, "no registrado", va="center", fontsize=9.5, color=INK2)
        ax.set_title(title, fontsize=10.5, loc="left", color=INK, pad=8)
        _frame(ax)
    return _save(fig, out_dir, "5_costo")


def fig_tradeoff(results: dict, cost: dict, out_dir: Path) -> Path:
    """Calidad (Span Hit@2000 en S3, con su intervalo) contra costo de indexar: el ganador es el de arriba a la
    izquierda."""
    order = _order(results)
    top = _top(results["s3"])
    xs = [cost[m]["s_per_100"] for m in order if cost[m]["s_per_100"]]
    fig, ((ax,),) = _canvas(6.5, 4.2, left=0.8, right=0.5, top=0.3, bottom=0.8)
    ax.set_xlim(0, max(xs) * 1.35)
    ax.set_ylim(0.3, 0.7)
    for m in order:
        x, v = cost[m]["s_per_100"], results["s3"]["splitters"][m]["span_hit@2000"]
        if x is None:
            continue
        color = _color(m, top)
        ax.plot([x, x], v["ci"], color=color, linewidth=PX / 2 * 1.1, alpha=0.6, zorder=3)
        _dot(ax, x, v["value"], color)
        ax.text(x + max(xs) * 0.03, v["value"], f"{SHORT[m]} · {_fmt(v['value'])}", va="center", fontsize=10.5,
                color=INK, fontweight="semibold")
    ax.set_xlabel("segundos por 100 textos (más a la izquierda = más rápido)", fontsize=10)
    ax.set_ylabel("Span Hit@2000 · S3", fontsize=10)
    _frame(ax, grid="y")
    return _save(fig, out_dir, "6_calidad_vs_costo")


def fig_subsets(results: dict, out_dir: Path) -> Path:
    order = _order(results)
    n = len(order)
    subsets = [("manual_deprecated", "Manual", ACCENT), ("synthetic", "Sintético", ORANGE)]
    sizes = {k: results["s3"]["splitters"][order[0]]["by_subset"][k]["targets"] for k, _, _ in subsets}
    fig, ((ax,),) = _canvas(7, 0.85 * n + 1.35, left=2.2, right=0.5, top=0.2, bottom=0.95)
    ax.set_xlim(0, 1.0)
    _rows(ax, [NAMES[m] for m in order])
    h, gap = 0.3, 0.035
    for i, m in enumerate(order):
        for j, (key, _, color) in enumerate(subsets):
            y = i + (j - 0.5) * (h + gap)
            v = results["s3"]["splitters"][m]["by_subset"][key]["span_hit@2000"]
            _bar(ax, y, v, h, color)
            ax.text(v + 0.015, y, _fmt(v), va="center", fontsize=9.5, color=INK)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _frame(ax)
    ax.set_xlabel("Span Hit@2000 · S3", fontsize=10)
    _legend(fig, [Patch(color=c) for _, _, c in subsets], [f"{label} (n = {sizes[k]})" for k, label, _ in subsets],
            ncol=2)
    return _save(fig, out_dir, "7_manual_vs_sintetico")


def embedder_figures(results: dict, manifest: dict, indexes: dict, out_dir: Path) -> list[Path]:
    """`results`: {"s3": análisis, "s2": análisis} (`embedders_<splitter>_analysis.json`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cost = costs(results, manifest, indexes)
    return [fig_span_hit(results, out_dir), fig_ties(results, out_dir), fig_budget(results, out_dir),
            fig_paired(results, out_dir), fig_cost(results, cost, out_dir), fig_tradeoff(results, cost, out_dir),
            fig_subsets(results, out_dir)]
