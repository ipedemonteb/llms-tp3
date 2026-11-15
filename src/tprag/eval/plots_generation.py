"""Gráficos del estudio del generador (metricas.md §3-§4). Los arma `src/scripts/plots_generation.py` a partir de los
análisis de las corridas permisivas de dev (`generation_<corrida>_analysis.json`), del detalle de cada corrida (para
el nivel 1 anterior) y del acuerdo de la revisión manual (`review_agreement.json`).

Misma convención que `plots.py` (skill dataviz): sin título (va en la diapositiva), fondo claro, PNG. Color = versión
del sistema: azul = base, naranja = reranker k1-rr30 (ranuras 1 y 2 de la paleta de referencia); gris = referencia
(nivel 1 anterior, totales). λ se distingue por el tipo de línea, no por otro color.
"""

import math
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tprag.eval.plots import (ACCENT, BASELINE, GRAY, INK, INK2, MUTED, ORANGE, _bar, _canvas, _dot, _frame,
                              _legend, _save, _whisker)
from tprag.eval.noise import CLEAN, NO_ANSWER, NOISY_FIRST, NOISY_LATER
from tprag.eval.threshold import BAD, GOOD, WRONG_CHANGE, Target, candidates
from tprag.generate.schema import REPLACED

VERSIONS = ("base", "rerank")
NAME = {"base": "Base", "rerank": "Reranker k1-rr30"}
COLOR = {"base": ACCENT, "rerank": ORANGE}
THRESHOLD_LABEL = {"base": "Umbral (coseno del top-1)", "rerank": "Umbral (máximo del puntaje de Qwen)"}
LAMBDA_STYLE = {1.0: ":", 2.0: "-", 3.0: "--"}
LINE = 1.5  # 2 px CSS a 200 dpi
MIN_N = 5  # Faithfulness y AR se grafican solo con al menos 5 scripts que pasan el umbral


# --- Datos ----------------------------------------------------------------------------------------------------

def previous_success(record: dict, target: dict) -> bool:
    """Éxito del nivel 1 sin el chequeo de nombres sin definir (antes del 2026-11-15), reconstruido de los chequeos que
    guarda el detalle por separado."""
    if target["expected"] != REPLACED:
        return target["success"]
    return (target["decision_ok"] and record["applicable"] and record["parses"] and target["target_gone"]
            and target["package_ok"] and not record["hallucinated_imports"])


def finite_rows(rows: list[dict]) -> list[tuple[float, dict]]:
    """Filas del barrido con umbral finito, en orden (sin "-inf" ni "inf", que no tienen lugar en el eje)."""
    return [(float(r["threshold"]), r) for r in rows if math.isfinite(float(r["threshold"]))]


def negative_rejection_curve(ts: list[Target]) -> list[dict]:
    """Por umbral finito, qué fracción de los negativos se abstiene por el umbral (no llega al LLM), cuál se abstiene
    el LLM (llega y el script queda igual) y cuál cambia el código (cambio indebido). Negative Rejection = las dos
    primeras."""
    negatives = [t for t in ts if not t.positive]
    out = []
    for th in (c for c in candidates(ts) if math.isfinite(c)):
        by_threshold = sum(t.score is None or t.score < th for t in negatives)
        by_llm = sum(not (t.score is None or t.score < th) and not t.changed for t in negatives)
        out.append({"threshold": th, "by_threshold": by_threshold / len(negatives), "by_llm": by_llm / len(negatives),
                    "negatives": len(negatives)})
    return out


def relevance_by_success(records: list[dict], judge: list[dict]) -> dict:
    """AR (a) y (b) promedio de los scripts juzgados, separados por el éxito del nivel 1 del ítem (con todo pasando al
    LLM): una métrica que mide la tarea tiene que dar más en los que pasan que en los que fallan."""
    success = {r["id"]: r["success"] for r in records}
    out = {}
    for ok in (True, False):
        rows = [j for j in judge if success.get(j["id"]) is ok]
        out[ok] = {"n": len(rows), **{m: sum(j[m]["score"] for j in rows) / len(rows) if rows else None
                                      for m in ("a", "b")}}
    return out


def _harm(r: dict) -> int:
    return r[BAD] + r[WRONG_CHANGE]


def _fmt_threshold(t: float) -> str:
    return f"{t:.2f}"


# --- Figuras --------------------------------------------------------------------------------------------------

def fig_tradeoff(analyses: dict[str, dict], out_dir: Path) -> Path:
    """Buenas vs. malas + cambios indebidos, un punto por umbral. Arriba de la recta y = 2x el puntaje con λ = 2 es
    positivo (conviene corregir)."""
    fig, ((ax,),) = _canvas(7, 4.8, left=0.9, right=0.5, top=0.25, bottom=1.05)
    xmax = max(_harm(r) for a in analyses.values() for r in a["sweep"]) * 1.2
    ymax = max(r[GOOD] for a in analyses.values() for r in a["sweep"]) * 1.15
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.plot([0, ymax / 2], [0, ymax], color=BASELINE, linewidth=LINE, linestyle="--", zorder=1)
    ax.text(ymax / 2 * 0.97, ymax * 0.97, "puntaje = 0 con λ = 2\n(arriba conviene corregir)", ha="right", va="top",
            fontsize=8.5, color=MUTED)
    for v, a in analyses.items():
        xs, ys = [_harm(r) for r in a["sweep"]], [r[GOOD] for r in a["sweep"]]
        ax.plot(xs, ys, color=COLOR[v], linewidth=LINE, zorder=2)
        ch, rule = a["chosen"], a["comparison"]["retrieval"]["row"]
        _dot(ax, _harm(ch), ch[GOOD], COLOR[v])
        ax.annotate(f"elegido ({_fmt_threshold(ch['threshold'])})", (_harm(ch), ch[GOOD]),
                    xytext=(_harm(ch) + xmax * 0.14, ch[GOOD] + (ymax * 0.06 if v == "rerank" else -ymax * 0.03)),
                    fontsize=8.5, color=INK2, va="center",
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8, "shrinkA": 2, "shrinkB": 5})
        ax.plot([_harm(rule)], [rule[GOOD]], marker="o", markersize=7, markerfacecolor="none",
                markeredgecolor=COLOR[v], markeredgewidth=1.5, linestyle="none", zorder=4)
        if v == "base":  # abajo a la derecha: arriba pasa la curva del reranker
            ax.text(xs[0] + xmax * 0.015, ys[0] - ymax * 0.035, f"{NAME[v]}: todo al LLM", fontsize=8.5, color=INK2)
        else:
            ax.text(xs[0], ys[0] + ymax * 0.03, f"{NAME[v]}: todo al LLM", fontsize=8.5, color=INK2, ha="right")
    _frame(ax, grid="y")
    ax.set_xlabel("Correcciones malas + cambios indebidos", fontsize=10)
    ax.set_ylabel("Correcciones buenas", fontsize=10)
    handles = [Line2D([], [], color=COLOR[v], linewidth=LINE) for v in analyses]
    handles += [Line2D([], [], marker="o", markersize=7, color=INK2, markeredgecolor="white", linestyle="none"),
                Line2D([], [], marker="o", markersize=7, markerfacecolor="none", markeredgecolor=INK2,
                       linestyle="none")]
    _legend(fig, handles, [NAME[v] for v in analyses] + ["umbral elegido (λ = 2)", "regla de retrieval (95%)"],
            ncol=4)
    return _save(fig, out_dir, "1_compromiso_buenas_malas")


def fig_utility(analyses: dict[str, dict], out_dir: Path) -> Path:
    """Puntaje vs. umbral con λ = 1, 2, 3, un panel por versión (los umbrales están en escalas distintas)."""
    fig, axes = _canvas(9, 3.9, cols=len(analyses), left=0.8, right=0.3, top=0.45, bottom=1.0, wspace=0.28)
    for ax, (v, a) in zip(axes[0], analyses.items()):
        rows = finite_rows(a["sweep"])
        ts = [t for t, _ in rows]
        for lam, style in LAMBDA_STYLE.items():
            ax.plot(ts, [r["utility"][str(lam)] for _, r in rows], color=COLOR[v], linestyle=style,
                    linewidth=LINE, drawstyle="steps-post")
        ax.axhline(0, color=BASELINE, linewidth=1)
        ch = a["chosen"]
        _dot(ax, ch["threshold"], ch["utility"]["2.0"], COLOR[v])
        ax.annotate(f"elegido: {_fmt_threshold(ch['threshold'])}", (ch["threshold"], ch["utility"]["2.0"]),
                    xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
        ax.set_title(NAME[v], loc="left", fontsize=10, color=INK2)
        _frame(ax, grid="y")
        ax.set_xlabel(THRESHOLD_LABEL[v], fontsize=9.5)
    axes[0][0].set_ylabel("Puntaje = buenas − λ × (malas + indebidos)", fontsize=9.5)
    _legend(fig, [Line2D([], [], color=INK2, linestyle=s, linewidth=LINE) for s in LAMBDA_STYLE.values()],
            [f"λ = {int(lam)}" for lam in LAMBDA_STYLE], ncol=3)
    return _save(fig, out_dir, "2_puntaje_vs_umbral")


def fig_level1(counts: dict[str, dict[str, dict]], out_dir: Path) -> Path:
    """Buenas, malas y precisión con todo pasando al LLM: nivel 1 anterior (gris) vs. corregido (color de la versión).
    `counts[versión]["anterior" | "corregido"]` = fila de `threshold.counts`."""
    panels = [("Correcciones buenas", lambda r: r[GOOD], None), ("Correcciones malas", lambda r: r[BAD], None),
              ("Precisión", lambda r: r["precision"], 1.0)]
    fig, axes = _canvas(9.5, 2.6, cols=3, left=1.5, right=0.35, top=0.35, bottom=1.0, wspace=0.18)
    versions = list(counts)
    h, gap = 0.32, 0.04
    for col, (ax, (label, get, limit)) in enumerate(zip(axes[0], panels)):
        top = limit or max(get(c[k]) for c in counts.values() for k in ("anterior", "corregido")) * 1.25
        ax.set_xlim(0, top)
        ax.set_ylim(len(versions) - 0.5, -0.5)
        ax.set_yticks(range(len(versions)))
        ax.set_yticklabels([NAME[v] for v in versions] if col == 0 else [""] * len(versions), color=INK2)
        for i, v in enumerate(versions):
            for j, (key, color) in enumerate((("anterior", GRAY), ("corregido", COLOR[v]))):
                y, value = i + (j - 0.5) * (h + gap), get(counts[v][key])
                _bar(ax, y, value, h, color)
                ax.text(value + top * 0.02, y, f"{value:.2f}" if limit else str(value), va="center", fontsize=9,
                        color=INK)
        ax.set_title(label, loc="left", fontsize=10, color=INK2)
        _frame(ax)
    _legend(fig, [Patch(color=GRAY)] + [Patch(color=COLOR[v]) for v in versions],
            ["Nivel 1 anterior"] + [f"{NAME[v]}, nivel 1 corregido" for v in versions], ncol=3)
    return _save(fig, out_dir, "3_nivel1_antes_despues")


def _agreement_rows(agreement: dict) -> list[tuple[float, str, str, dict]]:
    """(y, etiqueta, color, acuerdo) para gpt-4o-mini contra la revisión y contra Opus, en total y por versión."""
    out, y = [], 0.0
    for pair, against in (("gpt-4o-mini vs humano", "revisión manual"), ("gpt-4o-mini vs opus", "Opus 5.5")):
        for group, color in (("total", INK2), ("base", COLOR["base"]), ("rerank", COLOR["rerank"])):
            g = agreement["preservation"][pair][group]
            label = f"vs. {against} · " + ("total" if group == "total" else NAME[group].split()[0].lower())
            out.append((y, f"{label} (n = {g['n']})", color, g))
            y += 1
        y += 0.6
    return out


def fig_agreement(agreement: dict, out_dir: Path) -> Path:
    """Acuerdo exacto y kappa de Cohen de gpt-4o-mini (AR (b)) con IC 95%. "Opus vs. humano" no se grafica: la
    planilla no es independiente de Claude (ver review_agreement.md)."""
    rows = _agreement_rows(agreement)
    fig, axes = _canvas(9, 0.42 * len(rows) + 1.6, cols=2, left=2.9, right=0.4, top=0.35, bottom=0.6, wspace=0.12)
    panels = (("Acuerdo exacto", "exact", (0, 1)), ("Kappa de Cohen", "kappa", (-0.2, 1)))
    for col, (ax, (label, key, (lo, hi))) in enumerate(zip(axes[0], panels)):
        ax.set_xlim(lo, hi)
        ax.set_ylim(rows[-1][0] + 0.6, -0.6)
        ax.set_yticks([y for y, *_ in rows])
        ax.set_yticklabels([lab for _, lab, *_ in rows] if col == 0 else [""] * len(rows), color=INK2, fontsize=9)
        if lo < 0:
            ax.axvline(0, color=BASELINE, linewidth=1)
        for y, _, color, g in rows:
            if g[key] is None:
                continue
            if g[f"{key}_ci"]:
                _whisker(ax, y, *g[f"{key}_ci"])
            _dot(ax, g[key], y, color)
            ax.text((g[f"{key}_ci"] or [0, g[key]])[1] + 0.03, y, f"{g[key]:.2f}", fontsize=8.5, color=INK,
                    va="center")
        ax.set_title(label, loc="left", fontsize=10, color=INK2)
        _frame(ax)
    return _save(fig, out_dir, "4_acuerdo_juez")


def fig_mean_score(agreement: dict, out_dir: Path) -> Path:
    """Puntaje medio de preservación de cada evaluador (sí = 1, parcial = 0,5, no = 0), total y por versión."""
    raters = (("humano", "Revisión manual"), ("gpt-4o-mini", "gpt-4o-mini (juez)"), ("opus", "Opus 5.5"))
    groups = (("total", INK2, "Total"), ("base", COLOR["base"], NAME["base"]), ("rerank", COLOR["rerank"],
                                                                                   NAME["rerank"]))
    fig, ((ax,),) = _canvas(7, 3.4, left=1.8, right=0.5, top=0.2, bottom=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(len(raters) - 0.5, -0.5)
    ax.set_yticks(range(len(raters)))
    ax.set_yticklabels([label for _, label in raters], color=INK2)
    h, gap = 0.22, 0.03
    for i, (key, _) in enumerate(raters):
        for j, (group, color, _) in enumerate(groups):
            y, value = i + (j - 1) * (h + gap), agreement["mean_score"][key][group]
            _bar(ax, y, value, h, color)
            ax.text(value + 0.015, y, f"{value:.2f}", va="center", fontsize=8.5, color=INK)
    _frame(ax)
    ax.set_xlabel("Puntaje medio de preservación de la tarea (sí = 1, parcial = 0,5, no = 0)", fontsize=9.5)
    _legend(fig, [Patch(color=c) for _, c, _ in groups], [label for *_, label in groups], ncol=3)
    return _save(fig, out_dir, "5_puntaje_medio_evaluadores")


def fig_metrics_vs_threshold(analyses: dict[str, dict], out_dir: Path) -> Path:
    """Faithfulness, AR (b) y llamadas al LLM vs. umbral: una fila por versión (umbrales en escalas distintas).
    Faithfulness y AR solo donde pasan al menos MIN_N scripts (con menos, el promedio no dice nada)."""
    metrics = (("Faithfulness", "faithfulness", "faithfulness_n", (0, 1.05)),
               ("Answer Relevance (b)", "relevance_b", "relevance_b_n", (0, 1.05)),
               ("Llamadas al LLM", "llm_calls", None, None))
    fig, axes = _canvas(10, 5.6, cols=3, rows=len(analyses), left=0.75, right=0.3, top=0.4, bottom=0.65,
                        wspace=0.28, hspace=0.6)
    for row, (v, a) in enumerate(analyses.items()):
        rows, ch = finite_rows(a["sweep"]), a["chosen"]["threshold"]
        span = (rows[0][0], max(rows[-1][0], ch))
        for ax, (label, key, n_key, ylim) in zip(axes[row], metrics):
            ax.set_xlim(span[0] - (span[1] - span[0]) * 0.03, span[1] + (span[1] - span[0]) * 0.03)
            pts = [(t, r[key]) for t, r in rows if r[key] is not None and (n_key is None or r[n_key] >= MIN_N)]
            ax.plot([t for t, _ in pts], [y for _, y in pts], color=COLOR[v], linewidth=LINE, drawstyle="steps-post")
            ax.axvline(ch, color=MUTED, linewidth=1, linestyle="--")
            ax.set_ylim(*(ylim or (0, max(y for _, y in pts) * 1.1)))
            ax.set_title(f"{NAME[v]} · {label}", loc="left", fontsize=9.5, color=INK2)
            ax.set_xlabel(THRESHOLD_LABEL[v], fontsize=8.5)
            _frame(ax, grid="y")
        axes[row][0].text(ch, axes[row][0].get_ylim()[1] * 0.04, " elegido", fontsize=8, color=MUTED)
    return _save(fig, out_dir, "6_metricas_vs_umbral")


NOISE_GROUPS = ((CLEAN, "Contexto limpio"), (NOISY_FIRST, "Ruido, aceptable primero"),
                (NOISY_LATER, "Ruido, aceptable después"), (NO_ANSWER, "Sin paquete aceptable"))


def fig_rejection_noise_relevance(extra: dict[str, dict], out_dir: Path) -> Path:
    """Negative Rejection vs. umbral (por origen de la abstención), Noise Robustness y AR (a) vs. (b), con todo
    pasando al LLM en los dos últimos. `extra[versión]` = {"negatives": negative_rejection_curve,
    "noise": noise.summarize, "relevance": relevance_by_success}."""
    versions = list(extra)
    fig, axes = _canvas(10, 7.0, cols=2, rows=2, left=1.95, right=0.4, top=0.4, bottom=0.85, wspace=0.55,
                        hspace=0.45)
    for ax, v in zip(axes[0], versions):  # Negative Rejection, un panel por versión (umbrales en escalas distintas)
        rows = extra[v]["negatives"]
        ts = [r["threshold"] for r in rows]
        llm = [r["by_llm"] for r in rows]
        total = [r["by_llm"] + r["by_threshold"] for r in rows]
        ax.fill_between(ts, 0, llm, step="post", color=COLOR[v], alpha=0.35, linewidth=0)
        ax.fill_between(ts, llm, total, step="post", color=GRAY, alpha=0.3, linewidth=0)
        ax.plot(ts, total, color=COLOR[v], linewidth=LINE, drawstyle="steps-post")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{NAME[v]} · Negative Rejection (n = {rows[0]['negatives']})", loc="left", fontsize=9.5,
                     color=INK2)
        ax.set_xlabel(THRESHOLD_LABEL[v], fontsize=8.5)
        _frame(ax, grid="y")
    axes[0][0].set_ylabel("Fracción de negativos", fontsize=9)

    h, gap = 0.34, 0.04
    ax = axes[1][0]  # Noise Robustness: éxito del nivel 1 por tipo de contexto
    ax.set_xlim(0, 1)
    ax.set_ylim(len(NOISE_GROUPS) - 0.5, -0.5)
    ax.set_yticks(range(len(NOISE_GROUPS)))
    ax.set_yticklabels([label for _, label in NOISE_GROUPS], color=INK2, fontsize=9)
    for i, (g, _) in enumerate(NOISE_GROUPS):
        for j, v in enumerate(versions):
            y, cell = i + (j - 0.5) * (h + gap), extra[v]["noise"][g]
            value = cell["success"] or 0
            _bar(ax, y, value, h, COLOR[v])
            ax.text(value + 0.02, y, f"{value:.2f} (n = {cell['targets']})", va="center", fontsize=8, color=INK)
    robustness = ", ".join(f"{NAME[v].split()[0].lower()} {extra[v]['noise']['noise_robustness']:.2f}"
                           for v in versions if extra[v]["noise"]["noise_robustness"] is not None)
    ax.set_title(f"Noise Robustness (ruido / limpio: {robustness})", loc="left", fontsize=9.5, color=INK2)
    ax.set_xlabel("Éxito del nivel 1 (todo al LLM)", fontsize=8.5)
    _frame(ax)

    ax = axes[1][1]  # Answer Relevance (a) vs. (b) según el éxito del nivel 1
    cats = [("a", True, "AR (a) · pasan el nivel 1"), ("a", False, "AR (a) · fallan"),
            ("b", True, "AR (b) · pasan el nivel 1"), ("b", False, "AR (b) · fallan")]
    ax.set_xlim(0, 1)
    ax.set_ylim(len(cats) - 0.5, -0.5)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([label for *_, label in cats], color=INK2, fontsize=9)
    for i, (m, ok, _) in enumerate(cats):
        for j, v in enumerate(versions):
            y, cell = i + (j - 0.5) * (h + gap), extra[v]["relevance"][ok]
            _bar(ax, y, cell[m], h, COLOR[v])
            ax.text(cell[m] + 0.02, y, f"{cell[m]:.2f} (n = {cell['n']})", va="center", fontsize=8, color=INK)
    ax.set_title("Answer Relevance: (a) técnica de clase vs. (b) juez", loc="left", fontsize=9.5, color=INK2)
    ax.set_xlabel("Promedio (todo al LLM)", fontsize=8.5)
    _frame(ax)
    _legend(fig, [Patch(color=COLOR[v]) for v in versions]
            + [Patch(color=COLOR[versions[0]], alpha=0.35), Patch(color=GRAY, alpha=0.3)],
            [NAME[v] for v in versions] + ["se abstiene el LLM (tono claro)", "se abstiene por el umbral"], ncol=4)
    return _save(fig, out_dir, "7_rechazo_ruido_relevancia")


def generation_figures(analyses: dict[str, dict], level1: dict[str, dict[str, dict]], agreement: dict,
                       extra: dict[str, dict], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [fig_tradeoff(analyses, out_dir), fig_utility(analyses, out_dir), fig_level1(level1, out_dir),
            fig_agreement(agreement, out_dir), fig_mean_score(agreement, out_dir),
            fig_metrics_vs_threshold(analyses, out_dir), fig_rejection_noise_relevance(extra, out_dir)]
