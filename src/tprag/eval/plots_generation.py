"""Gráficos del estudio del generador (metricas.md §3-§4). Los arma `src/scripts/plots_generation.py` a partir de los
análisis de las corridas permisivas de dev (`generation_<corrida>_analysis.json`), del detalle de cada corrida (para
el nivel 1 anterior), del acuerdo de la revisión manual (`review_agreement.json`) y de la comparación de los dos
jueces sobre todos los scripts juzgados (`judge_vs_opus.json`).

Misma convención que `plots.py` (skill dataviz): sin título (va en la diapositiva), fondo claro, PNG. Color = versión
del sistema: azul = base, naranja = reranker k1-rr30 (ranuras 1 y 2 de la paleta de referencia); gris = referencia
(nivel 1 anterior, totales). λ se distingue por el tipo de línea, no por otro color.
"""

import math
from pathlib import Path

from matplotlib.legend_handler import HandlerTuple
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


def _chosen(a: dict) -> dict:
    """Fila del umbral elegido en el análisis: la regla de retrieval (el umbral más alto que deja pasar el 95% de los
    positivos de dev, calibrado antes del estudio del generador)."""
    return a["chosen"]


# --- Figuras --------------------------------------------------------------------------------------------------

def fig_tradeoff(analyses: dict[str, dict], out_dir: Path) -> Path:
    """Buenas vs. malas + cambios indebidos, un punto por umbral; el punto marca el umbral elegido (regla de retrieval).
    Sin la utilidad (la recta y = 2x y el umbral de máxima utilidad con λ = 2): va en otra diapositiva, con el gráfico 8."""
    fig, ((ax,),) = _canvas(7, 4.8, left=0.9, right=0.5, top=0.25, bottom=1.05)
    xmax = max(_harm(r) for a in analyses.values() for r in a["sweep"]) * 1.2
    ymax = max(r[GOOD] for a in analyses.values() for r in a["sweep"]) * 1.15
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    for v, a in analyses.items():
        xs, ys = [_harm(r) for r in a["sweep"]], [r[GOOD] for r in a["sweep"]]
        ax.plot(xs, ys, color=COLOR[v], linewidth=LINE, zorder=2)
        ch = _chosen(a)
        _dot(ax, _harm(ch), ch[GOOD], COLOR[v])
        # base: abajo a la izquierda; reranker: abajo a la derecha, entre su curva y la etiqueta "todo al LLM" de base
        xytext = ((_harm(ch) - xmax * 0.14, ch[GOOD] - ymax * 0.1) if v == "base" else
                  (_harm(ch) + xmax * 0.04, ch[GOOD] - ymax * 0.07))
        ax.annotate(f"elegido ({_fmt_threshold(ch['threshold'])})", (_harm(ch), ch[GOOD]), xytext=xytext,
                    fontsize=8.5, color=INK2, va="center", ha="right" if v == "base" else "left",
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8, "shrinkA": 2, "shrinkB": 5})
        if v == "base":  # abajo a la derecha: arriba pasa la curva del reranker
            ax.text(xs[0] + xmax * 0.015, ys[0] - ymax * 0.035, f"{NAME[v]}: todo al LLM", fontsize=8.5, color=INK2)
        else:
            ax.text(xs[0], ys[0] + ymax * 0.03, f"{NAME[v]}: todo al LLM", fontsize=8.5, color=INK2, ha="right")
    _frame(ax, grid="y")
    ax.set_xlabel("Correcciones malas + cambios indebidos", fontsize=10)
    ax.set_ylabel("Correcciones buenas", fontsize=10)
    handles = [Line2D([], [], color=COLOR[v], linewidth=LINE) for v in analyses]
    handles += [Line2D([], [], marker="o", markersize=7, color=INK2, markeredgecolor="white", linestyle="none")]
    _legend(fig, handles, [NAME[v] for v in analyses] + ["umbral elegido (regla de retrieval 95%)"], ncol=3,
            center=True)
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
        ch = _chosen(a)
        _dot(ax, ch["threshold"], ch["utility"]["2.0"], COLOR[v])
        ax.annotate(f"elegido: {_fmt_threshold(ch['threshold'])}", (ch["threshold"], ch["utility"]["2.0"]),
                    xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8.5, color=INK2)
        ax.set_title(NAME[v], loc="left", fontsize=10, color=INK2)
        _frame(ax, grid="y")
        ax.set_xlabel(THRESHOLD_LABEL[v], fontsize=9.5)
    axes[0][0].set_ylabel("Utilidad = buenas − λ × (malas + indebidos)", fontsize=9.5)
    _legend(fig, [Line2D([], [], color=INK2, linestyle=s, linewidth=LINE) for s in LAMBDA_STYLE.values()],
            [f"λ = {int(lam)}" for lam in LAMBDA_STYLE], ncol=3, center=True)
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
    # las filas ya dicen la versión: la leyenda solo distingue gris (anterior) de color (corregido, uno por versión)
    _legend(fig, [Patch(color=GRAY), tuple(Patch(color=COLOR[v]) for v in versions)],
            ["Éxito anterior", "Éxito corregido"], ncol=2, center=True,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)}, handlelength=2.4)
    return _save(fig, out_dir, "3_nivel1_antes_despues")


PAIRS = (("gpt-4o-mini vs humano", "gpt-4o-mini vs. humano", "4a_acuerdo_gpt_vs_humano"),
         ("gpt-4o-mini vs opus", "gpt-4o-mini vs. Opus 5.5", "4b_acuerdo_gpt_vs_opus"),
         ("opus vs humano", "Opus 5.5 vs. humano", "4c_acuerdo_opus_vs_humano"))
GROUPS = (("total", INK2, "Total"), ("base", COLOR["base"], NAME["base"]), ("rerank", COLOR["rerank"], NAME["rerank"]))


def fig_agreement_pair(agreement: dict, pair: str, pair_label: str, name: str, out_dir: Path) -> Path:
    """Acuerdo exacto y kappa de Cohen de un par de evaluadores de la preservación de la tarea (AR (b)), en total y
    por versión, con IC 95%. Las tres comparaciones usan las mismas escalas, para poder ponerlas una al lado de otra."""
    rows = [(y, f"{label} (n = {agreement['preservation'][pair][g]['n']})", color, agreement["preservation"][pair][g])
            for y, (g, color, label) in enumerate(GROUPS)]
    fig, axes = _canvas(9, 2.4, cols=2, left=2.2, right=0.3, top=0.4, bottom=0.55, wspace=0.12)
    panels = (("Acuerdo exacto", "exact", (0, 1)), ("Kappa de Cohen", "kappa", (-0.2, 1)))
    for col, (ax, (label, key, (lo, hi))) in enumerate(zip(axes[0], panels)):
        ax.set_xlim(lo, hi + 0.14)  # lugar para el valor a la derecha de un IC que llega a 1
        ax.set_xticks([x / 10 for x in range(int(lo * 10), 11, 2)])
        ax.set_ylim(len(rows) - 0.5, -0.5)
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
        ax.set_title(f"{pair_label} · {label.lower()}" if col == 0 else label, loc="left", fontsize=10, color=INK2)
        _frame(ax)
    return _save(fig, out_dir, name)


def fig_mean_score(judges: dict, out_dir: Path) -> Path:
    """Preservación media de la tarea según cada juez (sí = 1, parcial = 0,5, no = 0), total y por versión: gpt-4o-mini
    y Opus 5.5 con el mismo prompt sobre todos los scripts juzgados (`review_generation.py judge-vs-opus`). La revisión
    manual no va: está deprecada (sus valores siguen en `review_agreement.json`)."""
    raters = (("gpt-4o-mini", "gpt-4o-mini (juez)"), ("opus", "Opus 5.5"))
    groups = (("total", INK2, "Total"), ("base", COLOR["base"], NAME["base"]), ("rerank", COLOR["rerank"],
                                                                                   NAME["rerank"]))
    fig, ((ax,),) = _canvas(7, 2.7, left=1.8, right=0.5, top=0.2, bottom=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(len(raters) - 0.5, -0.5)
    ax.set_yticks(range(len(raters)))
    ax.set_yticklabels([label for _, label in raters], color=INK2)
    _frame(ax)
    ax.set_xlabel("Preservación media de la tarea (sí = 1, parcial = 0,5, no = 0)", fontsize=9.5)
    # centrar el bloque etiquetas + gráfico en la imagen (las etiquetas del eje y lo corren a la derecha); antes de
    # dibujar las barras, que se ubican en píxeles
    w = fig.get_size_inches()[0]
    box = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    pos = ax.get_position()
    ax.set_position([pos.x0 + (w - box.x1 - box.x0) / 2 / w, pos.y0, pos.width, pos.height])
    h, gap = 0.22, 0.03
    for i, (key, _) in enumerate(raters):
        for j, (group, color, _) in enumerate(groups):
            y, value = i + (j - 1) * (h + gap), judges["groups"][group]["mean"][key]
            _bar(ax, y, value, h, color)
            ax.text(value + 0.015, y, f"{value:.2f}", va="center", fontsize=8.5, color=INK)
    _legend(fig, [Patch(color=c) for _, c, _ in groups],
            [f"{label} (n = {judges['groups'][g]['n']})" for g, _, label in groups], ncol=3, center=True)
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
        rows, ch = finite_rows(a["sweep"]), _chosen(a)["threshold"]
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


def _threshold_panels(analyses: dict[str, dict], ylabel: str):
    """Figura con un panel por versión (umbrales en escalas distintas), con el mismo rango de x en 8a y 8b y la línea
    vertical del umbral elegido. Devuelve la figura y, por versión, (eje, filas finitas del barrido, umbral elegido)."""
    fig, axes = _canvas(10, 3.6, cols=len(analyses), left=0.8, right=0.3, top=0.4, bottom=1.05, wspace=0.25)
    panels = {}
    for ax, (v, a) in zip(axes[0], analyses.items()):
        rows, ch = finite_rows(a["sweep"]), _chosen(a)["threshold"]
        span = (rows[0][0], max(rows[-1][0], ch))
        ax.set_xlim(span[0] - (span[1] - span[0]) * 0.03, span[1] + (span[1] - span[0]) * 0.03)
        ax.axvline(ch, color=MUTED, linewidth=1, linestyle="--")
        ax.set_title(NAME[v], loc="left", fontsize=9.5, color=INK2)
        ax.set_xlabel(THRESHOLD_LABEL[v], fontsize=8.5)
        _frame(ax, grid="y")
        panels[v] = (ax, rows, ch)
    axes[0][0].set_ylabel(ylabel, fontsize=9.5)
    return fig, panels


def fig_precision_vs_threshold(analyses: dict[str, dict], out_dir: Path) -> Path:
    """8a: precisión vs. umbral, un panel por versión. Solo donde hay al menos MIN_N correcciones (con menos, un solo
    script la mueve de 0,5 a 1). Referencia: la precisión de mandar todo al LLM (fila de umbral más bajo)."""
    fig, panels = _threshold_panels(analyses, "Precisión")
    for v, (ax, rows, ch) in panels.items():
        pts = [(t, r["precision"]) for t, r in rows if r[GOOD] + _harm(r) >= MIN_N]
        ax.plot([t for t, _ in pts], [p for _, p in pts], color=COLOR[v], linewidth=LINE, drawstyle="steps-post")
        ax.axhline(analyses[v]["sweep"][0]["precision"], color=GRAY, linewidth=1, linestyle=":")
        ax.set_ylim(0, 1.05)
        ax.text(ch, 0.04, " elegido", fontsize=8, color=MUTED)
    _legend(fig, [Line2D([], [], color=GRAY, linewidth=1, linestyle=":")], ["precisión con todo al LLM"], ncol=1,
            center=True)
    return _save(fig, out_dir, "8a_precision_vs_umbral")


def fig_corrections_vs_threshold(analyses: dict[str, dict], out_dir: Path) -> Path:
    """8b: correcciones buenas y malas + indebidos vs. umbral (por import a corregir), un panel por versión."""
    fig, panels = _threshold_panels(analyses, "Correcciones (imports)")
    for v, (ax, rows, ch) in panels.items():
        ts = [t for t, _ in rows]
        ax.plot(ts, [r[GOOD] for _, r in rows], color=COLOR[v], linewidth=LINE, drawstyle="steps-post")
        ax.plot(ts, [_harm(r) for _, r in rows], color=COLOR[v], linewidth=LINE, linestyle="--", drawstyle="steps-post")
        ax.set_ylim(0, None)
        ax.text(ch, ax.get_ylim()[1] * 0.04, " elegido", fontsize=8, color=MUTED)
    _legend(fig, [Line2D([], [], color=INK2, linewidth=LINE), Line2D([], [], color=INK2, linewidth=LINE, linestyle="--")],
            ["buenas", "malas + indebidos"], ncol=2, center=True)
    return _save(fig, out_dir, "8b_correcciones_vs_umbral")


def _comma(x: float) -> str:
    return f"{x:.2f}".replace("-", "−").replace(".", ",")


def _counts(r: dict) -> str:
    wrong = f" / {r[WRONG_CHANGE]} indebido{'s' if r[WRONG_CHANGE] > 1 else ''}" if r[WRONG_CHANGE] else ""
    return f"{r[GOOD]} buenas / {r[BAD]} malas{wrong}"


def fig_threshold_rerank(a: dict, out_dir: Path) -> Path:
    """9: solo el reranker (el sistema final). Izquierda, correcciones buenas y malas + indebidos vs. umbral; derecha,
    precisión vs. umbral (con al menos MIN_N correcciones). Marca el umbral elegido (regla de retrieval) y el que pide
    precisión ≥ 0,8 (no elegido: casi no corrige nada)."""
    fig, axes = _canvas(10, 3.6, cols=2, left=0.8, right=0.3, top=0.4, bottom=1.05, wspace=0.25)
    rows, ch = finite_rows(a["sweep"]), _chosen(a)
    strict = a["comparison"]["precision"]["row"]
    ts = [t for t, _ in rows]
    span = (ts[0], max(ts[-1], ch["threshold"]))
    pad = (span[1] - span[0]) * 0.03
    left, right = axes[0]
    left.plot(ts, [r[GOOD] for _, r in rows], color=ORANGE, linewidth=LINE, drawstyle="steps-post")
    left.plot(ts, [_harm(r) for _, r in rows], color=ORANGE, linewidth=LINE, linestyle="--", drawstyle="steps-post")
    left.set_ylim(0, None)
    left.set_ylabel("Correcciones (imports)", fontsize=9.5)
    pts = [(t, r["precision"]) for t, r in rows if r[GOOD] + _harm(r) >= MIN_N]
    right.plot([t for t, _ in pts], [p for _, p in pts], color=ORANGE, linewidth=LINE, drawstyle="steps-post")
    right.set_ylim(0, 1.05)
    right.set_ylabel("Precisión", fontsize=9.5)
    for ax in (left, right):
        ax.set_xlim(span[0] - pad, span[1] + pad)
        ax.axvline(ch["threshold"], color=MUTED, linewidth=1, linestyle="--")
        ax.axvline(strict["threshold"], color=MUTED, linewidth=1, linestyle="-.")
        ax.set_xlabel("Umbral (score máximo del reranker)", fontsize=8.5)
        _frame(ax, grid="y")
    top = left.get_ylim()[1]
    left.text(ch["threshold"], top * 0.04, f" elegido ({_comma(ch['threshold'])}):\n {_counts(ch)}", fontsize=8,
              color=INK2)
    left.text(strict["threshold"], top * 0.9, f"precisión ≥ 0,8 ({_comma(strict['threshold'])}): {_counts(strict)} ",
              fontsize=8, color=INK2, ha="right")
    right.text(ch["threshold"], 0.04, f" elegido: precisión {_comma(ch['precision'])}", fontsize=8, color=INK2)
    right.text(strict["threshold"], 0.96, "precisión ≥ 0,8 ", fontsize=8, color=INK2, ha="right")
    _legend(fig, [Line2D([], [], color=ORANGE, linewidth=LINE), Line2D([], [], color=ORANGE, linewidth=LINE,
                                                                       linestyle="--")],
            ["buenas", "malas + indebidos"], ncol=2, center=True)
    return _save(fig, out_dir, "9_umbral_reranker")


NOISE_GROUPS = ((CLEAN, "Contexto limpio"), (NOISY_FIRST, "Ruido, aceptable primero"),
                (NOISY_LATER, "Ruido, aceptable después"), (NO_ANSWER, "Sin paquete aceptable"))


def fig_negative_rejection(extra: dict[str, dict], out_dir: Path) -> Path:
    """Negative Rejection vs. umbral, un panel por versión (umbrales en escalas distintas), separada por origen de la
    abstención: el LLM (tono claro de la versión) o el umbral (gris). `extra[versión]["negatives"]` =
    negative_rejection_curve."""
    versions = list(extra)
    fig, axes = _canvas(10, 3.7, cols=2, left=0.9, right=0.4, top=0.4, bottom=1.0, wspace=0.3)
    for ax, v in zip(axes[0], versions):
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
    _legend(fig, [tuple(Patch(color=COLOR[v], alpha=0.35) for v in versions), Patch(color=GRAY, alpha=0.3)],
            ["se abstiene el LLM", "se abstiene por el umbral"], ncol=2, center=True,
            handler_map={tuple: HandlerTuple(ndivide=None, pad=0.15)}, handlelength=2.4)
    return _save(fig, out_dir, "7a_rechazo_negativos")


def fig_noise_relevance(extra: dict[str, dict], out_dir: Path) -> Path:
    """Noise Robustness (éxito por tipo de contexto) y AR (a) vs. (b) según el éxito, con todo pasando al LLM.
    `extra[versión]` = {"noise": noise.summarize, "relevance": relevance_by_success}."""
    versions = list(extra)
    fig, axes = _canvas(10, 3.9, cols=2, left=1.95, right=0.4, top=0.4, bottom=0.95, wspace=0.55)
    h, gap = 0.34, 0.04
    ax = axes[0][0]  # Noise Robustness: éxito por tipo de contexto
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
    ax.set_title("Noise Robustness", loc="left", fontsize=9.5, color=INK2)  # el cociente va en la tabla, no acá
    ax.set_xlabel("Éxito (todo al LLM)", fontsize=8.5)
    _frame(ax)

    ax = axes[0][1]  # Answer Relevance (a) vs. (b) según el éxito
    cats = [("a", True, "RAGAS · exitosos"), ("a", False, "RAGAS · no exitosos"),
            ("b", True, "LLM as a Judge · exitosos"), ("b", False, "LLM as a Judge · no exitosos")]
    ax.set_xlim(0, 1)
    ax.set_ylim(len(cats) - 0.5, -0.5)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([label for *_, label in cats], color=INK2, fontsize=9)
    for i, (m, ok, _) in enumerate(cats):
        for j, v in enumerate(versions):
            y, cell = i + (j - 0.5) * (h + gap), extra[v]["relevance"][ok]
            _bar(ax, y, cell[m], h, COLOR[v])
            ax.text(cell[m] + 0.02, y, f"{cell[m]:.2f} (n = {cell['n']})", va="center", fontsize=8, color=INK)
    ax.set_title("Answer Relevance: RAGAS vs. LLM as a Judge", loc="left", fontsize=9.5, color=INK2)
    ax.set_xlabel("Promedio (todo al LLM)", fontsize=8.5)
    _frame(ax)
    _legend(fig, [Patch(color=COLOR[v]) for v in versions], [NAME[v] for v in versions], ncol=2, center=True)
    return _save(fig, out_dir, "7b_ruido_relevancia")


def success_vs_opus(records: list[dict], answers: list[dict], run: str, threshold: float) -> dict:
    """Veredictos de Opus con rúbrica por **import** (la unidad del Éxito automático): positivos que pasan el umbral, en
    scripts que Opus juzgó (los cambiados), separados por su éxito. Opus juzga el script entero; el veredicto vale para
    todos sus imports (en la corrida sin iterar ningún script juzgado mezcla imports exitosos y no exitosos)."""
    verdict = {a["id"]: a["output"]["preserves"] for a in answers if a["run"] == run}
    out = {ok: {"si": 0, "parcial": 0, "no": 0} for ok in (True, False)}
    for r in records:
        if r["id"] not in verdict:
            continue
        for t in r["targets"]:
            if t["expected"] == REPLACED and t["retrieval_score"] is not None and t["retrieval_score"] >= threshold:
                out[t["success"]][verdict[r["id"]]] += 1
    return out


def fig_validation_opus(success: dict, faith: dict, out_dir: Path) -> Path:
    """10a: cuánto coincide Opus 5.5 con rúbrica con lo que marcó cada métrica automática (sin iterar, reranker, dev),
    separado por lo que dijo la métrica. Éxito automático por import (`success_vs_opus`): exitoso coincide con `si`,
    no exitoso con `no` (`parcial` no coincide). Faithfulness por nombre de la API (`faith_vs_opus.json`): fiel coincide
    con respaldado, no fiel con no respaldado."""
    conf = faith["confusion"]
    rows = (  # (etiqueta, coincide, total); None = separación entre métricas
        ("Éxito Automático: exitoso", success[True]["si"], sum(success[True].values())),
        ("Éxito Automático: no exitoso", success[False]["no"], sum(success[False].values())),
        None,
        ("Faithfulness: fiel", conf["fiel_respaldado"], conf["fiel_respaldado"] + conf["falso_fiel"]),
        ("Faithfulness: no fiel", conf["no_fiel_no_respaldado"], conf["falso_no_fiel"] + conf["no_fiel_no_respaldado"]),
    )
    fig, ((ax,),) = _canvas(8, 2.9, left=2.3, right=1.3, top=0.15, bottom=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ticks = [y for y, r in enumerate(rows) if r]
    ax.set_yticks(ticks)
    ax.set_yticklabels([rows[y][0] for y in ticks], color=INK2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Coincide con Opus", fontsize=9.5)
    _frame(ax)
    for y in ticks:
        _, ok, n = rows[y]
        _bar(ax, y, ok / n, 0.6, ACCENT)
        ax.text(ok / n + 0.015, y, f"{round(100 * ok / n)}%  ({ok} / {n})", va="center", fontsize=9, color=INK)
    return _save(fig, out_dir, "10a_validacion_exito_faithfulness")


def fig_validation_judge(rubric: dict, out_dir: Path) -> Path:
    """10b: veredictos de preservación de la tarea de gpt-4o-mini (el juez de LLM as a Judge, sin rúbrica) y de Opus 5.5
    con rúbrica sobre los mismos scripts (sin iterar, reranker, dev; `judge_rubric.json`). Barras agrupadas por
    veredicto: muestran que gpt contesta casi todo `parcial` donde Opus contesta `no`. El valor de cada veredicto va en
    el eje y el promedio de cada juez (LLM as a Judge), en la leyenda."""
    total = rubric["runs"]["rerank"]["judges"]["sin rúbrica"]["total"]
    raters = (("gpt-4o-mini", "gpt-4o-mini (LLM as a Judge)", GRAY), ("opus", "Opus 5.5", ACCENT))
    verdicts = (("si", "Sí (= 1)"), ("parcial", "Parcial (= 0,5)"), ("no", "No (= 0)"))
    fig, ((ax,),) = _canvas(7, 3.2, left=1.25, right=0.6, top=0.15, bottom=1.05)
    xmax = max(total["verdicts"][r][v] for r, _, _ in raters for v, _ in verdicts)
    ax.set_xlim(0, xmax * 1.12)
    ax.set_ylim(len(verdicts) - 0.5, -0.5)
    ax.set_yticks(range(len(verdicts)))
    ax.set_yticklabels([label for _, label in verdicts], color=INK2)
    ax.set_xlabel(f"Scripts (n = {total['n']})", fontsize=9.5)
    _frame(ax)
    h, gap = 0.34, 0.04
    for i, (v, _) in enumerate(verdicts):
        for j, (r, _, color) in enumerate(raters):
            y, n = i + (j - 0.5) * (h + gap), total["verdicts"][r][v]
            _bar(ax, y, n, h, color)
            ax.text(n + xmax * 0.012, y, str(n), va="center", fontsize=9, color=INK)
    _legend(fig, [Patch(color=c) for _, _, c in raters],
            [f"{label}: {total['mean'][r]:.2f}".replace("0.", "0,") for r, label, _ in raters], ncol=2, center=True)
    return _save(fig, out_dir, "10b_validacion_juez")


CAUSES = {  # causa del diagnóstico → (etiqueta, ¿del retrieval?); la correcta = una librería aceptable en el contexto
    "migra_a_medias": ("Migra a medias", False),
    "no_aplica_los_cambios": ("No aplica los cambios", False),
    "elige_mal_con_la_correcta": ("Elige mal, con la correcta en el contexto", False),
    "elige_mal_sin_la_correcta": ("Elige mal, sin la correcta en el contexto", True),
    "no_parsea": ("No parsea", False),
    "debajo_del_umbral": ("Debajo del umbral", True),
    "llm_se_abstiene_con_la_correcta": ("Se abstiene, con la correcta en el contexto", False),
    "llm_se_abstiene_sin_la_correcta": ("Se abstiene, sin la correcta en el contexto", True),
    "respuesta_cortada": ("Respuesta cortada", False),
    "abstencion_postvalidacion": ("Abstención de la post-validación", False),
    "otro": ("Otro", False),
}


def fig_diagnosis(diagnosis: dict, out_dir: Path, name: str = "11_diagnostico", xmax: int | None = None) -> Path:
    """11: por qué fallan los imports no exitosos (`diagnose_generation.py`; una causa por import), una barra por causa
    ordenadas de mayor a menor. Azul = del generador (tenía la librería correcta en el contexto o el problema es cómo
    la usó); gris = del retrieval (no llegó al LLM o no tenía la correcta). Sin las causas con 0 imports. `xmax` fija
    la escala para comparar dos diagnósticos (14 usa la de 11)."""
    rows = sorted(((n, *CAUSES[c]) for c, n in diagnosis["causes"].items() if n), key=lambda r: -r[0])
    totals = {flag: sum(n for n, _, f in rows if f is flag) for flag in (False, True)}
    color = {False: ACCENT, True: GRAY}
    fig, ((ax,),) = _canvas(8, 3.4, left=3.0, right=0.4, top=0.15, bottom=0.95)
    xmax = xmax or rows[0][0]
    ax.set_xlim(0, xmax * 1.1)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for _, label, _ in rows], color=INK2)
    ax.set_xlabel(f"Imports no exitosos (n = {sum(totals.values())})", fontsize=9.5)
    _frame(ax)
    for y, (n, _, retrieval) in enumerate(rows):
        _bar(ax, y, n, 0.62, color[retrieval])
        ax.text(n + xmax * 0.012, y, str(n), va="center", fontsize=9, color=INK)
    _legend(fig, [Patch(color=color[False]), Patch(color=color[True])],
            [f"Generador: {totals[False]}", f"Retrieval: {totals[True]}"], ncol=2, center=True)
    return _save(fig, out_dir, name)


STEP_METRICS = (("success_pos", "Éxito Automático", ACCENT), ("precision", "Precisión", ORANGE),
                ("negative_rejection", "Negative Rejection", "#1baf7a"))  # ranuras 1-3 de la paleta de referencia


def fig_steps(steps: dict[str, dict], out_dir: Path, name: str = "12_resultados_por_paso") -> Path:
    """12: Éxito automático, Precisión y Negative Rejection por paso de la iteración (sin iterar y las variantes que
    quedaron), con el umbral elegido. `steps`: nombre → `counts` (tprag.eval.threshold) de esa corrida. También para
    test (15), con otro `name`."""
    fig, ((ax,),) = _canvas(8, 3.4, left=0.7, right=0.3, top=0.2, bottom=0.95)
    w = 0.26
    ax.set_xlim(-0.6, len(steps) - 0.4)
    ax.set_ylim(0, 1.12)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(list(steps), color=INK2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "0,25", "0,5", "0,75", "1"])
    _frame(ax, grid="y")
    for i, c in enumerate(steps.values()):
        for j, (key, _, color) in enumerate(STEP_METRICS):
            x = i + (j - 1) * w
            ax.bar(x, c[key], width=w - 0.03, color=color, linewidth=0)
            ax.text(x, c[key] + 0.02, f"{c[key]:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=8.5,
                    color=INK)
    _legend(fig, [Patch(color=color) for _, _, color in STEP_METRICS], [label for _, label, _ in STEP_METRICS],
            ncol=3, center=True)
    return _save(fig, out_dir, name)


def fig_confirmed(steps: dict[str, dict], positives: int, judges: dict[str, dict], out_dir: Path,
                  name: str = "13_exito_confirmado") -> Path:
    """13: por paso, cuatro barras en el mismo eje (0-1): Éxito automático y Éxito confirmado, en imports sobre los
    positivos de dev (`steps`: nombre → `success_vs_opus`; confirmado = éxito automático en un script que Opus con
    rúbrica marca `si`), y LLM as a Judge (gpt-4o-mini) y la preservación según Opus 5.5 con rúbrica, promedios por
    script sobre los mismos scripts cambiados (`judges`: nombre → {"gpt-4o-mini": media, "opus": media}; sí = 1,
    parcial = 0,5, no = 0). La leyenda dice la unidad de cada barra."""
    series = (
        (f"Éxito Automático (imports, n = {positives})", "#2a78d6",
         lambda k: (sum(steps[k][True].values()) / positives, sum(steps[k][True].values()))),
        (f"Éxito Confirmado por Opus 5.5 (imports, n = {positives})", "#104281",
         lambda k: (steps[k][True]["si"] / positives, steps[k][True]["si"])),
        ("LLM as a Judge, gpt-4o-mini (media por script)", GRAY, lambda k: (judges[k]["gpt-4o-mini"], None)),
        ("Preservación según Opus 5.5 (media por script)", "#1baf7a", lambda k: (judges[k]["opus"], None)),
    )
    names = list(steps)
    fig, ((ax,),) = _canvas(10, 3.8, left=0.6, right=0.3, top=0.2, bottom=1.25)
    w = 0.2
    ax.set_xlim(-0.55, len(names) - 0.45)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, color=INK2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "0,25", "0,5", "0,75", "1"])
    _frame(ax, grid="y")
    for i, k in enumerate(names):
        for j, (_, color, get) in enumerate(series):
            v, n = get(k)
            x = i + (j - 1.5) * w
            ax.bar(x, v, width=w - 0.025, color=color, linewidth=0)
            label = f"{v:.2f}".replace(".", ",") + (f"\n({n})" if n is not None else "")
            ax.text(x, v + 0.015, label, ha="center", va="bottom", fontsize=8, color=INK, linespacing=1.1)
    _legend(fig, [Patch(color=c) for _, c, _ in series], [label for label, _, _ in series], ncol=2, center=True)
    return _save(fig, out_dir, name)


def fig_dev_vs_test(dev: dict, test: dict, passed: dict[str, float], out_dir: Path, variant: str = "k1-rr30") -> Path:
    """17: el retrieval con reranker en dev y en test (`rerank_full_{dev,test_iter7check}_analysis.json`), con IC 95%
    por bootstrap, más la fracción de positivos que pasan el umbral elegido en dev (`passed`: split → fracción; sin IC).
    Explica por qué el sistema cae en test: llegan menos imports al LLM y con peor contexto."""
    metrics = (("pkg_hit@1", "Package Hit@1"), ("span_hit@2000", "Span Hit@2000"), ("auroc_rr", "AUROC del reranker"))
    splits = (("dev", "Dev", GRAY), ("test", "Test", ACCENT))
    labels = [label for _, label in metrics] + ["Positivos que pasan el umbral"]
    fig, ((ax,),) = _canvas(9, 3.5, left=0.6, right=0.3, top=0.2, bottom=0.95)
    w = 0.34
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, color=INK2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "0,25", "0,5", "0,75", "1"])
    _frame(ax, grid="y")
    for j, (split, _, color) in enumerate(splits):
        m = (dev if split == "dev" else test)["variants"][variant]["metrics"]
        values = [(m[k]["value"], m[k]["ci"]) for k, _ in metrics] + [(passed[split], None)]
        for i, (v, ci) in enumerate(values):
            x = i + (j - 0.5) * w
            ax.bar(x, v, width=w - 0.03, color=color, linewidth=0)
            top = v
            if ci:
                ax.errorbar(x, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor=INK2, elinewidth=0.9, capsize=3)
                top = ci[1]
            ax.text(x, top + 0.02, f"{v:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=8.5, color=INK)
    _legend(fig, [Patch(color=c) for _, _, c in splits], [label for _, label, _ in splits], ncol=2, center=True)
    return _save(fig, out_dir, "17_retrieval_dev_vs_test")


def generation_figures(analyses: dict[str, dict], level1: dict[str, dict[str, dict]], agreement: dict,
                       judges: dict, extra: dict[str, dict], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [fig_tradeoff(analyses, out_dir), fig_utility(analyses, out_dir), fig_level1(level1, out_dir),
            *(fig_agreement_pair(agreement, pair, label, name, out_dir) for pair, label, name in PAIRS),
            fig_mean_score(judges, out_dir),
            fig_metrics_vs_threshold(analyses, out_dir), fig_precision_vs_threshold(analyses, out_dir),
            fig_corrections_vs_threshold(analyses, out_dir), fig_threshold_rerank(analyses["rerank"], out_dir),
            fig_negative_rejection(extra, out_dir),
            fig_noise_relevance(extra, out_dir)]
