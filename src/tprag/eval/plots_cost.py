"""Gráficos del análisis de costo (extra de la consigna). Los arma `src/scripts/analyze_cost.py` a partir de su
resultado (`results/study/cost_analysis.json`): no leen nada más.

Misma convención que `plots.py` (skill dataviz): sin título (va en la diapositiva), fondo claro, PNG.
- Tokens por llamada: una barra apilada por versión; la entrada en tonos de azul (de lo fijo a lo variable) y la
  salida en naranja, porque cuesta 4 veces más por token.
- Latencia del reranker: azul = Qwen3 (el elegido, con k1-rr30 marcado), gris = gte-modernbert.
- USD cada 1.000 scripts: azul = base, naranja = reranker k1-rr30 (como en `plots_generation.py`).
"""

from pathlib import Path

from matplotlib.patches import Patch

from tprag.eval.plots import ACCENT, GRAY, INK, INK2, ORANGE, SURFACE, _bar, _canvas, _frame, _legend, _rows, _save

VERSIONS = ("base", "rerank")
NAME = {"base": "Base", "rerank": "Reranker k1-rr30"}
COLOR = {"base": ACCENT, "rerank": ORANGE}
INPUT_PARTS = (("system", "Instrucciones", "#a9c8ef"), ("overhead", "Esquema y formato", "#6fa3e3"),
               ("script", "Script", "#2a78d6"), ("context", "Fragmentos recuperados", "#1a4f91"))
REGIMES = (("permisivo_95", "Umbral permisivo (pasa 95% de positivos)"), ("elegido", "Umbral elegido en dev"))


def _thousands(v: float) -> str:
    return f"{v:,.0f}".replace(",", ".")


def token_segments(result: dict) -> dict[str, list[tuple[str, float, str]]]:
    """Por versión: (nombre, tokens medios, color) de cada parte de la llamada, entrada primero y salida al final."""
    overhead = result["calibration"]["input_overhead"]
    out = {}
    for v in VERSIONS:
        g = result["generator"][v]
        parts = g["input_parts_mean"] | {"overhead": overhead}
        out[v] = [(label, parts[key], color) for key, label, color in INPUT_PARTS]
        out[v].append(("Salida", g["per_call"]["output_mean"], ORANGE))
    return out


def fig_tokens(result: dict, out_dir: Path) -> Path:
    segments = token_segments(result)
    total = max(sum(t for _, t, _ in segs) for segs in segments.values())
    fig, axes = _canvas(9.0, 2.1, left=1.9, top=0.2, bottom=0.95)
    ax = axes[0][0]
    _frame(ax)
    ax.set_xlim(0, total * 1.12)
    _rows(ax, [NAME[v] for v in VERSIONS])
    for y, v in enumerate(VERSIONS):
        x = 0.0
        for _, tokens, color in segments[v]:
            _bar(ax, y, x + tokens, 0.55, color, x0=x, radius_px=0)
            if tokens > total * 0.06:
                ax.text(x + tokens / 2, y, _thousands(tokens), ha="center", va="center", fontsize=8.5,
                        color=INK if color in ("#a9c8ef", "#6fa3e3") else SURFACE)
            x += tokens
        ax.text(x + total * 0.01, y, f"{_thousands(x)} tokens", va="center", fontsize=9, color=INK2)
    ax.set_xlabel("Tokens medios por llamada a gpt-4o-mini")
    handles = [Patch(color=c) for _, _, c in segments[VERSIONS[0]]]
    _legend(fig, handles, [n for n, _, _ in segments[VERSIONS[0]]], ncol=5)
    return _save(fig, out_dir, "1_tokens_por_llamada")


def fig_reranker_latency(result: dict, out_dir: Path, chosen: tuple[str, int] = ("k1", 30)) -> Path:
    rows = result["reranker"]
    fig, axes = _canvas(7.5, 3.2, left=0.8, top=0.3, bottom=0.95)
    ax = axes[0][0]
    _frame(ax, grid="y")
    for rid, color in (("k1", ACCENT), ("k2", GRAY)):
        pts = sorted((r["k"], r["seconds_per_query"]) for r in rows if r["reranker"] == rid)
        ax.plot([k for k, _ in pts], [s for _, s in pts], color=color, linewidth=1.5, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        for k, s in pts:
            ax.annotate(f"{s:.1f} s".replace(".", ","), (k, s), textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8.5, color=INK2)
    best = next(r for r in rows if (r["reranker"], r["k"]) == chosen)
    ax.plot([best["k"]], [best["seconds_per_query"]], marker="o", markersize=12, markerfacecolor="none",
            markeredgecolor=INK, markeredgewidth=1.2, zorder=4)
    ks = sorted({r["k"] for r in rows})
    ax.set_xticks(ks)
    ax.set_xlabel("Candidatos reordenados por consulta (K)")
    ax.set_ylabel("Segundos por consulta")
    ax.set_ylim(0, max(r["seconds_per_query"] for r in rows) * 1.18)
    labels = {r["reranker"]: r["label"] for r in rows}
    _legend(fig, [Patch(color=ACCENT), Patch(color=GRAY), Patch(facecolor="none", edgecolor=INK)],
            [labels["k1"], labels["k2"], "Elegido (k1-rr30)"], ncol=3)
    return _save(fig, out_dir, "2_latencia_reranker")


def fig_usd_per_thousand(result: dict, out_dir: Path) -> Path:
    """Un panel por régimen de umbral, con la misma escala: muestra cuánto ahorra el umbral elegido."""
    ops = result["operating_points"]
    peak = max(o["usd_per_1000_scripts"] for v in VERSIONS for o in ops[v].values())
    fig, axes = _canvas(10.0, 2.4, cols=2, left=1.9, right=0.3, top=0.45, bottom=0.75, wspace=0.75)
    for col, (regime, title) in enumerate(REGIMES):
        ax = axes[0][col]
        _frame(ax)
        ax.set_xlim(0, peak * 1.7)
        _rows(ax, [NAME[v] for v in VERSIONS], show=col == 0)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK2, pad=8)
        for y, v in enumerate(VERSIONS):
            o = ops[v][regime]
            usd = o["usd_per_1000_scripts"]
            _bar(ax, y, usd, 0.6, COLOR[v])
            ax.text(usd + peak * 0.03, y, f"{usd:.3f} USD ({o['llm_calls']}/{o['scripts']} al LLM)".replace(".", ","),
                    va="center", fontsize=9, color=INK2)
        ax.set_xlabel("USD cada 1.000 scripts con imports problemáticos" if col == 0 else "USD cada 1.000 scripts")
    return _save(fig, out_dir, "3_usd_cada_1000_scripts")


def cost_figures(result: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [fig_tokens(result, out_dir), fig_reranker_latency(result, out_dir), fig_usd_per_thousand(result, out_dir)]
