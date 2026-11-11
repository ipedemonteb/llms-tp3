"""Gráficos de la etapa 2 (docs/agents/plan_experimentos_indice.md §9): compara los embedders locales. No usa API, ni
embeddings, ni Chroma: lee lo que ya dejaron los scripts anteriores.

Hace falta tener antes los análisis de los dos splitters:
    uv run python src/scripts/analyze_splitters.py --vary model --splitter s3 --models e4 e3 e5 --no-plots
    uv run python src/scripts/analyze_splitters.py --vary model --splitter s2 --models e4 e3 e5 --no-plots

Uso (necesita el grupo `analysis`, matplotlib):
    uv run --group analysis python src/scripts/plot_embedders.py

Lee results/study/embedders_{s3,s2}_analysis.json, embedders_manifest.json (memoria del paso de índice) y los
reportes de results/index/. Escribe results/study/figures/embedders/*.png (se regeneran en segundos, no se versionan).

El costo se mide en segundos por texto realmente calculado (use_cases − cache_hits), no en el tiempo total del índice:
E4 se indexó con parte del caché ya llena y su tiempo total no es comparable. La memoria solo existe para las corridas
de esta etapa (E3 y E5); en E4 la figura dice "no registrado".
"""

import argparse
import json

from tprag.eval.plots_embedders import embedder_figures
from tprag.split.store import ROOT

STUDY_DIR, INDEX_DIR = ROOT / "results" / "study", ROOT / "results" / "index"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(STUDY_DIR / "figures" / "embedders"))
    args = parser.parse_args()

    results = {}
    for s in ("s3", "s2"):
        path = STUDY_DIR / f"embedders_{s}_analysis.json"
        if not path.exists():
            print(f"falta {path.relative_to(ROOT)}: correr analyze_splitters.py --vary model --splitter {s} (ver arriba)")
            return 1
        results[s] = json.loads(path.read_text(encoding="utf-8"))
    if set(results["s3"]["splitters"]) != set(results["s2"]["splitters"]):
        print("s3 y s2 no tienen los mismos modelos: no se pueden poner lado a lado")
        return 1
    manifest_path = STUDY_DIR / "embedders_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    indexes = {name: json.loads((INDEX_DIR / f"{name}.json").read_text(encoding="utf-8"))
               for r in results.values() for name in r["collections"].values()}

    try:
        from pathlib import Path  # noqa: PLC0415

        paths = embedder_figures(results, manifest, indexes, Path(args.out))
    except ImportError as e:
        print(f"falta matplotlib ({e}): usar `uv run --group analysis`")
        return 1
    for path in paths:
        print("->", path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
