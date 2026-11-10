"""Muestra de bloques de código para el dataset sintético de dev (docs/agents/plan_experimentos_indice.md §2, dataset_eval.md
§3.6). NO USA API ni modelos: es un script con semilla fija, así la muestra es reproducible.

Uso:
    uv run python src/scripts/sample_synthetic.py            # escribe data/eval/synthetic_sample.json
    uv run python src/scripts/sample_synthetic.py --dry-run  # solo muestra el resumen

Toma bloques de Python de la doc normalizada (no de los chunks, para no favorecer a ningún splitter) de las librerías de
`data/eval/corpus_subset.json`. Un bloque entra solo si:
- es Python, de 5 a 60 líneas con código, 150 a 2500 caracteres, sin salida de consola (`>>>`);
- usa la librería de su documento (import, o el alias habitual en pandas y numpy);
- sobrevive al tope de 150 por librería y queda entero dentro de un chunk en TODOS los splitters (s0-s3, char, tok):
  si no, el índice no contiene la respuesta y ninguna configuración podría encontrarla.
Reparte `--per-library` bloques por librería, de documentos distintos cuando se puede, y el resto de `--total` entre las
librerías con más candidatos.
"""

import argparse
import json
import random
import re
from collections import defaultdict

from tprag.eval.verify import EVAL_DIR
from tprag.normalize.store import load_normalized
from tprag.split.markdown import parse
from tprag.split.select import DEFAULT_CAP, select
from tprag.split.store import PROCESSED_DIR, load_chunks

CONFIGS = ("s0-1024", "s1-1024", "s2-1024", "s3-1024", "char-1024", "tok-1024")
MODULE = {"scikit-learn": "sklearn", "scikit-image": "skimage", "pymupdf": "(?:fitz|pymupdf)",
          "python-stdlib": "(?:unittest|mock)"}
ALIAS = {"pandas": r"\bpd\.", "numpy": r"\bnp\."}  # los docs asumen el import hecho
SEED = 2026


def _covered(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(a <= start and end <= b for a, b in spans)


def candidates(subset: list[str]) -> dict[str, list[dict]]:
    docs = load_normalized(packages=subset)
    surviving = {}
    for cfg in CONFIGS:
        chunks = [c for c in load_chunks(PROCESSED_DIR / "outputs_as_comments" / cfg / "chunks.jsonl")
                  if c.package in subset]
        spans = defaultdict(list)
        for c in select(chunks, cap=DEFAULT_CAP):
            spans[c.doc_id].append((c.start, c.end))
        surviving[cfg] = spans

    out = defaultdict(list)
    for doc in docs:
        package, text = doc.metadata["package"], doc.page_content
        uses = re.compile(rf"^\s*(?:import|from)\s+{MODULE.get(package, re.escape(package.replace('-', '_')))}\b"
                          + (f"|{ALIAS[package]}" if package in ALIAS else ""), re.M)
        blocks = [b for b in parse(text).blocks if b.kind == "code"]
        for number, b in enumerate(blocks):
            if b.lang not in ("python", "py"):
                continue
            code = "\n".join(text[b.start:b.end].split("\n")[1:-1])
            lines = [line for line in code.split("\n") if line.strip()]
            if ">>>" in code or not 5 <= len(lines) <= 60 or not 150 <= len(code) <= 2500 or not uses.search(code):
                continue
            if all(_covered(surviving[cfg].get(doc.metadata["id"], []), b.start, b.end) for cfg in CONFIGS):
                out[package].append({
                    "package": package, "category": doc.metadata["category"], "doc_id": doc.metadata["id"],
                    "source_path": doc.metadata["source_path"], "block": number, "start": b.start, "end": b.end,
                    "n_lines": len(lines), "code": code})
    return out


def pick(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """n bloques, de documentos distintos mientras se pueda."""
    pool = sorted(pool, key=lambda c: (c["source_path"], c["block"]))
    rng.shuffle(pool)
    chosen, docs = [], set()
    for c in pool:
        if len(chosen) < n and c["doc_id"] not in docs:
            chosen.append(c)
            docs.add(c["doc_id"])
    for c in pool:
        if len(chosen) < n and c not in chosen:
            chosen.append(c)
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-library", type=int, default=3)
    parser.add_argument("--total", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    subset = json.loads((EVAL_DIR / "corpus_subset.json").read_text(encoding="utf-8"))["packages"]
    pools = candidates(subset)
    rng = random.Random(SEED)
    sample = []
    for package in subset:
        sample += pick(pools[package], args.per_library, rng)
    # el resto, un bloque más para cada una de las librerías con más candidatos
    for package in sorted(subset, key=lambda p: -len(pools[p]))[:max(args.total - len(sample), 0)]:
        sample += pick([c for c in pools[package] if c not in sample], 1, rng)
    per_package = {p: sum(c["package"] == p for c in sample) for p in subset}
    print(f"{len(sample)} bloques de {sum(v > 0 for v in per_package.values())} librerías; candidatos totales "
          f"{sum(map(len, pools.values()))}")
    print(per_package)
    if args.dry_run:
        return 0
    out = EVAL_DIR / "synthetic_sample.json"
    out.write_text(json.dumps({
        "seed": SEED, "per_library": args.per_library, "total": len(sample), "configs_checked": list(CONFIGS),
        "candidates_per_library": {p: len(pools[p]) for p in subset}, "blocks": sample,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
