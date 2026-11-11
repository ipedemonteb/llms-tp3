"""Estadísticas de los chunks de un splitter, para comparar splitters sin usar embeddings (plan de experimentos, etapa 1).

Funciones puras (sin Chroma ni modelos):
- `chunk_stats`: cuántos chunks hay y cómo se reparten sus tamaños (tokens de cl100k_base, como en el resto del proyecto).
- `cut_code_stats`: cuántos bloques de código de la documentación NO quedan enteros dentro de algún chunk. Los bloques
  de los ítems sintéticos de dev se eligieron enteros en todos los splitters (sample_synthetic.py), así que esta es la
  única vista que muestra cuánto código parte cada uno.
- `model_token_stats`: tokens que ve el embedder (su propio tokenizer) y qué parte supera su límite (se trunca).
"""

from collections import Counter, defaultdict
from collections.abc import Callable
from math import ceil
from statistics import mean

from langchain_core.documents import Document

from tprag.split.chunk import Chunk
from tprag.split.markdown import parse

PY_LANGS = ("python", "py")


def quantiles(values: list[int | float]) -> dict:
    """n, media, mediana, p90, p99 y máximo (rango más cercano)."""
    if not values:
        return {"n": 0}
    v = sorted(values)
    at = lambda p: v[max(ceil(p * len(v)) - 1, 0)]  # noqa: E731
    return {"n": len(v), "mean": round(mean(v), 1), "median": at(0.5), "p90": at(0.9), "p99": at(0.99), "max": v[-1]}


def chunk_stats(chunks: list[Chunk]) -> dict:
    return {
        "chunks": len(chunks), "tokens_total": sum(c.n_tokens for c in chunks),
        "n_tokens": quantiles([c.n_tokens for c in chunks]),
        "with_code": sum(c.n_code_blocks > 0 for c in chunks), "use_cases": sum(c.is_use_case for c in chunks),
        "by_kind": dict(sorted(Counter(c.kind for c in chunks).items())),
        "by_package": dict(sorted(Counter(c.package for c in chunks).items())),
    }


def cut_code_stats(docs: list[Document], chunks: list[Chunk]) -> dict:
    """Bloques de código que ningún chunk contiene entero, sobre `docs` y todos los chunks de esos documentos (antes
    del filtro y el tope). Con solapamiento basta que un chunk lo tenga entero; un bloque que ningún chunk cubre
    (p. ej. fuera de las secciones que usa S2) también cuenta como no entero."""
    spans = defaultdict(list)
    for c in chunks:
        spans[c.doc_id].append((c.start, c.end))
    totals = {"all": [0, 0], "python": [0, 0]}
    for d in docs:
        doc_spans = spans.get(d.metadata["id"], [])
        for b in parse(d.page_content, d.metadata["source_path"]).blocks:
            if b.kind != "code":
                continue
            intact = any(s <= b.start and b.end <= e for s, e in doc_spans)
            for key in ("all", "python") if b.lang in PY_LANGS else ("all",):
                totals[key][0] += 1
                totals[key][1] += intact
    return {key: {"blocks": n, "intact": ok, "not_intact_rate": round(1 - ok / n, 4) if n else None}
            for key, (n, ok) in totals.items()}


def model_token_stats(texts: list[str], count_tokens: Callable[[list[str]], list[int]], max_length: int) -> dict:
    """Tokens por texto con el tokenizer del embedder; `over_max` es la parte que se trunca con `max_length`."""
    lengths = count_tokens(texts)
    return {**quantiles(lengths), "max_length": max_length,
            "over_max": round(sum(n > max_length for n in lengths) / len(lengths), 4) if lengths else None}
