"""Métricas de la fase 0 (splitters.md §3.10): sin embeddings ni consultas, solo mirando los chunks."""

import random
import statistics
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict

from .chunk import MIN_PROSE_CHARS, Chunk
from .markdown import ParsedDoc
from .select import DEFAULT_CAP, select
from .tokens import TokenCounter

# Límite de entrada de cada modelo de embeddings, en tokens del tokenizer usado para cortar (estudio §4.1).
# Los locales se suman cuando se instale el grupo opcional (su tokenizer cuenta distinto).
MODEL_LIMITS = {"text-embedding-3-small": 8191, "text-embedding-3-large": 8191}
HISTOGRAM_BINS = (50, 100, 200, 400, 600, 800, 1024, 2048)


def _pct(part: int, total: int) -> float:
    return round(100 * part / total, 2) if total else 0.0


def _histogram(values: list[int]) -> dict[str, int]:
    edges = (0, *HISTOGRAM_BINS)
    hist = {f"{a}-{b}": 0 for a, b in zip(edges, edges[1:])} | {f">{HISTOGRAM_BINS[-1]}": 0}
    for v in values:
        i = bisect_left(HISTOGRAM_BINS, v)
        key = f">{HISTOGRAM_BINS[-1]}" if i == len(HISTOGRAM_BINS) else f"{edges[i]}-{edges[i + 1]}"
        hist[key] += 1
    return hist


def size_stats(chunks: list[Chunk]) -> dict:
    """Métrica 1: cantidad de chunks y distribución de tamaños en tokens."""
    sizes = sorted(c.n_tokens for c in chunks)
    if not sizes:
        return {"chunks": 0}
    return {"chunks": len(sizes), "min": sizes[0], "median": statistics.median(sizes),
            "p90": sizes[int(0.9 * (len(sizes) - 1))], "max": sizes[-1], "mean": round(statistics.mean(sizes), 1),
            "histogram": _histogram(sizes)}


def code_and_prose(chunks: list[Chunk]) -> dict:
    """Métrica 2: chunks con código y explicación juntos (algún bloque + ≥ 80 caracteres de prosa)."""
    n = sum(c.n_code_blocks > 0 and c.prose_chars >= MIN_PROSE_CHARS for c in chunks)
    return {"chunks": n, "pct": _pct(n, len(chunks))}


def cut_code_blocks(chunks: list[Chunk], parsed: dict[str, ParsedDoc]) -> dict:
    """Métrica 3: bloques de código con algún borde de chunk adentro (quedan partidos entre chunks)."""
    bounds = defaultdict(list)
    for c in chunks:
        bounds[c.doc_id] += [c.start, c.end]
    total = cut = 0
    for doc_id, p in parsed.items():
        bs = sorted(bounds.get(doc_id, []))
        for b in p.blocks:
            if b.kind != "code":
                continue
            total += 1
            if bisect_left(bs, b.end) > bisect_right(bs, b.start):  # algún borde estrictamente adentro
                cut += 1
    return {"code_blocks": total, "cut": cut, "pct": _pct(cut, total)}


def truncated(chunks: list[Chunk], count: TokenCounter) -> dict:
    """Métrica 4: chunks que superan el límite de cada modelo, contando el prefijo que se embebe con el chunk."""
    sizes = [c.n_tokens + count(c.prefix) + 2 for c in chunks]  # + "\n\n" entre prefijo y texto (aprox.)
    return {m: {"chunks": sum(s > lim for s in sizes), "pct": _pct(sum(s > lim for s in sizes), len(sizes))}
            for m, lim in MODEL_LIMITS.items()}


def use_cases(chunks: list[Chunk], cap: int = DEFAULT_CAP) -> dict:
    """Métrica 5: chunks que pasan el filtro y casos de uso por librería, sin tope y con tope (round_robin)."""
    no_cap = Counter(c.package for c in select(chunks, cap=None))
    with_cap = Counter(c.package for c in select(chunks, cap=cap))
    return {"pct_chunks_use_case": _pct(sum(no_cap.values()), len(chunks)),
            "total_no_cap": sum(no_cap.values()), "total_cap": sum(with_cap.values()), "cap": cap,
            "packages_over_cap": sum(n > cap for n in no_cap.values()),
            "packages_without_use_cases": sorted({c.package for c in chunks} - set(no_cap)),
            "per_package": {p: {"no_cap": no_cap[p], "cap": with_cap[p]} for p in sorted(no_cap)}}


def oversized(chunks: list[Chunk], max_tokens: int) -> dict:
    """Métrica 6: chunks por encima del máximo (en S3, un bloque solo que ya lo supera)."""
    over = [c for c in chunks if c.n_tokens > max_tokens]
    return {"chunks": len(over), "pct": _pct(len(over), len(chunks)),
            "by_package": dict(Counter(c.package for c in over).most_common())}


def border_sample(chunks: list[Chunk], n: int = 30, seed: int = 0, context: int = 600) -> str:
    """Métrica 7: bordes de corte al azar entre chunks seguidos de la misma sección, en Markdown para leer."""
    ordered = sorted(chunks, key=lambda c: (c.doc_id, c.start))
    pairs = [(a, b) for a, b in zip(ordered, ordered[1:]) if a.section_id == b.section_id]
    sample = random.Random(seed).sample(pairs, min(n, len(pairs)))
    out = [f"# Muestra de bordes de corte ({len(sample)} de {len(pairs)}, semilla {seed})", "",
           "Para cada borde: ¿el texto entre bloques quedó con el código que corresponde? (splitters.md §3.2)", ""]
    for i, (a, b) in enumerate(sample, 1):
        out += [f"## {i}. {a.prefix}", "", f"`{a.package}/{a.source_path}`, corte en el carácter {a.end}", "",
                "**Final del chunk anterior:**", "", "~~~~~~~~ text", a.text[-context:], "~~~~~~~~", "",
                "**Comienzo del chunk siguiente:**", "", "~~~~~~~~ text", b.text[:context], "~~~~~~~~", "",
                "- [ ] Bien asignado", ""]
    return "\n".join(out)
