"""Filtro de casos de uso y tope por librería (splitters.md §3.4, §3.5). Se aplica al armar el índice (§3.7).

Con tope, primero entran los `kind` de mayor prioridad; dentro del mismo `kind` decide la estrategia:

- `round_robin` (default): en rondas, el primero de cada página, después el segundo de cada una, etc.
- `in_order`: archivo por archivo, en orden alfabético.
- `random`: al azar con semilla fija.
"""

import random
from collections import defaultdict

from .chunk import Chunk

DEFAULT_CAP = 150
STRATEGIES = ("round_robin", "in_order", "random")
# plan §7.5; readme y docstrings no estaban en la lista: readme junto a quickstart, docstrings al final
KIND_PRIORITY = ("migration", "quickstart", "readme", "tutorial", "guide", "examples", "docstrings")


def _order(chunks: list[Chunk], strategy: str, rng: random.Random) -> list[Chunk]:
    chunks = sorted(chunks, key=lambda c: (c.source_path, c.start))
    if strategy == "in_order":
        return chunks
    if strategy == "random":
        rng.shuffle(chunks)
        return chunks
    if strategy == "round_robin":
        pages = defaultdict(list)
        for c in chunks:
            pages[c.source_path].append(c)
        queues = list(pages.values())
        return [q[i] for i in range(max(map(len, queues), default=0)) for q in queues if i < len(q)]
    raise ValueError(f"estrategia desconocida: {strategy}")


def _kind_rank(kind: str) -> int:
    return KIND_PRIORITY.index(kind) if kind in KIND_PRIORITY else len(KIND_PRIORITY)


def select(chunks: list[Chunk], cap: int | None = DEFAULT_CAP, strategy: str = "round_robin",
           seed: int = 0) -> list[Chunk]:
    """Casos de uso que van al índice: los que pasan el filtro y, con `cap`, a lo sumo `cap` por librería."""
    if strategy not in STRATEGIES:
        raise ValueError(f"estrategia desconocida: {strategy}")
    by_package = defaultdict(list)
    for c in chunks:
        if c.is_use_case:
            by_package[c.package].append(c)
    rng = random.Random(seed)
    out = []
    for package in sorted(by_package):
        pkg = by_package[package]
        if cap is None or len(pkg) <= cap:
            out += pkg
            continue
        by_kind = defaultdict(list)
        for c in pkg:
            by_kind[c.kind].append(c)
        chosen = []
        for kind in sorted(by_kind, key=_kind_rank):
            chosen += _order(by_kind[kind], strategy, rng)[:cap - len(chosen)]
            if len(chosen) >= cap:
                break
        out += chosen
    return out
