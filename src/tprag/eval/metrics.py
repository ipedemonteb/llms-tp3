"""Métricas del retrieval (paso 7, docs/agents/evaluacion.md §2.2, §2.3). Funciones puras: sin API ni Chroma.

Un *hit* es un chunk recuperado, en orden: `{"id", "package", "score", "doc_id", "start", "end", "n_tokens"}` (score =
similitud coseno). Un *span* gold es `(doc_id, inicio, fin)` de un bloque de código en la doc normalizada.

- **Librería** (Package Hit@k, MRR): sobre el ranking de librerías distintas que arma cada estrategia de elección.
- **Caso de uso** (Span Hit@k, nDCG@10, Span Hit con presupuesto): sobre el ranking de chunks; un chunk acierta si
  cubre ≥ 50% de un bloque gold.
- **Confianza** (AUROC): score del mejor chunk, positivos (con reemplazo) contra negativos.
"""

from collections import defaultdict
from math import log2
from statistics import mean

PACKAGE_KS = (1, 3, 5, 10)
SPAN_KS = (1, 3, 5, 10)
NDCG_K = 10
CONTEXT_BUDGET = 2000  # tokens: Span Hit con presupuesto fijo, para comparar splitters
MIN_GOLD_COVERAGE = 0.5
VOTE_NS = (5, 10, 20)  # N de las variantes de votación (10 = base; 5 y 20 se prueban en dev)

Span = tuple[str, int, int]


def strategies() -> dict[str, tuple[str, int | None]]:
    """Nombre → (estrategia, N). `best_chunk` es la base; las otras, variantes calculadas en la misma corrida."""
    out = {"best_chunk": ("best_chunk", None)}
    for kind in ("sum_rr", "count"):
        out.update({f"{kind}@{n}": (kind, n) for n in VOTE_NS})
    return out


def library_ranking(packages: list[str], strategy: str, n: int | None = None) -> list[str]:
    """Librerías distintas en orden de elección. `packages`: el paquete de cada chunk recuperado, en orden.

    - `best_chunk`: cada librería vale su mejor chunk (orden de primera aparición).
    - `sum_rr`: suma de 1/posición de sus chunks en el top-`n`.
    - `count`: cantidad de chunks en el top-`n`.
    En las votaciones, los empates y las librerías que quedan fuera del top-`n` siguen el orden de primera aparición.
    """
    first = list(dict.fromkeys(packages))
    if strategy == "best_chunk":
        return first
    if strategy not in ("sum_rr", "count"):
        raise ValueError(f"estrategia desconocida: {strategy}")
    votes = defaultdict(float)
    for rank, pkg in enumerate(packages[:n], start=1):
        votes[pkg] += 1 / rank if strategy == "sum_rr" else 1
    voted = sorted(votes, key=lambda p: (-votes[p], first.index(p)))
    return voted + [p for p in first if p not in votes]


def package_rank(ranking: list[str], acceptable: list[str]) -> int | None:
    """Posición (desde 1) del primer paquete aceptable, o None si no aparece."""
    return next((i for i, p in enumerate(ranking, start=1) if p in acceptable), None)


def covered_spans(hit: dict, spans: list[Span]) -> set[int]:
    """Índices de los bloques gold que el chunk cubre en ≥ 50%."""
    out = set()
    for i, (doc_id, start, end) in enumerate(spans):
        overlap = max(0, min(hit["end"], end) - max(hit["start"], start))
        if hit["doc_id"] == doc_id and overlap >= MIN_GOLD_COVERAGE * (end - start):
            out.add(i)
    return out


def span_rank(hits: list[dict], spans: list[Span]) -> int | None:
    """Posición (desde 1) del primer chunk que cubre algún bloque gold."""
    return next((i for i, h in enumerate(hits, start=1) if covered_spans(h, spans)), None)


def ndcg(hits: list[dict], spans: list[Span], k: int = NDCG_K) -> float:
    """Relevancia binaria: un chunk suma si cubre un bloque gold que ningún chunk anterior cubría (así varios chunks
    sobre el mismo bloque no cuentan dos veces). Ideal: un bloque gold distinto en cada una de las primeras
    posiciones."""
    seen, dcg = set(), 0.0
    for i, h in enumerate(hits[:k]):
        new = covered_spans(h, spans) - seen
        if new:
            dcg += 1 / log2(i + 2)
            seen |= new
    ideal = sum(1 / log2(i + 2) for i in range(min(len(spans), k)))
    return dcg / ideal if ideal else 0.0


def budget_hit(hits: list[dict], spans: list[Span], budget: int = CONTEXT_BUDGET) -> bool:
    """¿Algún chunk de los que entran en `budget` tokens (en orden; el primero entra siempre) cubre un bloque gold?"""
    used = 0
    for i, h in enumerate(hits):
        used += h["n_tokens"]
        if i and used > budget:
            return False
        if covered_spans(h, spans):
            return True
    return False


def auroc(positives: list[float], negatives: list[float]) -> float | None:
    """Probabilidad de que un positivo tenga más score que un negativo (empates: 1/2). None si falta un lado."""
    if not positives or not negatives:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def target_metrics(hits: list[dict], acceptable: list[str], spans: list[Span]) -> dict:
    """Lo que se guarda por target en el detalle (`.targets.jsonl`). En los negativos solo cuenta `top1_score`."""
    packages = [h["package"] for h in hits]
    return {
        "top1_score": hits[0]["score"] if hits else None,
        "package_rank": {name: package_rank(library_ranking(packages, s, n), acceptable)
                         for name, (s, n) in strategies().items()} if acceptable else None,
        "span_rank": span_rank(hits, spans) if spans else None,
        "ndcg@10": ndcg(hits, spans) if spans else None,
        "span_hit@budget": budget_hit(hits, spans) if spans else None,
    }


def _rate(values: list[bool]) -> float | None:
    return round(mean(values), 4) if values else None


def aggregate(records: list[dict]) -> dict:
    """Promedios sobre los targets positivos de `records` (los que tienen `metrics.package_rank`)."""
    pos = [r["metrics"] for r in records if r["metrics"]["package_rank"] is not None]
    with_gold = [m for m in pos if m["ndcg@10"] is not None]  # span_rank None también es "no lo encontró"
    out = {"targets": len(pos), "package": {}}
    for name in strategies():
        ranks = [m["package_rank"][name] for m in pos]
        out["package"][name] = {
            **{f"hit@{k}": _rate([r is not None and r <= k for r in ranks]) for k in PACKAGE_KS},
            "mrr": _rate([1 / r if r else 0.0 for r in ranks]),
        }
    out["span"] = {
        **{f"hit@{k}": _rate([m["span_rank"] is not None and m["span_rank"] <= k for m in with_gold])
           for k in SPAN_KS},
        f"ndcg@{NDCG_K}": _rate([m["ndcg@10"] for m in with_gold]),
        f"hit@{CONTEXT_BUDGET}tok": _rate([m["span_hit@budget"] for m in with_gold]),
    }
    return out


def summarize(records: list[dict]) -> dict:
    """Métricas del total, por subconjunto y por idioma, más el AUROC del score del mejor chunk."""
    def group(key: str) -> dict:
        keys = sorted({r[key] for r in records})
        return {k: aggregate([r for r in records if r[key] == k]) for k in keys}

    scores = [(r["metrics"]["package_rank"] is not None, r["metrics"]["top1_score"]) for r in records
              if r["metrics"]["top1_score"] is not None]
    pos = [s for is_pos, s in scores if is_pos]
    neg = [s for is_pos, s in scores if not is_pos]
    value = auroc(pos, neg)
    return {
        "all": aggregate(records), "by_subset": group("subset"), "by_lang": group("lang"),
        "auroc": {"value": round(value, 4) if value is not None else None, "positives": len(pos),
                  "negatives": len(neg)},
    }
