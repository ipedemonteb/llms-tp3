"""Split dev/test del dataset de evaluación (dataset_eval.md §2.4, §3.7).

50/50 dentro de cada conjunto, **agrupado por librería de reemplazo**: todos los ítems cuyo target se reemplaza por lo
mismo van del mismo lado (PyPDF2 y PyPDF4 → pypdf; los manuales de toml y el negativo `toml.dump` → tomllib), para no
ajustar con un ítem y medir con otro casi igual. El reemplazo sale de la lista del validador (`DEPRECATED`), resolviendo
el import a paquete igual que él. Un target `not_found` (inventado) es su propio grupo. Los negativos se balancean
además por `negative_kind`. Determinista: misma semilla y mismos ítems, mismo split.
"""

import random
from collections import Counter, defaultdict

from tprag.validate.deprecated import DEPRECATED
from tprag.validate.packages import candidates, normalize

SEED = 2026
SUBSET_ORDER = ("manual_deprecated", "manual_hallucination", "negative", "synthetic")


def group_key(item: dict) -> str:
    for t in item["targets"]:
        names, _ = candidates(t["module"])
        for name in names:
            if (dep := DEPRECATED.get(normalize(name))) is not None:
                return "reemplazo:" + "+".join(dep.replacement)
    return "item:" + item["id"]


def assign_splits(items: list[dict], seed: int = SEED) -> dict[str, str]:
    """id → `dev` | `test`. Los conjuntos se procesan en orden; un grupo que ya tiene lado (de un conjunto anterior)
    lo conserva, y el resto se reparte para equilibrar la cantidad de ítems por lado."""
    rng = random.Random(seed)
    side: dict[str, str] = {}  # grupo → lado
    by_subset = defaultdict(list)
    for it in items:
        by_subset[it["subset"]].append(it)
    subsets = [s for s in SUBSET_ORDER if s in by_subset] + sorted(set(by_subset) - set(SUBSET_ORDER))

    for subset in subsets:
        strata = defaultdict(list)
        for it in by_subset[subset]:
            strata[it.get("negative_kind", "")].append(it)
        for stratum in sorted(strata):
            groups = defaultdict(list)
            for it in strata[stratum]:
                groups[group_key(it)].append(it)
            count = Counter({"dev": 0, "test": 0})
            for g, members in groups.items():
                if g in side:
                    count[side[g]] += len(members)
            free = sorted(g for g in groups if g not in side)
            rng.shuffle(free)
            free.sort(key=lambda g: -len(groups[g]))  # los grandes primero; el orden entre iguales lo da la semilla
            for g in free:
                if count["dev"] == count["test"]:
                    choice = rng.choice(("dev", "test"))
                else:
                    choice = min(count, key=count.get)
                side[g] = choice
                count[choice] += len(groups[g])
    return {it["id"]: side[group_key(it)] for it in items}
