"""Bootstrap y comparaciones por pares para comparar configuraciones (plan de experimentos, etapa 1).

Se remuestrean los ÍTEMS de dev (no los targets): los targets de un mismo script están correlacionados. Todas las
configuraciones usan los mismos remuestreos, así la diferencia entre dos es por pares (mismos ítems, mismos sorteos).
Funciones puras sobre arrays de numpy; no leen resultados ni usan Chroma.
"""

import numpy as np

LEVEL = 95


def resample_weights(n_items: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_items): cuántas veces entra cada ítem en cada remuestreo."""
    idx = np.random.default_rng(seed).integers(0, n_items, size=(n_boot, n_items))
    weights = np.zeros((n_boot, n_items), dtype=np.int32)
    np.add.at(weights, (np.arange(n_boot)[:, None], idx), 1)
    return weights


def rate(values: np.ndarray, item_of: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
    """Proporción de aciertos (`values` en 0/1, uno por target) y su distribución bootstrap. `item_of`: ítem de cada
    target."""
    w = weights[:, item_of]
    total = w.sum(1)
    boot = np.where(total > 0, (w * values).sum(1) / np.maximum(total, 1), np.nan)
    return float(values.mean()), boot


def auroc(pos: np.ndarray, neg: np.ndarray, pos_item: np.ndarray, neg_item: np.ndarray,
          weights: np.ndarray) -> tuple[float, np.ndarray]:
    """AUROC (empates = 1/2) y su distribución bootstrap, remuestreando ítems."""
    wins = (pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])
    wp, wn = weights[:, pos_item], weights[:, neg_item]
    denominator = wp.sum(1) * wn.sum(1)
    boot = np.where(denominator > 0, ((wp @ wins) * wn).sum(1) / np.maximum(denominator, 1), np.nan)
    return float(wins.mean()), boot


def interval(boot: np.ndarray) -> list[float]:
    lo, hi = np.nanpercentile(boot, [(100 - LEVEL) / 2, 100 - (100 - LEVEL) / 2])
    return [round(float(lo), 4), round(float(hi), 4)]


def compare(point_a: float, boot_a: np.ndarray, point_b: float, boot_b: np.ndarray) -> dict:
    """A − B por pares. `empate` si el intervalo de la diferencia incluye el cero."""
    lo, hi = interval(boot_a - boot_b)
    verdict = "empate" if lo <= 0 <= hi else ("mejor" if lo > 0 else "peor")
    return {"diff": round(point_a - point_b, 4), "ci": [lo, hi], "verdict": verdict}
