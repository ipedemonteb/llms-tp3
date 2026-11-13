"""Grilla del estudio del reranker (docs/agents/plan_reranker.md) y regla de decisión. Es la fuente única: la usan el
runner (`run_rerank_study.py`) y el análisis (`analyze_rerank.py`).

Diseño: sobre la base del estudio de hiperparámetros (S3 · 1024 · E4 · R1 · tope 150), cada variante es un reranker,
una cantidad de candidatos K y un modo (`rr`: orden del reranker; `rrf`: fusión con el orden denso). Los puntajes se
calculan una vez sobre los 50 recuperados; cada variante es un corte de esos puntajes.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from tprag.index.rerank import RERANKERS, fuse, rerank
from tprag.split.store import ROOT

from .hyperparams import BASE

POOL = 50  # candidatos que se puntúan (los que guarda el paso 7)
KS = (10, 20, 30, 50)
MODES = {"rr": "reranker", "rrf": "fusión RRF"}
RESULTS_DIR = ROOT / "results" / "eval" / "rerank"


@dataclass(frozen=True)
class RerankVariant:
    reranker: str  # id de RERANKERS
    k: int
    mode: str  # "rr" o "rrf"

    @property
    def id(self) -> str:
        return f"{self.reranker}-{self.mode}{self.k}"

    @property
    def label(self) -> str:
        return f"{RERANKERS[self.reranker].label} · {MODES[self.mode]} · top-{self.k}"

    def apply(self, hits: list[dict], scores: list[float]) -> list[dict]:
        return (rerank if self.mode == "rr" else fuse)(hits, scores, self.k)


VARIANTS = tuple(RerankVariant(r, k, m) for r in RERANKERS for m in MODES for k in KS)
BY_ID = {v.id: v for v in VARIANTS}
BASE_VARIANT = BASE  # la configuración del índice sobre la que se reordena


def scores_path(collection: str, reranker: str, split: str) -> Path:
    """Puntajes de un reranker sobre los 50 candidatos de cada target (los deja `run_rerank_study.py`)."""
    return RESULTS_DIR / f"{collection}__qA__{reranker}__{split}.scores.jsonl"


def targets_path(collection: str, variant_id: str, split: str) -> Path:
    """Detalle por target de una variante, en el mismo formato que el `.targets.jsonl` del paso 7."""
    return RESULTS_DIR / f"{collection}__qA__{variant_id}__{split}.targets.jsonl"


def write_jsonl_once(path: Path, rows: list[dict]) -> str:
    """Escribe `rows` como JSONL sin pisar resultados anteriores: si el archivo no existe lo crea ("nuevo"); si existe
    con el mismo contenido no lo toca ("igual"); si existe con otro contenido falla y lo deja como estaba."""
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    if path.exists():
        if path.read_text(encoding="utf-8") == text:
            return "igual"
        raise FileExistsError(f"{path} ya existe con otro contenido: no se pisa")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "nuevo"


def parse_pairs(specs: list[str]) -> list[tuple[str, str]]:
    """`["k1-rr30:k2-rrf10", ...]` → [("k1-rr30", "k2-rrf10"), ...] (A − B). Valida que las variantes existan."""
    out = []
    for s in specs:
        a, sep, b = s.partition(":")
        if not sep or a not in BY_ID or b not in BY_ID or a == b:
            raise ValueError(f"par inválido: {s!r} (formato A:B con ids de la grilla, p. ej. k1-rr30:k2-rrf10)")
        out.append((a, b))
    return out


# Los mismos pares del desvío del estudio original (plan_reranker.md §7, rerank_choice.json)
DEFAULT_PAIRS = ("k1-rr30:k2-rrf10", "k1-rr20:k2-rrf10", "k1-rr50:k1-rr30", "k1-rr30:k1-rr20")


def choose(rows: dict[str, dict]) -> str | None:
    """Variante que reemplaza a la base, o None.

    `rows[id]` = {"span_verdict": veredicto de Span Hit@2000 contra la base ("mejor", "empate" o "peor"),
    "pkg_verdict": ídem en Package Hit@1, "seconds": segundos por consulta (costo), "span": Span Hit@2000}.

    Regla (fijada antes de correr): solo cuentan las que mejoran Span Hit@2000 de forma significativa y no son peores
    en Package Hit@1. De esas gana la más barata; a igual costo, la de mayor Span Hit@2000.
    """
    eligible = [i for i, r in rows.items() if r["span_verdict"] == "mejor" and r["pkg_verdict"] != "peor"]
    if not eligible:
        return None
    return min(eligible, key=lambda i: (rows[i]["seconds"], -rows[i]["span"]))
