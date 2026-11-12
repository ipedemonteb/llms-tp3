"""Rerankers locales (docs/agents/plan_reranker.md): un segundo paso que reordena los chunks que ya devolvió Chroma.

Se usan solo en consulta: no tocan el índice. Cada par (consulta, chunk) se puntúa por separado, así que el puntaje
no depende de qué otros candidatos haya y se puede guardar en un caché (SQLite, como el de embeddings).

- K1 `Qwen/Qwen3-Reranker-0.6B`: LLM chico; el puntaje es la probabilidad de "yes" ante una instrucción de la tarea.
- K2 `Alibaba-NLP/gte-reranker-modernbert-base`: cross-encoder clásico (ModernBERT, 149M).

Los dos corren con `CrossEncoder` de sentence-transformers (≥ 5.4, que soporta los rerankers tipo Qwen3) y necesitan el
grupo opcional `local`. `rerank` y `fuse` son funciones puras: reciben los hits y los puntajes.
"""

import hashlib
import sqlite3
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tprag.split.store import ROOT

CACHE_PATH = ROOT / "data" / "index" / "cache" / "rerank.sqlite"
MAX_LENGTH = 2048  # tokens por par: cubre consulta + chunk de 1024 (cl100k) con margen; se informa cuántos se truncan
RRF_C = 60  # constante de Reciprocal Rank Fusion (Cormack et al. 2009): el valor estándar, fijado antes de correr
INSTRUCTION = ("Given Python code that uses a library that may not exist or is deprecated, retrieve the documentation "
               "passage of a real Python library that shows how to do the same thing.")

Pair = tuple[str, str]


@dataclass(frozen=True)
class RerankerModel:
    id: str
    name: str  # repo de Hugging Face
    label: str
    instruction: str | None = None  # solo los que aceptan instrucción (Qwen3-Reranker)
    max_length: int = MAX_LENGTH
    trust_remote_code: bool = False
    # float16 en la GPU de la Mac: K2 pasa de 0,37 a 0,22 s/par con los mismos puntajes (difieren en el 3.er decimal)
    dtype: str = "float16"

    @property
    def cache_key(self) -> str:
        """Nombre real + largo + precisión + instrucción: si cambia alguno, el puntaje es otro."""
        key = f"{self.name}@len{self.max_length}@{self.dtype}"
        if self.instruction:
            key += "@inst-" + hashlib.sha256(self.instruction.encode()).hexdigest()[:12]
        return key


RERANKERS = {
    "k1": RerankerModel("k1", "Qwen/Qwen3-Reranker-0.6B", "Qwen3-Reranker-0.6B", instruction=INSTRUCTION),
    "k2": RerankerModel("k2", "Alibaba-NLP/gte-reranker-modernbert-base", "gte-reranker-modernbert-base"),
}


class Scorer(Protocol):
    def score(self, pairs: list[Pair]) -> list[float]: ...


class LocalReranker:
    """Modelo de `CrossEncoder` en la Mac (MPS si hay, si no CPU). Se carga al primer uso. Devuelve el puntaje crudo
    del modelo (logit o probabilidad según el modelo): sirve para ordenar y para el AUROC, no para comparar modelos."""

    def __init__(self, model: RerankerModel, batch_size: int = 1):
        self.model, self.batch_size = model, batch_size
        self.peak_mb = 0.0  # memoria máxima reservada en la GPU (MPS) durante el puntaje
        self._ce = None

    def _load(self):
        if self._ce is None:
            import torch  # noqa: PLC0415  (grupo opcional `local`)
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            kwargs = {"prompts": {"tarea": self.model.instruction},
                      "default_prompt_name": "tarea"} if self.model.instruction else {}
            dtype = getattr(torch, self.model.dtype) if device == "mps" else torch.float32  # en CPU, float32
            self._ce = CrossEncoder(self.model.name, device=device, max_length=self.model.max_length,
                                    trust_remote_code=self.model.trust_remote_code, model_kwargs={"dtype": dtype},
                                    **kwargs)
        return self._ce

    def score(self, pairs: list[Pair]) -> list[float]:
        import torch  # noqa: PLC0415

        scores = self._load().predict(pairs, batch_size=self.batch_size, convert_to_numpy=True,
                                      show_progress_bar=len(pairs) > 64 and sys.stderr.isatty())
        if torch.backends.mps.is_available():
            self.peak_mb = max(self.peak_mb, torch.mps.driver_allocated_memory() / 2**20)
            torch.mps.empty_cache()  # como en LocalEmbeddings: sin esto la GPU compartida se llena y la Mac swapea
        return [float(s) for s in scores]

    def n_tokens(self, texts: Sequence[str]) -> list[int]:
        """Tokens de cada texto con el tokenizer del modelo, sin tokens especiales (para estimar el truncado)."""
        tok = self._load().tokenizer
        return [len(ids) for ids in tok(list(texts), add_special_tokens=False)["input_ids"]]


def _key(pair: Pair) -> str:
    return hashlib.sha256((pair[0] + "\x00" + pair[1]).encode()).hexdigest()


class CachedScores:
    """Envuelve un reranker: los pares ya puntuados no se vuelven a calcular. El modelo se crea recién cuando hace falta
    (así `missing` y el dry-run no descargan nada). Se guarda cada lote: si la corrida se corta, se retoma."""

    def __init__(self, model_id: str, factory: Callable[[], Scorer], path: Path = CACHE_PATH, commit_every: int = 64):
        self.model_id, self._factory, self.commit_every = model_id, factory, commit_every
        self._inner: Scorer | None = None
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS scores (model TEXT, key TEXT, score REAL, PRIMARY KEY (model, key))")
        self.hits = 0
        self.misses = 0

    @property
    def inner(self) -> Scorer:
        if self._inner is None:
            self._inner = self._factory()
        return self._inner

    def _get(self, keys: list[str]) -> dict[str, float]:
        found = {}
        for i in range(0, len(keys), 500):  # límite de parámetros de SQLite
            batch = keys[i:i + 500]
            rows = self._db.execute(f"SELECT key, score FROM scores WHERE model = ? AND key IN "
                                    f"({','.join('?' * len(batch))})", [self.model_id, *batch])
            found.update(rows)
        return found

    def missing(self, pairs: list[Pair]) -> list[Pair]:
        """Pares distintos que todavía no están en el caché (los que habría que calcular)."""
        unique = list(dict.fromkeys(pairs))
        found = self._get([_key(p) for p in unique])
        return [p for p in unique if _key(p) not in found]

    def score(self, pairs: list[Pair], progress: Callable[[int, int], None] | None = None) -> list[float]:
        keys = [_key(p) for p in pairs]
        found = self._get(keys)
        todo = {k: p for k, p in zip(keys, pairs) if k not in found}
        self.hits += len(pairs) - len(todo)
        self.misses += len(todo)
        items = list(todo.items())
        for i in range(0, len(items), self.commit_every):
            batch = items[i:i + self.commit_every]
            values = self.inner.score([p for _, p in batch])
            self._db.executemany("INSERT OR REPLACE INTO scores VALUES (?, ?, ?)",
                                 [(self.model_id, k, v) for (k, _), v in zip(batch, values)])
            self._db.commit()
            found.update((k, v) for (k, _), v in zip(batch, values))
            if progress:
                progress(min(i + self.commit_every, len(items)), len(items))
        return [found[k] for k in keys]


def cached_reranker(model_id: str, path: Path = CACHE_PATH, batch_size: int = 1) -> CachedScores:
    model = RERANKERS[model_id]
    return CachedScores(model.cache_key, lambda: LocalReranker(model, batch_size), path)


def rerank(hits: list[dict], scores: Sequence[float], k: int) -> list[dict]:
    """Ordena los primeros `k` hits por puntaje del reranker (mayor primero; empate: el orden denso). Los demás quedan
    detrás, en su orden. Cada hit conserva su `score` (coseno) y suma `dense_rank` y, si fue reordenado,
    `rerank_score`. `scores` se alinea con `hits` y tiene al menos `k` valores (o tantos como hits)."""
    k = min(k, len(hits))
    if len(scores) < k:
        raise ValueError(f"faltan puntajes: {len(scores)} para {k} candidatos")
    head = [{**h, "dense_rank": i, "rerank_score": float(s)} for i, (h, s) in enumerate(zip(hits[:k], scores), 1)]
    head.sort(key=lambda h: (-h["rerank_score"], h["dense_rank"]))
    return head + [{**h, "dense_rank": i} for i, h in enumerate(hits[k:], k + 1)]


def fuse(hits: list[dict], scores: Sequence[float], k: int, c: int = RRF_C) -> list[dict]:
    """Reciprocal Rank Fusion de los primeros `k`: 1/(c + puesto denso) + 1/(c + puesto del reranker). Empate: el
    orden denso. Mismo formato que `rerank`, más `rrf`."""
    by_rerank = rerank(hits, scores, k)[:min(k, len(hits))]
    fused = [{**h, "rrf": 1 / (c + h["dense_rank"]) + 1 / (c + r)} for r, h in enumerate(by_rerank, 1)]
    fused.sort(key=lambda h: (-h["rrf"], h["dense_rank"]))
    return fused + [{**h, "dense_rank": i} for i, h in enumerate(hits[len(fused):], len(fused) + 1)]
