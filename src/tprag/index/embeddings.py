"""Modelos de embeddings por id (estudio §4.1) y caché en disco para no pagar dos veces el mismo texto.

`langchain.embeddings.CacheBackedEmbeddings` no está en las dependencias (vive en el paquete `langchain` clásico), así
que el caché es propio: SQLite con una fila por (modelo, hash del texto). Además permite saber, antes de llamar a la
API, qué textos faltan (`missing`), que es lo que usa la estimación de costo.
"""

import hashlib
import sqlite3
from array import array
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from langchain_core.embeddings import Embeddings

from tprag.split.store import ROOT

CACHE_PATH = ROOT / "data" / "index" / "cache" / "embeddings.sqlite"


@dataclass(frozen=True)
class EmbeddingModel:
    id: str
    name: str  # nombre real del modelo (el vector no depende de por dónde se pida)
    api_name: str  # cómo se lo pide: el proxy de la cátedra (OPENAI_BASE_URL) usa el prefijo "openai/"
    usd_per_m_tokens: float  # precio verificado en developers.openai.com/api/docs/pricing el 2026-10-27
    tokenizer: str = "cl100k_base"
    dimensions: int | None = None  # None = las del modelo; text-embedding-3 permite reducirlas

    @property
    def cache_key(self) -> str:
        """Clave del caché: nombre real + dimensiones, no el id corto (si un id cambia de modelo, no se mezclan)."""
        return self.name if self.dimensions is None else f"{self.name}@{self.dimensions}"


MODELS = {
    "e1": EmbeddingModel("e1", "text-embedding-3-small", "openai/text-embedding-3-small", 0.02),
    "e2": EmbeddingModel("e2", "text-embedding-3-large", "openai/text-embedding-3-large", 0.13),
}


def openai_embeddings(model: EmbeddingModel) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415  (pide la API key al construirse)

    # tiktoken_model_name: LangChain parte los textos largos con tiktoken y no reconoce el nombre con prefijo
    return OpenAIEmbeddings(model=model.api_name, tiktoken_model_name=model.name, dimensions=model.dimensions)


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class CachedEmbeddings(Embeddings):
    """Envuelve un modelo: lo que ya está en el caché no se vuelve a pedir. `model_id` separa los vectores de cada
    modelo (`EmbeddingModel.cache_key`). El modelo se crea recién cuando hace falta (así `missing` y el dry-run
    funcionan sin API key)."""

    def __init__(self, model_id: str, factory: Callable[[], Embeddings], path: Path = CACHE_PATH):
        self.model_id = model_id
        self._factory = factory
        self._inner: Embeddings | None = None
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS vectors (model TEXT, key TEXT, vector BLOB, "
                         "PRIMARY KEY (model, key))")
        self.hits = 0
        self.misses = 0

    def _get(self, keys: list[str]) -> dict[str, list[float]]:
        found = {}
        for i in range(0, len(keys), 500):  # límite de parámetros de SQLite
            batch = keys[i:i + 500]
            rows = self._db.execute(f"SELECT key, vector FROM vectors WHERE model = ? AND key IN "
                                    f"({','.join('?' * len(batch))})", [self.model_id, *batch])
            found.update((k, array("f", v).tolist()) for k, v in rows)
        return found

    def missing(self, texts: list[str]) -> list[str]:
        """Textos distintos que todavía no están en el caché (los que se pagarían)."""
        unique = dict.fromkeys(texts)
        found = self._get([_key(t) for t in unique])
        return [t for t in unique if _key(t) not in found]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [_key(t) for t in texts]
        found = self._get(keys)
        todo = {k: t for k, t in zip(keys, texts) if k not in found}
        self.hits += len(texts) - len(todo)
        self.misses += len(todo)
        if todo:
            if self._inner is None:
                self._inner = self._factory()
            # float32 también la primera vez: así un índice armado de cero y uno rehecho desde el caché son iguales
            vectors = [array("f", v) for v in self._inner.embed_documents(list(todo.values()))]
            self._db.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)",
                                 [(self.model_id, k, v.tobytes()) for k, v in zip(todo, vectors)])
            self._db.commit()
            found.update((k, v.tolist()) for k, v in zip(todo, vectors))
        return [found[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        # para OpenAI consulta y documento se embeben igual; los modelos asimétricos (E5) necesitarán su wrapper
        return self.embed_documents([text])[0]


def cached(model_id: str, path: Path = CACHE_PATH) -> CachedEmbeddings:
    model = MODELS[model_id]
    return CachedEmbeddings(model.cache_key, lambda: openai_embeddings(model), path)
