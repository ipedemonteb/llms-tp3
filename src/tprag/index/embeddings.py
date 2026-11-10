"""Modelos de embeddings por id (estudio §4.1) y caché en disco para no pagar dos veces el mismo texto.

E1 y E2 son de OpenAI (API, se pagan por token). E3 a E5 son locales (sentence-transformers, 0 USD, corren en la
Mac): requieren el grupo opcional de dependencias `local` (`uv sync --group local`).

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
    provider: str = "openai"  # "openai" (API) o "local" (sentence-transformers)
    query_prefix: str = ""  # modelos asimétricos (nomic): texto que va antes de la consulta...
    doc_prefix: str = ""  # ...y antes del documento
    max_seq_length: int = 2048  # solo local: tokens del propio modelo que lee; lo que pasa de acá se trunca
    trust_remote_code: bool = False  # solo local: jina y nomic traen código propio en el repo del modelo

    @property
    def asymmetric(self) -> bool:
        return bool(self.query_prefix or self.doc_prefix)

    @property
    def cache_key(self) -> str:
        """Clave del caché: nombre real + dimensiones, no el id corto (si un id cambia de modelo, no se mezclan)."""
        return self.name if self.dimensions is None else f"{self.name}@{self.dimensions}"


MODELS = {
    "e1": EmbeddingModel("e1", "text-embedding-3-small", "openai/text-embedding-3-small", 0.02),
    "e2": EmbeddingModel("e2", "text-embedding-3-large", "openai/text-embedding-3-large", 0.13),
    # Locales (plan_experimentos_indice.md §3.1). api_name vacío y precio 0: no hay API ni se paga por token. Los
    # tokens que cuenta `estimate` son los de cl100k_base, solo una referencia del tamaño (cada modelo tiene el suyo).
    "e3": EmbeddingModel("e3", "BAAI/bge-m3", "", 0.0, provider="local"),
    "e4": EmbeddingModel("e4", "jinaai/jina-embeddings-v2-base-code", "", 0.0, provider="local",
                         trust_remote_code=True),
    "e5": EmbeddingModel("e5", "nomic-ai/nomic-embed-text-v1.5", "", 0.0, provider="local", trust_remote_code=True,
                         query_prefix="search_query: ", doc_prefix="search_document: "),
}


def openai_embeddings(model: EmbeddingModel) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings  # noqa: PLC0415  (pide la API key al construirse)

    # tiktoken_model_name: LangChain parte los textos largos con tiktoken y no reconoce el nombre con prefijo
    return OpenAIEmbeddings(model=model.api_name, tiktoken_model_name=model.name, dimensions=model.dimensions)


class LocalEmbeddings(Embeddings):
    """Modelo de sentence-transformers en la Mac (MPS si hay, si no CPU). Vectores normalizados (coseno). Agrega el
    prefijo de consulta o de documento que pida el modelo. Se carga al primer uso."""

    def __init__(self, model: EmbeddingModel, batch_size: int = 8):
        if model.dimensions is not None:
            raise NotImplementedError("reducir dimensiones de un modelo local todavía no está soportado")
        self.model, self.batch_size = model, batch_size
        self._st = None

    def _load(self):
        if self._st is None:
            import torch  # noqa: PLC0415  (grupo opcional `local`)
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            self._st = SentenceTransformer(self.model.name, device=device,
                                           trust_remote_code=self.model.trust_remote_code)
            self._st.max_seq_length = self.model.max_seq_length
        return self._st

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        vectors = self._load().encode([prefix + t for t in texts], batch_size=self.batch_size,
                                      normalize_embeddings=True, show_progress_bar=len(texts) > 64)
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, self.model.doc_prefix)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], self.model.query_prefix)[0]


def model_embeddings(model: EmbeddingModel) -> Embeddings:
    return openai_embeddings(model) if model.provider == "openai" else LocalEmbeddings(model)


def _key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class CachedEmbeddings(Embeddings):
    """Envuelve un modelo: lo que ya está en el caché no se vuelve a pedir. `model_id` separa los vectores de cada
    modelo (`EmbeddingModel.cache_key`). El modelo se crea recién cuando hace falta (así `missing` y el dry-run
    funcionan sin API key). Con `asymmetric` (modelos con prefijo distinto para consulta y documento) los vectores de
    consulta se guardan aparte."""

    def __init__(self, model_id: str, factory: Callable[[], Embeddings], path: Path = CACHE_PATH,
                 asymmetric: bool = False):
        self.model_id = model_id
        self.asymmetric = asymmetric
        self._factory = factory
        self._inner: Embeddings | None = None
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS vectors (model TEXT, key TEXT, vector BLOB, "
                         "PRIMARY KEY (model, key))")
        self.hits = 0
        self.misses = 0

    def _get(self, keys: list[str], model_id: str | None = None) -> dict[str, list[float]]:
        found = {}
        for i in range(0, len(keys), 500):  # límite de parámetros de SQLite
            batch = keys[i:i + 500]
            rows = self._db.execute(f"SELECT key, vector FROM vectors WHERE model = ? AND key IN "
                                    f"({','.join('?' * len(batch))})", [model_id or self.model_id, *batch])
            found.update((k, array("f", v).tolist()) for k, v in rows)
        return found

    def missing(self, texts: list[str]) -> list[str]:
        """Textos distintos que todavía no están en el caché (los que se pagarían)."""
        unique = dict.fromkeys(texts)
        found = self._get([_key(t) for t in unique])
        return [t for t in unique if _key(t) not in found]

    def _embed(self, texts: list[str], model_id: str, embed: Callable[[list[str]], list[list[float]]]):
        keys = [_key(t) for t in texts]
        found = self._get(keys, model_id)
        todo = {k: t for k, t in zip(keys, texts) if k not in found}
        self.hits += len(texts) - len(todo)
        self.misses += len(todo)
        if todo:
            # float32 también la primera vez: así un índice armado de cero y uno rehecho desde el caché son iguales
            vectors = [array("f", v) for v in embed(list(todo.values()))]
            self._db.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)",
                                 [(model_id, k, v.tobytes()) for k, v in zip(todo, vectors)])
            self._db.commit()
            found.update((k, v.tolist()) for k, v in zip(todo, vectors))
        return [found[k] for k in keys]

    def _model(self) -> Embeddings:
        if self._inner is None:
            self._inner = self._factory()
        return self._inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, self.model_id, lambda todo: self._model().embed_documents(todo))

    def embed_query(self, text: str) -> list[float]:
        if not self.asymmetric:  # OpenAI: consulta y documento se embeben igual
            return self.embed_documents([text])[0]
        return self._embed([text], self.model_id + "#query", lambda todo: [self._model().embed_query(todo[0])])[0]


def cached(model_id: str, path: Path = CACHE_PATH) -> CachedEmbeddings:
    model = MODELS[model_id]
    return CachedEmbeddings(model.cache_key, lambda: model_embeddings(model), path, model.asymmetric)
