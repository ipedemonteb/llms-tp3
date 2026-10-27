"""Armado de una colección de Chroma a partir de `chunks.jsonl` (estudio §4, §5.4; splitters.md §3.7).

Una colección por configuración, con distancia coseno. Al indexar se aplican el filtro de casos de uso y el tope
(`tprag.split.select`). Se embebe la representación R1 (prefijo + chunk); el texto sin prefijo se recupera con
`chunk_text`.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from tprag.split.chunk import Chunk
from tprag.split.select import DEFAULT_CAP, select
from tprag.split.store import PROCESSED_DIR, ROOT, load_chunks

CHROMA_DIR = ROOT / "data" / "index" / "chroma"
REPRESENTATIONS = ("r1",)
BATCH = 1000  # por debajo del máximo de Chroma (~5.400)


@dataclass
class IndexConfig:
    chunks: str = "s3-1024"  # carpeta de data/processed/<modo>/
    console_mode: str = "outputs_as_comments"
    model: str = "e1"
    representation: str = "r1"
    cap: int | None = DEFAULT_CAP
    strategy: str = "round_robin"
    packages: list[str] | None = field(default=None)  # None = todas

    def collection_name(self) -> str:
        """`s3-1024__e1__r1__cap150`; con otra estrategia `cap150-random`; sin tope `nocap`; con un subconjunto de
        librerías se agrega `__sub-<hash>` (la lista va en el reporte)."""
        cap = "nocap" if self.cap is None else f"cap{self.cap}"
        if self.cap is not None and self.strategy != "round_robin":
            cap += f"-{self.strategy.replace('_', '')}"
        name = f"{self.chunks}__{self.model}__{self.representation}__{cap}"
        if self.console_mode != "outputs_as_comments":
            name += f"__{self.console_mode}"
        if self.packages:
            name += "__sub-" + hashlib.sha1(",".join(sorted(self.packages)).encode()).hexdigest()[:8]
        return name


def select_chunks(cfg: IndexConfig) -> list[Chunk]:
    chunks = load_chunks(PROCESSED_DIR / cfg.console_mode / cfg.chunks / "chunks.jsonl")
    if cfg.packages:
        unknown = set(cfg.packages) - {c.package for c in chunks}
        if unknown:
            raise ValueError(f"librerías sin chunks: {sorted(unknown)}")
        chunks = [c for c in chunks if c.package in cfg.packages]
    return select(chunks, cap=cfg.cap, strategy=cfg.strategy)


def to_document(c: Chunk) -> Document:
    """Chroma solo acepta metadata escalar: `section_path` va como el prefijo y `code_langs` separado por comas."""
    metadata = {
        "package": c.package, "category": c.category, "kind": c.kind, "source_path": c.source_path,
        "doc_id": c.doc_id, "prefix": c.prefix, "section_id": c.section_id, "start": c.start, "end": c.end,
        "n_tokens": c.n_tokens, "n_code_blocks": c.n_code_blocks, "code_langs": ",".join(c.code_langs),
        "prose_chars": c.prose_chars,
    }
    return Document(id=c.id, page_content=c.embed_text(), metadata=metadata)


def chunk_text(doc: Document) -> str:
    """El chunk sin el prefijo (lo que recibe el generador)."""
    return doc.page_content.removeprefix(doc.metadata["prefix"] + "\n\n")


def open_collection(name: str, embeddings: Embeddings, persist_directory: Path = CHROMA_DIR) -> Chroma:
    return Chroma(collection_name=name, embedding_function=embeddings, persist_directory=str(persist_directory),
                  collection_configuration={"hnsw": {"space": "cosine"}})


def build(name: str, docs: list[Document], embeddings: Embeddings, persist_directory: Path = CHROMA_DIR) -> Chroma:
    """Crea (o vuelve a crear) la colección `name` con `docs`. Con el caché, rehacerla no vuelve a pagar."""
    ids = [d.id for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError("hay ids de chunk repetidos")
    store = open_collection(name, embeddings, persist_directory)
    store.reset_collection()
    for i in range(0, len(docs), BATCH):
        store.add_documents(docs[i:i + BATCH], ids=ids[i:i + BATCH])
    return store
