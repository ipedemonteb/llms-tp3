"""Armado de la colección (index/build.py), con embeddings falsos y Chroma en un directorio temporal."""

from langchain_core.embeddings import DeterministicFakeEmbedding

from tprag.index.build import IndexConfig, build, chunk_text, open_collection, to_document
from tprag.split.chunk import Chunk


def chunk(i: int, package: str = "httpx") -> Chunk:
    return Chunk(id=f"id{i}", package=package, category="http", kind="guide", source_path="docs/a.md", doc_id="d",
                 section_path=[package, "Clients", "Usage"], section_id="s", text=f"texto {i}\n\n```python\nx\n```",
                 start=i * 10, end=i * 10 + 5, n_tokens=5, n_code_blocks=1, code_langs=["python", "bash"],
                 prose_chars=100, is_use_case=True)


def test_collection_name():
    assert IndexConfig().collection_name() == "s3-1024__e1__r1__cap150"
    assert IndexConfig(cap=None).collection_name() == "s3-1024__e1__r1__nocap"
    assert IndexConfig(strategy="in_order").collection_name() == "s3-1024__e1__r1__cap150-inorder"
    a = IndexConfig(packages=["requests", "httpx"]).collection_name()
    assert a.startswith("s3-1024__e1__r1__cap150__sub-")
    assert a == IndexConfig(packages=["httpx", "requests"]).collection_name()  # no depende del orden


def test_document_metadata_and_text():
    doc = to_document(chunk(1))
    assert doc.page_content.startswith("httpx > Clients > Usage\n\ntexto 1")  # R1: prefijo + chunk
    assert doc.metadata["code_langs"] == "python,bash"
    assert all(isinstance(v, (str, int, float, bool)) for v in doc.metadata.values())
    assert chunk_text(doc) == chunk(1).text


def test_build_and_rebuild(tmp_path):
    emb = DeterministicFakeEmbedding(size=8)
    docs = [to_document(chunk(i)) for i in range(3)]
    build("test__col", docs, emb, tmp_path)
    store = build("test__col", docs[:2], emb, tmp_path)  # rehacer reemplaza, no suma
    assert store._collection.count() == 2
    reopened = open_collection("test__col", emb, tmp_path)
    assert reopened._collection.configuration_json["hnsw"]["space"] == "cosine"
    hit = reopened.similarity_search(docs[1].page_content, k=1)[0]
    assert hit.id == "id1" and hit.metadata["package"] == "httpx"
