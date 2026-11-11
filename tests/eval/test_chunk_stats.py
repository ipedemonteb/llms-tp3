"""Estadísticas de chunks (eval/chunk_stats.py): sin embeddings."""

from langchain_core.documents import Document

from tprag.eval.chunk_stats import chunk_stats, cut_code_stats, model_token_stats, quantiles
from tprag.split.chunk import Chunk
from tprag.split.markdown import parse

TEXT = "# T\n\nIntro text.\n\n```python\na = 1\n```\n\nMore text.\n\n```bash\npip install x\n```\n"
DOC = Document(page_content=TEXT, metadata={"id": "d1", "source_path": "a.md"})
PY, SH = [b for b in parse(TEXT, "a.md").blocks if b.kind == "code"]


def chunk(start, end, n_tokens=10, package="p", kind="guide", code=0):
    return Chunk(id=f"{start}-{end}", package=package, category="c", kind=kind, source_path="a.md", doc_id="d1",
                 section_path=[package], section_id="s", text=TEXT[start:end], start=start, end=end,
                 n_tokens=n_tokens, n_code_blocks=code, code_langs=[], prose_chars=0, is_use_case=bool(code))


def test_quantiles():
    assert quantiles([]) == {"n": 0}
    out = quantiles(list(range(1, 101)))
    assert (out["n"], out["median"], out["p90"], out["p99"], out["max"]) == (100, 50, 90, 99, 100)


def test_chunk_stats():
    out = chunk_stats([chunk(0, 10, 100, "p", code=1), chunk(10, 20, 300, "q", "readme")])
    assert out["chunks"] == 2 and out["tokens_total"] == 400 and out["with_code"] == 1 and out["use_cases"] == 1
    assert out["by_package"] == {"p": 1, "q": 1} and out["by_kind"] == {"guide": 1, "readme": 1}


def test_cut_code_counts_blocks_no_chunk_contains_whole():
    whole = [chunk(0, len(TEXT))]
    assert cut_code_stats([DOC], whole)["all"] == {"blocks": 2, "intact": 2, "not_intact_rate": 0.0}
    cut = [chunk(0, PY.start + 5), chunk(PY.start + 5, len(TEXT))]  # parte el bloque Python en dos
    out = cut_code_stats([DOC], cut)
    assert out["python"] == {"blocks": 1, "intact": 0, "not_intact_rate": 1.0}
    assert out["all"]["intact"] == 1  # el bloque bash quedó entero en el segundo chunk
    assert cut_code_stats([DOC], [])["all"]["not_intact_rate"] == 1.0  # ningún chunk lo cubre


def test_overlap_counts_as_intact_if_one_chunk_has_it_whole():
    overlapping = [chunk(0, PY.start + 5), chunk(PY.start - 3, PY.end + 3), chunk(PY.end, len(TEXT))]
    assert cut_code_stats([DOC], overlapping)["python"]["intact"] == 1


def test_model_token_stats():
    out = model_token_stats(["a", "b", "c", "d"], lambda texts: [5, 10, 1600, 20], max_length=1536)
    assert out["over_max"] == 0.25 and out["max"] == 1600 and out["max_length"] == 1536
    assert model_token_stats([], lambda texts: [], 10)["over_max"] is None
