"""S0-S3 y la metadata de los chunks. Los tokens se cuentan como palabras, para no depender de tiktoken."""

import pytest
from langchain_core.documents import Document

from tprag.split import splitters
from tprag.split.markdown import parse
from tprag.split.splitters import SPLITTERS, SplitConfig, split_document


@pytest.fixture(autouse=True)
def word_tokens(monkeypatch):
    monkeypatch.setattr(splitters, "token_counter", lambda name: lambda text: len(text.split()))


def doc(text: str, path: str = "docs/advanced/clients.md") -> Document:
    return Document(page_content=text, metadata={"id": "d1", "package": "httpx", "category": "http",
                                                 "source_path": path, "kind": "guide"})


def code(lines: int, word: str = "x") -> str:
    return "``` python\n" + "\n".join(f"{word}{i} = {i}" for i in range(lines)) + "\n```"


def prose(words: int, word: str = "word") -> str:
    return " ".join([word] * words)


def code_spans(text: str) -> list[tuple[int, int]]:
    return [(b.start, b.end) for b in parse(text).blocks if b.kind == "code"]


def cuts_code(text: str, spans) -> bool:
    return any(b0 < s < b1 or b0 < e < b1 for s, e in spans for b0, b1 in code_spans(text))


USAGE = f"""## Usage

The recommended way is a context manager:

{code(2)}

Alternatively, close it:

{code(1, "y")}

## Making requests

Once you have a client, send requests. For example:

{code(2, "z")}

These methods accept the same arguments. For example, headers:

{code(2, "w")}

Trailing note about the last example.
"""


def test_s3_small_pairs_merge_within_section():
    chunks = split_document(doc(USAGE), "s3", SplitConfig(max_tokens=1024))
    assert len(chunks) == 2  # una por sección: los pares chicos se juntan pero no se cruzan secciones
    usage, requests = chunks
    assert usage.text.startswith("The recommended way") and usage.text.endswith("```")
    assert "Alternatively" in usage.text and "Once you have" not in usage.text
    assert requests.text.endswith("Trailing note about the last example.")  # texto final → último par
    assert usage.section_path == ["httpx", "advanced/clients", "Usage"]
    assert usage.prefix == "httpx > advanced/clients > Usage"
    assert usage.embed_text().startswith("httpx > advanced/clients > Usage\n\nThe recommended way")
    assert USAGE[usage.start:usage.end] == usage.text


def test_s3_text_between_blocks_goes_with_next_block():
    text = f"## S\n\n{prose(30, 'a')}\n\n{code(3)}\n\n{prose(30, 'b')}\n\n{code(3, 'y')}\n"
    chunks = split_document(doc(text), "s3", SplitConfig(max_tokens=50))
    assert len(chunks) == 2
    assert chunks[0].text.startswith("a a") and chunks[0].text.endswith("```")
    assert chunks[1].text.startswith("b b") and "y0 = 0" in chunks[1].text


def test_s3_oversized_code_keeps_its_explanation():
    big = code(200)  # ~600 palabras: más que el máximo
    text = (f"## S\n\n{prose(10)}\n\n{code(3)}\n\n{prose(10, 'explains')} big.\n\n{big}\n\n"
            f"{prose(10)}\n\n{code(3, 'y')}\n")
    chunks = split_document(doc(text), "s3", SplitConfig(max_tokens=100))
    assert not cuts_code(text, [(c.start, c.end) for c in chunks])
    (big_chunk,) = [c for c in chunks if "x199 = 199" in c.text]
    assert big_chunk.n_tokens > 100 and big_chunk.text.startswith("explains")  # la explicación va con su código


def test_s3_long_explanation_keeps_closest_paragraphs_with_code():
    text = f"## S\n\n{prose(40, 'far')}\n\n{prose(40, 'near')}\n\n{code(3)}\n"  # 40 + 40 + 12 > 60
    chunks = split_document(doc(text), "s3", SplitConfig(max_tokens=60))
    assert len(chunks) == 2
    assert chunks[0].text.startswith("far") and chunks[0].n_code_blocks == 0
    assert chunks[1].text.startswith("near") and chunks[1].n_code_blocks == 1 and chunks[1].n_tokens <= 60


def test_s3_optional_minimum():
    # Par A casi llena el máximo; el par C (chico) no entra con él
    text = f"## S\n\n{prose(80, 'a')}\n\n{code(3)}\n\n{prose(3, 'c')}\n\n{code(1, 'y')}\n"
    default = split_document(doc(text), "s3", SplitConfig(max_tokens=95))
    assert len(default) == 2 and default[1].n_tokens < 10  # sin mínimo (default): C queda suelto
    with_min = split_document(doc(text), "s3", SplitConfig(max_tokens=95, min_tokens=10))
    assert len(with_min) == 1 and with_min[0].n_tokens > 95  # con mínimo: se une aunque pase el máximo


def test_s3_section_without_code_is_not_a_use_case():
    text = f"## Why\n\n{prose(40)}\n\n## Usage\n\n{prose(20)} explanation.\n\n{code(2)}\n"
    chunks = split_document(doc(text), "s3")
    assert [c.is_use_case for c in chunks] == [False, True]
    assert chunks[1].code_langs == ["python"] and chunks[1].n_code_blocks == 1


def test_s3_never_cuts_code_with_sections_over_max():
    # S2 sí puede cortar código: subdivide con S1 las secciones que superan el máximo
    text = "\n\n".join(f"## Section {i}\n\n{prose(15)}\n\n{code(20 + i)}" for i in range(6)) + "\n"
    chunks = split_document(doc(text), "s3", SplitConfig(max_tokens=60))
    assert chunks and not cuts_code(text, [(c.start, c.end) for c in chunks])


def test_size_splitter_offsets_with_overlap():
    # El overlap se mide en tokens; las posiciones tienen que seguir siendo exactas (bug de add_start_index)
    text = "\n\n".join(f"Paragraph {i} " + prose(30, f"w{i}") for i in range(20)) + "\n"
    chunks = split_document(doc(text), "s0", SplitConfig(max_tokens=70, overlap=35))
    assert len(chunks) > 3 and all(text[c.start:c.end] == c.text for c in chunks)
    assert all(a.start < b.start < a.end for a, b in zip(chunks, chunks[1:]))  # se solapan


def test_s2_one_chunk_per_section():
    chunks = split_document(doc(USAGE), "s2")
    assert [c.section_path[-1] for c in chunks] == ["Usage", "Making requests"]


@pytest.mark.parametrize("name", ["s0", "s1"])
def test_size_splitters_respect_max_and_offsets(name):
    text = "\n\n".join(f"## Section {i}\n\n{prose(40)}\n\n{code(10)}" for i in range(10)) + "\n"
    chunks = split_document(doc(text), name, SplitConfig(max_tokens=80, overlap=10))
    assert len(chunks) > 1
    assert all(c.n_tokens <= 80 for c in chunks)
    assert all(text[c.start:c.end] == c.text for c in chunks)


def test_registry_and_ids():
    assert set(SPLITTERS) == {"s0", "s1", "s2", "s3"}
    a = split_document(doc(USAGE), "s3")
    b = split_document(doc(USAGE), "s3")
    assert [c.id for c in a] == [c.id for c in b] and len({c.id for c in a}) == len(a)  # determinístico
