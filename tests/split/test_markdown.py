"""Parser de estructura del .md normalizado (split/markdown.py)."""

from tprag.split.markdown import parse, path_title

DOC = """# Clients

Intro to clients.

## Usage

The recommended way:

``` python
# not a heading
with httpx.Client() as client:
    ...
```

#### Detail

Small print.

### Closing

  ``` python
  client.close()
  ```
"""


def test_sections_and_blocks():
    p = parse(DOC, "docs/advanced/clients.md")
    assert p.title == "Clients"  # el único H1 y primer título es el título de la página
    assert [s.headings for s in p.sections] == [[], ["Usage"], ["Usage", "Closing"]]
    usage = p.sections[1]
    assert [b.kind for b in usage.blocks] == ["text", "code", "text", "text"]  # el H4 queda como texto
    code = usage.blocks[1]
    assert code.lang == "python" and "# not a heading" in DOC[code.start:code.end]
    assert DOC[code.start:code.end].endswith("```")
    closing = p.sections[2].blocks
    assert [b.kind for b in closing] == ["code"] and "client.close()" in DOC[closing[0].start:closing[0].end]


def test_title_falls_back_to_path():
    p = parse("## Why\n\nText.\n\n## Usage\n\nMore.\n", "docs/advanced/clients.md")
    assert p.title == "advanced/clients"
    assert [s.headings for s in p.sections] == [["Why"], ["Usage"]]
    # dos H1: ninguno es el título de la página; son secciones
    p = parse("# A\n\nx\n\n# B\n\ny\n", "doc/source/guide.rst")
    assert p.title == "guide" and [s.headings for s in p.sections] == [["A"], ["B"]]


def test_path_title():
    assert path_title("docs/advanced/clients.md") == "advanced/clients"
    assert path_title("doc/source/user/quickstart.rst") == "user/quickstart"
    assert path_title("examples/plot_demo.py") == "examples/plot_demo"
    assert path_title("README.md") == "README"


def test_unclosed_fence_and_longer_fences():
    text = "Text.\n\n```` python\n```\nstill code\n````\n\nAfter.\n\n```\nnever closed\n"
    blocks = parse(text).blocks
    assert [b.kind for b in blocks] == ["text", "code", "text", "code"]
    assert "still code" in text[blocks[1].start:blocks[1].end]
    assert text[blocks[3].start:blocks[3].end].endswith("never closed")


def test_document_without_headings():
    p = parse("Just text.\n\n``` python\nx = 1\n```\n", "README.md")
    assert len(p.sections) == 1 and p.sections[0].headings == [] and p.title == "README"
