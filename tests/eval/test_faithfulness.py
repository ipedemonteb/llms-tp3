"""Faithfulness por heurística (eval/faithfulness.py): sin API ni índice."""

import json

import pytest

from tprag.eval.faithfulness import changed_lines, claims, faithfulness, load_chunk_texts, mentions, record_context

ORIGINAL = """from PyPDF2 import PdfFileReader

def pages(path):
    reader = PdfFileReader(open(path, "rb"))
    return [reader.getPage(i).extractText() for i in range(reader.numPages)]
"""

FIXED = """from pypdf import PdfReader

def pages(path):
    reader = PdfReader(path)
    return [page.extract_text() for page in reader.pages]
"""

CONTEXT = """```python
from pypdf import PdfReader

reader = PdfReader("example.pdf")
page = reader.pages[0]
print(page.extract_text())
```"""


def names(original, fixed, context=""):
    return {c.name for c in claims(original, fixed, context)}


def test_changed_lines():
    assert changed_lines("a\nb\nc\n", "a\nB\nc\nd\n") == {2, 4}


def test_claims_follow_assignments_and_loops():
    assert names(ORIGINAL, FIXED) == {"pypdf", "PdfReader", "pages", "extract_text"}


def test_faithful_when_all_names_in_context():
    out = faithfulness(ORIGINAL, FIXED, CONTEXT)
    assert out["score"] == 1.0 and out["claims"] == 4 and out["unfaithful"] == []


def test_name_not_in_context_is_unfaithful():
    fixed = FIXED.replace("extract_text()", "extract_texts()")
    out = faithfulness(ORIGINAL, fixed, CONTEXT)
    assert out["unfaithful"] == ["extract_texts"] and out["score"] == 3 / 4


def test_whole_word_match():
    assert mentions("reader.pages[0]", "pages")
    assert not mentions("PdfReaderX()", "PdfReader")


def test_unchanged_lines_are_not_claims():
    original = "from PyPDF2 import PdfReader\n\nreader = PdfReader('a.pdf')\nn = len(reader.pages)\n"
    fixed = "from pypdf import PdfReader\n\nreader = PdfReader('a.pdf')\nn = len(reader.pages)\n"
    assert names(original, fixed) == {"pypdf", "PdfReader"}


def test_keyword_arguments_are_claims():
    fixed = FIXED.replace("PdfReader(path)", "PdfReader(path, strict=True)")
    out = faithfulness(ORIGINAL, fixed, CONTEXT)
    assert "strict" in out["unfaithful"]


def test_basic_type_methods_are_not_claims():
    original = "import oldpdf\n\nr = oldpdf.Reader('a')\n"
    fixed = "import pypdf\n\nr = pypdf.PdfReader('a')\nwords = r.pages[0].extract_text().split()\n"
    assert names(original, fixed) == {"pypdf", "PdfReader", "pages", "extract_text"}


def test_stdlib_helper_import_is_ignored_but_stdlib_replacement_counts():
    original = "import pytz\n\ntz = pytz.timezone('America/Buenos_Aires')\n"
    fixed = ("import json\nfrom zoneinfo import ZoneInfo\n\ntz = ZoneInfo('America/Buenos_Aires')\n"
             "print(json.dumps({}))\n")
    context = "from zoneinfo import ZoneInfo\n\nZoneInfo('America/Los_Angeles')"
    assert names(original, fixed, context) == {"zoneinfo", "ZoneInfo"}
    assert faithfulness(original, fixed, context)["score"] == 1.0


def test_no_change_and_syntax_error():
    assert faithfulness(ORIGINAL, ORIGINAL, CONTEXT)["score"] is None
    out = faithfulness(ORIGINAL, "from pypdf import (\n", CONTEXT)
    assert out["score"] is None and out["error"] == "no parsea"


def test_record_context_uses_saved_text_or_lookup():
    record = {"id": "x", "contexts": [{"chunks": [{"id": "c1", "text": "uno"}, {"id": "c2"}]}]}
    assert record_context(record, {"c2": "dos"}) == "uno\n\ndos"
    with pytest.raises(KeyError):
        record_context(record)


def test_load_chunk_texts(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(json.dumps({"id": i, "text": f"texto {i}"}) for i in ("a", "b", "c")) + "\n")
    assert load_chunk_texts(path, {"a", "c"}) == {"a": "texto a", "c": "texto c"}
