"""Métricas de la fase 0 (split/metrics.py)."""

from tprag.split.chunk import Chunk
from tprag.split.markdown import parse
from tprag.split.metrics import border_sample, cut_code_blocks, oversized, size_stats

TEXT = "Intro text.\n\n``` python\nx = 1\ny = 2\n```\n\nMore.\n"


def chunk(start: int, end: int, n_tokens: int = 10, section: str = "s") -> Chunk:
    return Chunk(id=f"{start}", package="p", category="c", kind="guide", source_path="a.md", doc_id="d",
                 section_path=["p", "a", "S"], section_id=section, text=TEXT[start:end], start=start, end=end,
                 n_tokens=n_tokens, n_code_blocks=1, code_langs=["python"], prose_chars=10, is_use_case=False)


def test_cut_code_blocks():
    parsed = {"d": parse(TEXT)}
    code_start, code_end = TEXT.index("```"), TEXT.rindex("```") + 3
    whole = [chunk(0, code_end), chunk(code_end, len(TEXT))]  # el borde cae justo en el final del bloque
    assert cut_code_blocks(whole, parsed) == {"code_blocks": 1, "cut": 0, "pct": 0.0}
    middle = TEXT.index("y = 2")
    split = [chunk(0, middle), chunk(middle, len(TEXT))]
    assert cut_code_blocks(split, parsed)["cut"] == 1
    assert code_start < middle < code_end


def test_size_stats_and_oversized():
    chunks = [chunk(0, 5, n) for n in (10, 60, 900, 1500)]
    s = size_stats(chunks)
    assert s["chunks"] == 4 and s["max"] == 1500
    assert s["histogram"]["0-50"] == 1 and s["histogram"]["50-100"] == 1 and s["histogram"]["1024-2048"] == 1
    assert oversized(chunks, 1024)["chunks"] == 1


def test_border_sample_only_within_section():
    chunks = [chunk(0, 10, section="s1"), chunk(10, 20, section="s1"), chunk(20, 30, section="s2")]
    md = border_sample(chunks, n=5)
    assert "1 de 1" in md and md.count("- [ ] Bien asignado") == 1
