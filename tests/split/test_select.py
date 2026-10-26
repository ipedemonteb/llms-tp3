"""Filtro y tope por librería (split/select.py)."""

from tprag.split.chunk import Chunk
from tprag.split.select import select


def chunk(package: str, kind: str, path: str, start: int, use_case: bool = True) -> Chunk:
    return Chunk(id=f"{package}/{path}:{start}", package=package, category="c", kind=kind, source_path=path,
                 doc_id=path, section_path=[package], section_id="s", text="t", start=start, end=start + 1,
                 n_tokens=1, n_code_blocks=1, code_langs=["python"], prose_chars=100, is_use_case=use_case)


# lib grande: 2 de quickstart y 3 páginas de guide con 4 casos de uso cada una
BIG = ([chunk("big", "quickstart", "q.md", i) for i in range(2)]
       + [chunk("big", "guide", f"{p}.md", i) for p in "abc" for i in range(4)])
SMALL = [chunk("small", "guide", "x.md", 0), chunk("small", "guide", "x.md", 1, use_case=False)]


def test_filter_and_no_cap():
    out = select(BIG + SMALL, cap=None)
    assert len(out) == 15  # 14 de big + 1 de small (el otro no pasa el filtro)


def test_cap_priority_by_kind_then_round_robin():
    out = select(BIG + SMALL, cap=5)
    big = [c.id for c in out if c.package == "big"]
    assert big == ["big/q.md:0", "big/q.md:1", "big/a.md:0", "big/b.md:0", "big/c.md:0"]
    assert [c.id for c in out if c.package == "small"] == ["small/x.md:0"]  # por debajo del tope: entra todo


def test_cap_in_order_and_random():
    in_order = [c.id for c in select(BIG, cap=5, strategy="in_order")]
    assert in_order[2:] == ["big/a.md:0", "big/a.md:1", "big/a.md:2"]
    r1 = [c.id for c in select(BIG, cap=5, strategy="random", seed=1)]
    r2 = [c.id for c in select(BIG, cap=5, strategy="random", seed=1)]
    assert r1 == r2 and len(r1) == 5  # reproducible con la misma semilla
    assert set(r1[:2]) == {"big/q.md:0", "big/q.md:1"}  # la prioridad por kind se respeta igual
