"""Un chunk y su metadata (splitters.md §3.7), calculados a partir de su posición en el documento normalizado.

Los splitters solo deciden dónde cortar (`start`, `end`); todo lo demás sale de acá, igual para todos.
"""

import hashlib
from dataclasses import dataclass

from langchain_core.documents import Document

from .markdown import ParsedDoc
from .tokens import TokenCounter

MIN_PROSE_CHARS = 80  # filtro de casos de uso (estudio §3.3)
PATH_SEP = " > "


@dataclass
class Chunk:
    id: str
    package: str
    category: str
    kind: str
    source_path: str
    doc_id: str
    section_path: list[str]  # paquete, título de la página, títulos de sección (§3.3)
    section_id: str
    text: str  # sin prefijo: lo que recibe el generador
    start: int
    end: int
    n_tokens: int
    n_code_blocks: int  # bloques que se tocan con el chunk (enteros o no)
    code_langs: list[str]
    prose_chars: int
    is_use_case: bool  # código Python + ≥ 80 caracteres de explicación

    @property
    def prefix(self) -> str:
        return PATH_SEP.join(self.section_path)

    def embed_text(self) -> str:
        """Lo que se embebe (representación R1): prefijo + chunk. Se arma al indexar, no se guarda."""
        return f"{self.prefix}\n\n{self.text}"


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def make_chunk(doc: Document, parsed: ParsedDoc, start: int, end: int, count: TokenCounter) -> Chunk:
    m = doc.metadata
    section = parsed.section_at(start)
    code = [b for b in parsed.blocks if b.kind == "code" and _overlap(start, end, b.start, b.end)]
    prose = sum(_overlap(start, end, b.start, b.end) for b in parsed.blocks if b.kind == "text")
    langs = list(dict.fromkeys(b.lang for b in code))
    text = parsed.text[start:end]
    return Chunk(
        id=_hash(f"{m['package']}/{m['source_path']}:{start}-{end}"),
        package=m["package"], category=m["category"], kind=m["kind"], source_path=m["source_path"],
        doc_id=m["id"],
        section_path=[m["package"], parsed.title, *section.headings],
        section_id=_hash(f"{m['package']}/{m['source_path']}#{section.index}"),
        text=text, start=start, end=end, n_tokens=count(text),
        n_code_blocks=len(code), code_langs=langs, prose_chars=prose,
        is_use_case="python" in langs and prose >= MIN_PROSE_CHARS,
    )
