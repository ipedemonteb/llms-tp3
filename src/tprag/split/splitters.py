"""S0-S3 (splitters.md §3.8). Todos tienen la misma interfaz y se registran por nombre en `SPLITTERS`.

Un splitter recibe el documento ya parseado y devuelve dónde cortar: una lista de (start, end) sobre el texto
normalizado. `split_document()` arma los chunks con su metadata (chunk.py), igual para todos. Para sumar uno nuevo
(p. ej. S4, §3.9) alcanza con una función más en `SPLITTERS`.
"""

from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from .chunk import Chunk, make_chunk
from .markdown import Block, ParsedDoc, Section, parse
from .tokens import DEFAULT_TOKENIZER, TokenCounter, token_counter

Span = tuple[int, int]


@dataclass(frozen=True)
class SplitConfig:
    max_tokens: int = 1024
    min_tokens: int = 0  # solo S3; 0 = sin mínimo (default, splitters.md §3.1)
    overlap: int = 128  # solo S0 y S1
    tokenizer: str = DEFAULT_TOKENIZER

    @property
    def count(self) -> TokenCounter:
        return token_counter(self.tokenizer)


# --- S0 y S1: por tamaño (LangChain) -------------------------------------------------------------------------

def _recursive(text: str, cfg: SplitConfig, overlap: int, markdown: bool, offset: int = 0) -> list[Span]:
    kwargs = dict(chunk_size=cfg.max_tokens, chunk_overlap=overlap, length_function=cfg.count)
    splitter = (RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN, **kwargs) if markdown
                else RecursiveCharacterTextSplitter(**kwargs))
    # Las posiciones se buscan acá y no con `add_start_index`: LangChain resta el overlap como caracteres, pero
    # nosotros lo medimos en tokens, y la búsqueda arrancaría después del comienzo real del chunk.
    spans, prev = [], -1
    for chunk in splitter.split_text(text):
        start = text.find(chunk, prev + 1)  # cada chunk empieza después del anterior
        if start < 0:
            raise ValueError(f"no se encontró el chunk en el documento: {chunk[:80]!r}")
        spans.append((offset + start, offset + start + len(chunk)))
        prev = start
    return spans


def s0(parsed: ParsedDoc, cfg: SplitConfig) -> list[Span]:
    """Por tamaño con los separadores por defecto (párrafo, línea, espacio). Línea de base."""
    return _recursive(parsed.text, cfg, cfg.overlap, markdown=False)


def s1(parsed: ParsedDoc, cfg: SplitConfig) -> list[Span]:
    """Por tamaño, prefiriendo cortar en títulos y bordes de bloque de Markdown."""
    return _recursive(parsed.text, cfg, cfg.overlap, markdown=True)


# --- S2: una sección por chunk ---------------------------------------------------------------------------------

def _content_span(section: Section) -> Span | None:
    """Del primer al último bloque de la sección (sin la línea del título, que va en el prefijo)."""
    if not section.blocks:
        return None
    return section.blocks[0].start, section.blocks[-1].end


def s2(parsed: ParsedDoc, cfg: SplitConfig) -> list[Span]:
    """Una sección (H1-H3) por chunk; las que superan el máximo se subdividen con S1 (sin overlap).

    Equivale a `MarkdownHeaderTextSplitter`, pero con las secciones del parser propio: el de LangChain no devuelve
    posiciones, que hacen falta para Span Hit@k.
    """
    spans = []
    for section in parsed.sections:
        span = _content_span(section)
        if span is None:
            continue
        text = parsed.text[span[0]:span[1]]
        if cfg.count(text) <= cfg.max_tokens:
            spans.append(span)
        else:
            spans += _recursive(text, cfg, 0, markdown=True, offset=span[0])
    return spans


# --- S3: casos de uso (propio) ---------------------------------------------------------------------------------

def _units(blocks: list[Block]) -> list[list[Block]]:
    """Pares "texto + bloque de código que sigue" (§3.1, §3.2).

    El texto entre dos bloques va con el siguiente; el texto del final de la sección, con el último par. Una sección
    sin código da una sola unidad de texto (no va a pasar el filtro de casos de uso).
    """
    units, pending = [], []
    for b in blocks:
        pending.append(b)
        if b.kind == "code":
            units.append(pending)
            pending = []
    if pending:
        if units:
            units[-1] += pending
        else:
            units.append(pending)
    return units


def _pack(items: list[list[Block]], measure: Callable[[list[Block]], int], max_tokens: int) -> list[list[Block]]:
    """Junta items consecutivos mientras el texto del grupo no supere el máximo. Un item que solo ya lo supera queda
    solo. Se mide el texto real del grupo (con lo que hay entre los bloques), no la suma de las partes."""
    groups: list[list[Block]] = []
    for item in items:
        if groups and measure(groups[-1] + item) <= max_tokens:
            groups[-1] = groups[-1] + item
        else:
            groups.append(list(item))
    return groups


def _split_oversized(unit: list[Block], measure: Callable[[list[Block]], int], max_tokens: int) -> list[list[Block]]:
    """Un par que supera el máximo (splitters.md §3.11). Cada par tiene a lo sumo un bloque de código.

    - Si el código solo ya supera el máximo, el par queda entero: explicación + código en un chunk más grande que el
      máximo (si no, el código y su explicación terminan en chunks distintos y el filtro descarta los dos).
    - Si no, el código se queda con los párrafos más cercanos que entren; los anteriores van a chunks de texto aparte.
    """
    code_i = next((i for i, b in enumerate(unit) if b.kind == "code"), None)
    if code_i is None:
        return _pack([[b] for b in unit], measure, max_tokens)
    if measure([unit[code_i]]) > max_tokens:
        return [unit]
    head = code_i
    while head > 0 and measure(unit[head - 1:code_i + 1]) <= max_tokens:
        head -= 1
    before = _pack([[b] for b in unit[:head]], measure, max_tokens)
    return before + _pack([unit[head:code_i + 1]] + [[b] for b in unit[code_i + 1:]], measure, max_tokens)


def _merge_small(groups: list[list[Block]], measure: Callable[[list[Block]], int],
                 min_tokens: int) -> list[list[Block]]:
    """Mínimo opcional: un grupo por debajo del mínimo se une al anterior de la sección (o al siguiente, si es el
    primero), aunque el resultado supere el máximo. Después de `_pack`, es el único caso en que un grupo queda chico."""
    out: list[list[Block]] = []
    carry: list[Block] = []  # primer grupo chico, a la espera del siguiente
    for g in groups:
        g = carry + g
        carry = []
        if measure(g) >= min_tokens:
            out.append(g)
        elif out:
            out[-1] = out[-1] + g
        else:
            carry = g
    if carry:
        out.append(carry)  # la sección entera es más chica que el mínimo: queda como está
    return out


def s3(parsed: ParsedDoc, cfg: SplitConfig) -> list[Span]:
    """Pares explicación + código que se juntan hasta el máximo, sin cortar código ni cruzar secciones (§3.1)."""
    text, count = parsed.text, cfg.count
    size = lambda blocks: count(text[blocks[0].start:blocks[-1].end])  # noqa: E731

    spans = []
    for section in parsed.sections:
        if not section.blocks:
            continue
        units = []
        for unit in _units(section.blocks):
            if size(unit) <= cfg.max_tokens:
                units.append(unit)
            else:
                units += _split_oversized(unit, size, cfg.max_tokens)
        groups = _pack(units, size, cfg.max_tokens)
        if cfg.min_tokens:
            groups = _merge_small(groups, size, cfg.min_tokens)
        spans += [(g[0].start, g[-1].end) for g in groups]
    return spans


# --- Registro --------------------------------------------------------------------------------------------------

SPLITTERS: dict[str, Callable[[ParsedDoc, SplitConfig], list[Span]]] = {"s0": s0, "s1": s1, "s2": s2, "s3": s3}


def split_document(doc: Document, splitter: str, cfg: SplitConfig = SplitConfig()) -> list[Chunk]:
    parsed = parse(doc.page_content, doc.metadata["source_path"])
    return [make_chunk(doc, parsed, start, end, cfg.count) for start, end in SPLITTERS[splitter](parsed, cfg)]
