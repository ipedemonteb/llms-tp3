"""Estructura de un .md normalizado: secciones, párrafos y bloques de código con su posición (splitters.md §3).

Todos los splitters trabajan con posiciones (`start`, `end`) sobre el texto normalizado, para que Span Hit@k pueda
comparar chunks de splitters distintos (estudio §5.2).

- Secciones: las cortan los títulos H1-H3 (como S2, §3.8). Los H4-H6 quedan como texto dentro de la sección.
- Bloques de código: ``` o ~~~, también indentados (dentro de listas). Un `#` dentro del código no es un título.
- Texto: párrafos separados por líneas en blanco (incluye listas, tablas y separadores).
"""

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

SECTION_LEVELS = 3
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*([^`\s]*)")
_PATH_PREFIXES = {"docs", "doc", "source", "en"}  # carpetas que no dicen nada del tema de la página


@dataclass
class Block:
    kind: str  # text | code
    start: int
    end: int
    lang: str = ""


@dataclass
class Section:
    index: int
    headings: list[str]  # títulos desde H1 hasta el de esta sección (sin el título de la página)
    start: int  # desde el título (o el comienzo del documento)
    end: int
    blocks: list[Block] = field(default_factory=list)  # contenido, sin la línea del título


@dataclass
class ParsedDoc:
    text: str
    title: str  # título de la página, o la ruta del archivo si no tiene (§3.3)
    sections: list[Section]

    @property
    def blocks(self) -> list[Block]:
        return [b for s in self.sections for b in s.blocks]

    def section_at(self, offset: int) -> Section:
        for s in reversed(self.sections):
            if s.start <= offset:
                return s
        return self.sections[0]


def path_title(source_path: str) -> str:
    """`docs/advanced/clients.md` → `advanced/clients` (sin extensiones ni carpetas genéricas al principio)."""
    parts = list(PurePosixPath(source_path).parts)
    if not parts:
        return ""
    while len(parts) > 1 and parts[0] in _PATH_PREFIXES:
        parts.pop(0)
    parts[-1] = parts[-1].split(".")[0] or parts[-1]
    return "/".join(parts)


def _lines(text: str):
    """(inicio, fin sin el salto de línea, línea) de cada línea."""
    pos = 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        yield pos, pos + len(content), content
        pos += len(line)


def parse(text: str, source_path: str = "") -> ParsedDoc:
    headings: list[tuple[int, int, str]] = []  # (inicio, nivel, título) de los H1-H3
    blocks: list[Block] = []
    fence: tuple[str, int] | None = None  # (carácter, largo) del bloque abierto
    para_start = para_end = -1

    def close_paragraph():
        nonlocal para_start
        if para_start >= 0:
            blocks.append(Block("text", para_start, para_end))
            para_start = -1

    for start, end, line in _lines(text):
        if fence:
            stripped = line.strip()
            if stripped and set(stripped) == {fence[0]} and len(stripped) >= fence[1]:
                blocks[-1].end = end
                fence = None
            continue
        m = _FENCE.match(line)
        if m:
            close_paragraph()
            fence = (m.group(1)[0], len(m.group(1)))
            blocks.append(Block("code", start, end, m.group(2).lower()))
            continue
        h = _HEADING.match(line)
        if h and len(h.group(1)) <= SECTION_LEVELS:
            close_paragraph()
            headings.append((start, len(h.group(1)), h.group(2)))
            continue
        if not line.strip():
            close_paragraph()
        else:
            if para_start < 0:
                para_start = start
            para_end = end
    close_paragraph()
    if fence:  # bloque sin cerrar: llega hasta el final
        blocks[-1].end = len(text.rstrip("\r\n"))

    # Título de la página: el H1 si es el primer título y el único H1; si no, la ruta del archivo.
    h1s = [h for h in headings if h[1] == 1]
    title_heading = headings[0] if headings and headings[0][1] == 1 and len(h1s) == 1 else None
    title = title_heading[2] if title_heading else path_title(source_path)

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (nivel, título)
    bounds = [(0, None)] + [(h[0], h) for h in headings]
    for i, (start, h) in enumerate(bounds):
        if h is not None and h is not title_heading:
            stack = [s for s in stack if s[0] < h[1]] + [(h[1], h[2])]
        end = bounds[i + 1][0] if i + 1 < len(bounds) else len(text)
        content = [b for b in blocks if start <= b.start < end]
        sections.append(Section(len(sections), [t for _, t in stack], start, end, content))
    # La sección vacía antes del primer título no aporta nada (salvo que el documento no tenga títulos)
    if len(sections) > 1 and not sections[0].blocks:
        sections.pop(0)
        for i, s in enumerate(sections):
            s.index = i
    return ParsedDoc(text, title, sections)
