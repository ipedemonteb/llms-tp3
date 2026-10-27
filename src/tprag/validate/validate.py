"""Script → un resultado por import (validador.md §2.1, §2.5).

`status` decide si el import va al RAG (`deprecated`, `not_found`) y es lo único que miran las métricas; `warnings`
solo informa en el reporte.
"""

from dataclasses import dataclass, field
from datetime import date

from .deprecated import DEPRECATED
from .imports import extract_imports
from .packages import candidates, is_stdlib, normalize
from .pypi import Fetch, PackageInfo, PyPIError

INACTIVE = "Development Status :: 7 - Inactive"
DEFAULT_YEARS = 3
TO_RAG = ("deprecated", "not_found")


@dataclass
class ImportResult:
    module: str
    line: int
    names: list[str]
    usages: list[int]
    status: str  # stdlib | ok | deprecated | not_found | unknown
    package: str | None = None  # paquete de PyPI resuelto (o el primer candidato si no existe)
    resolved_by: str | None = None  # manual | pipreqs | variants
    replacement: list[str] | None = None  # solo de la lista manual
    reason: str | None = None
    warnings: list[dict] = field(default_factory=list)

    @property
    def to_rag(self) -> bool:
        return self.status in TO_RAG


def _warnings(info: PackageInfo, years: int, today: date) -> list[dict]:
    out = []
    if info.last_release and (today - date.fromisoformat(info.last_release)).days > years * 365.25:
        out.append({"code": "no_recent_release", "detail": f"última release {info.last_release}"})
    if info.vulnerabilities:
        out.append({"code": "known_vulnerabilities", "detail": ", ".join(info.vulnerabilities)})
    return out


def _check(module: str, fetch: Fetch, years: int, today: date) -> dict:
    names, source = candidates(module)
    for name in names:
        try:
            info = fetch(name)
        except PyPIError as e:
            return {"status": "unknown", "package": name, "resolved_by": source, "reason": str(e)}
        if info is None:
            continue
        result = {"status": "ok", "package": info.name, "resolved_by": source,
                  "warnings": _warnings(info, years, today)}
        if (dep := DEPRECATED.get(normalize(info.name))) is not None:
            result.update(status="deprecated", replacement=dep.replacement, reason=dep.reason)
        elif INACTIVE in info.classifiers:
            result.update(status="deprecated", reason=f"el autor lo marcó como inactivo ({INACTIVE})")
        return result
    return {"status": "not_found", "package": names[0], "resolved_by": source,
            "reason": "no existe en PyPI: candidato a alucinación"}


def validate(source: str, fetch: Fetch, years: int = DEFAULT_YEARS, today: date | None = None) -> list[ImportResult]:
    """Levanta `SyntaxError` si el script no parsea. Cada módulo se consulta una sola vez."""
    today = today or date.today()
    checked = {}
    results = []
    for ref in extract_imports(source):
        base = {"module": ref.module, "line": ref.line, "names": ref.names, "usages": ref.usages}
        if is_stdlib(ref.module):
            results.append(ImportResult(**base, status="stdlib"))
            continue
        if ref.module not in checked:
            checked[ref.module] = _check(ref.module, fetch, years, today)
        results.append(ImportResult(**base, **checked[ref.module]))
    return results
