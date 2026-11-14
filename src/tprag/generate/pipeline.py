"""Flujo del generador (generador.md §2.2): validador → contexto por import problemático → umbral → LLM (o no) →
reemplazos → validador sobre el script corregido → script corregido + reporte.

Se parte en `prepare` (todo lo que no usa el LLM: sirve para el dry-run) y `finish` (aplica la respuesta). El LLM se
recibe como parámetro: cualquier `(instrucciones, mensaje) -> GeneratorOutput` (en los tests, uno falso).
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_chroma import Chroma

from tprag.eval.metrics import CONTEXT_BUDGET
from tprag.validate.packages import normalize
from tprag.validate.pypi import Fetch
from tprag.validate.validate import ImportResult, validate

from .context import Reranker, TargetContext, build_context
from .prompt import SYSTEM, user_message
from .schema import NO_REPLACEMENT, REPLACED, GeneratorOutput

LLM = Callable[[str, str], GeneratorOutput]


@dataclass
class Prepared:
    code: str
    fetch: Fetch  # PyPI, para volver a validar el script corregido en `finish`
    results: list[ImportResult]
    contexts: list[TargetContext]  # uno por módulo problemático (deprecated / not_found)
    user: str | None  # mensaje para el LLM; None si ningún import pasó el umbral (no se lo llama)

    @property
    def prompt(self) -> tuple[str, str] | None:
        return (SYSTEM, self.user) if self.user is not None else None


@dataclass
class FixResult:
    code: str  # el script corregido (o el original si no hubo cambios o falló algún reemplazo)
    report: list[dict]
    llm_called: bool
    edits: list[dict] = field(default_factory=list)
    edit_errors: list[str] = field(default_factory=list)


def prepare(code: str, store: Chroma, fetch: Fetch, threshold: float, budget: int = CONTEXT_BUDGET,
            reranker: Reranker | None = None) -> Prepared:
    """Levanta `SyntaxError` si el script no parsea. Con `reranker`, `threshold` es el del puntaje del reranker."""
    results = validate(code, fetch)
    problems = dict.fromkeys((r.module, r.status) for r in results if r.to_rag)
    contexts = [build_context(store, code, module, status, threshold, budget, reranker=reranker)
                for module, status in problems]
    passed = [c for c in contexts if c.passed]
    return Prepared(code, fetch, results, contexts, user_message(code, passed) if passed else None)


def apply_edits(code: str, edits: list[dict]) -> tuple[str, list[str]]:
    """Aplica los reemplazos en orden. Todo o nada: si alguno no se encuentra o aparece más de una vez, devuelve el
    código original con los errores (un script a medio corregir puede quedar roto)."""
    new, errors = code, []
    for i, e in enumerate(edits, start=1):
        n = new.count(e["antes"]) if e["antes"] else 0
        if n != 1:
            errors.append(f"reemplazo {i}: el fragmento {'no está' if n == 0 else f'aparece {n} veces'} en el script")
            continue
        new = new.replace(e["antes"], e["despues"], 1)
    return (code, errors) if errors else (new, [])


def import_key(r: ImportResult) -> tuple[str, tuple[str, ...]]:
    """Un import se identifica por módulo + nombres: `import unittest` y `from unittest import mock` son distintos."""
    return r.module, tuple(r.names)


def new_imports(before: list[ImportResult], after: list[ImportResult]) -> list[dict]:
    """Imports del script corregido que no estaban en el original, con el paquete que resuelve el validador (para la
    stdlib, el módulo)."""
    old = {import_key(r) for r in before}
    return [{"module": r.module, "package": r.module if r.status == "stdlib" else r.package, "status": r.status}
            for r in after if import_key(r) not in old]


def _matches(claim: str, imp: dict) -> bool:
    """¿Lo que escribió el LLM (`google.oauth2.service_account`, `pypdf`) corresponde a este import nuevo?"""
    c = claim.strip()
    return (normalize(c) == normalize(imp["package"] or "") or c == imp["module"]
            or c.startswith(imp["module"] + ".") or imp["module"].startswith(c + "."))


def resolve_package(claim: str | None, imports: list[dict]) -> str | None:
    """Paquete de reemplazo según el código (post-validación, generador.md §3): el import nuevo que coincide con lo
    que dijo el LLM; si no coincide ninguno y hay un solo paquete nuevo, ese; si no, None."""
    packages = list(dict.fromkeys(i["package"] for i in imports if i["package"]))
    if claim:
        for imp in imports:
            if imp["package"] and _matches(claim, imp):
                return imp["package"]
    return packages[0] if len(packages) == 1 else None


def _entry(r: ImportResult, ctx: TargetContext | None, decision: dict | None) -> dict:
    entry = {"module": r.module, "line": r.line, "status": r.status, "package": r.package,
             "validator_reason": r.reason, "warnings": r.warnings}
    if ctx is None:
        return entry
    entry["retrieval_score"] = ctx.score
    if ctx.reranker:
        entry["reranker"] = ctx.reranker  # retrieval_score es el puntaje del reranker, no el coseno
    if not ctx.passed:
        return entry | {"decision": NO_REPLACEMENT, "replacement_package": None,
                        "reason": "el mejor fragmento recuperado queda por debajo del umbral de confianza",
                        "sources": []}
    if decision is None:
        return entry | {"decision": NO_REPLACEMENT, "replacement_package": None,
                        "reason": "el LLM no devolvió una decisión para este import", "sources": []}
    return entry | {k: decision[k] for k in ("script_usage", "matching_fragment", "decision", "replacement_package",
                                             "reason", "sources")}


def finish(prep: Prepared, output: GeneratorOutput | None) -> FixResult:
    """Arma el script y el reporte. Las decisiones del LLM se asocian por nombre de módulo; los reemplazos solo se
    aplican si el LLM reemplazó al menos uno.

    Post-validación: el script corregido vuelve a pasar por el validador, y `replacement_package` es el paquete que
    quedó importado en el código, no el que escribió el LLM (que se guarda en `replacement_package_llm`). Con varios
    imports reemplazados en el mismo script, `new_imports` es la lista de todos los imports nuevos.
    """
    decisions = {d.module: d.model_dump() for d in output.imports} if output else {}
    by_module = {c.module: c for c in prep.contexts}
    report = [_entry(r, by_module.get(r.module) if r.to_rag else None, decisions.get(r.module))
              for r in prep.results]
    edits = [e.model_dump() for e in output.edits] if output else []
    replaced = any(e.get("decision") == REPLACED for e in report)
    code, errors = apply_edits(prep.code, edits) if replaced else (prep.code, [])
    if replaced:
        try:
            added = new_imports(prep.results, validate(code, prep.fetch)) if code != prep.code else []
        except SyntaxError:
            added = []  # el script corregido no parsea: no hay imports que leer (lo cuenta la evaluación)
        for e in report:
            if e.get("decision") == REPLACED:
                e["replacement_package_llm"] = e["replacement_package"]
                e["replacement_package"] = resolve_package(e["replacement_package_llm"], added)
                e["new_imports"] = added
    return FixResult(code, report, output is not None, edits, errors)


def fix(code: str, store: Chroma, fetch: Fetch, llm: LLM, threshold: float, budget: int = CONTEXT_BUDGET,
        reranker: Reranker | None = None) -> FixResult:
    prep = prepare(code, store, fetch, threshold, budget, reranker)
    return finish(prep, llm(*prep.prompt) if prep.prompt else None)
