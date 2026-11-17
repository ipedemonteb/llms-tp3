"""Flujo del generador (generador.md §2.2): validador → contexto por import problemático → umbral → LLM (o no) →
reemplazos → validador sobre el script corregido → script corregido + reporte.

Iteración (iteracion_ej4.md §3.2), apagada por defecto (`answer(..., retry="edits" | "script")`): si la corrección
tiene problemas que se ven sin el dataset (un reemplazo que no se pudo aplicar, código que no parsea, nombres nuevos sin
definir), se le devuelven al LLM una vez; si después sigue con problemas, el sistema se abstiene y devuelve el script
original. El reintento pide la respuesta en formato "edits" (reemplazos) o "script" (el script completo; necesita un
LLM con el esquema `ScriptOutput`). El primer intento es "edits" (el del sistema sin iteración) salvo con
`first="script"` (la variante uniforme: el script completo también en el primer intento). Con `context_check` (se suma a
la anterior, `__iter4`), también es un problema usar nombres de la API nueva que no aparecen en los fragmentos (la
heurística de Faithfulness, `tprag.eval.faithfulness`). Sin iteración, todo es como antes: un intento en formato
"edits".

Se parte en `prepare` (todo lo que no usa el LLM: sirve para el dry-run) y `finish` (aplica la respuesta). El LLM se
recibe como parámetro: cualquier `(instrucciones, mensaje) -> GeneratorOutput` (en los tests, uno falso).
"""

import ast
import difflib
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from langchain_chroma import Chroma

from tprag.eval.faithfulness import faithfulness, record_context
from tprag.eval.metrics import CONTEXT_BUDGET
from tprag.eval.query import target_refs
from tprag.validate.packages import normalize
from tprag.validate.pypi import Fetch
from tprag.validate.validate import ImportResult, validate

from .context import Reranker, TargetContext, build_context
from .llm import TruncatedOutput
from .names import new_undefined_names
from .prompt import PROMPTS, SYSTEM, retry_message, user_message
from .schema import NO_REPLACEMENT, REPLACED, GeneratorOutput, ScriptOutput

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
    llm_error: str | None = None  # "length": la respuesta se cortó por el tope de salida (el script queda igual)
    llm_raw: str | None = None  # el texto cortado, para ver qué pasó
    output_format: str = "edits"  # "script" si la respuesta fue el script completo (`ScriptOutput`)
    changed_lines: int | None = None  # variante uniforme: líneas del original que cambió el script final
    retry: dict | None = None  # con retry=True y problemas en la primera respuesta: qué pasó (ver `answer`)


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


def _entry(r: ImportResult, ctx: TargetContext | None, decision: dict | None, llm_error: str | None = None) -> dict:
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
    if llm_error == "length":
        return entry | {"decision": NO_REPLACEMENT, "replacement_package": None,
                        "reason": "la respuesta del LLM se cortó por superar el tope de tokens de salida",
                        "sources": []}
    if decision is None:
        return entry | {"decision": NO_REPLACEMENT, "replacement_package": None,
                        "reason": "el LLM no devolvió una decisión para este import", "sources": []}
    keys = ("script_usage", "candidates", "matching_fragment", "decision", "replacement_package", "reason", "sources")
    return entry | {k: decision[k] for k in keys if k in decision}  # `candidates`: solo el primer prompt v2


def script_code(original: str, script: str) -> str:
    """El script que devolvió el LLM en formato "script", con el salto de línea final del original (el LLM suele
    sacarlo); si solo difiere en eso, el original."""
    if script.rstrip("\n") == original.rstrip("\n"):
        return original
    return script.rstrip("\n") + ("\n" if original.endswith("\n") else "")


def changed_lines(original: str, fixed: str) -> int:
    """Líneas del original que el corregido borra o reemplaza (control de que el formato "script" no reescriba de más)."""
    return sum(1 for line in difflib.ndiff(original.splitlines(), fixed.splitlines()) if line.startswith("- "))


def finish(prep: Prepared, output: GeneratorOutput | ScriptOutput | None, llm_error: str | None = None,
           llm_raw: str | None = None) -> FixResult:
    """Arma el script y el reporte. Las decisiones del LLM se asocian por nombre de módulo; los reemplazos solo se
    aplican si el LLM reemplazó al menos uno. Con `llm_error` (se llamó al LLM pero la respuesta no sirve), el script
    queda igual.

    Post-validación: el script corregido vuelve a pasar por el validador, y `replacement_package` es el paquete que
    quedó importado en el código, no el que escribió el LLM (que se guarda en `replacement_package_llm`). Con varios
    imports reemplazados en el mismo script, `new_imports` es la lista de todos los imports nuevos.

    Con `ScriptOutput` (formato "script") no hay reemplazos que aplicar: si el LLM reemplazó algún import, el script es
    el que devolvió.
    """
    decisions = {d.module: d.model_dump() for d in output.imports} if output else {}
    by_module = {c.module: c for c in prep.contexts}
    report = [_entry(r, by_module.get(r.module) if r.to_rag else None, decisions.get(r.module), llm_error)
              for r in prep.results]
    edits = [e.model_dump() for e in output.edits] if isinstance(output, GeneratorOutput) else []
    replaced = any(e.get("decision") == REPLACED for e in report)
    if isinstance(output, ScriptOutput):
        code, errors = (script_code(prep.code, output.script) if replaced else prep.code), []
    else:
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
    return FixResult(code, report, output is not None or llm_error is not None, edits, errors, llm_error, llm_raw,
                     output_format="script" if isinstance(output, ScriptOutput) else "edits")


_EDIT_ERROR = re.compile(r"reemplazo (\d+): el fragmento (?:no está|aparece (\d+) veces)")


def context_text(prep: Prepared) -> str:
    """El texto de los fragmentos que recibió el LLM (títulos y texto), el mismo que usa Faithfulness."""
    return record_context({"id": "", "contexts": [{"chunks": c.chunks} for c in prep.contexts if c.passed]})


def _original_line(original: str, fixed: str, n: int) -> str | None:
    """La línea del original que corresponde a la línea `n` del corregido, si hay una clara: el corregido la cambió
    dentro de un bloque reemplazado del mismo tamaño. None si la línea es nueva, si quedó igual o si no se sabe."""
    a, b = original.splitlines(), fixed.splitlines()
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if j1 <= n - 1 < j2:
            return a[i1 + n - 1 - j1] if tag == "replace" and i2 - i1 == j2 - j1 else None
    return None


def _name_lines(code: str, names: list[str], limit: int = 3) -> list[tuple[int, str]]:
    """Líneas del script (que parsea) donde se usa cada nombre, hasta `limit` por nombre, en orden."""
    lines = code.splitlines()
    found: dict[str, set[int]] = {n: set() for n in names}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Name) and node.id in found:
            found[node.id].add(node.lineno)
    numbers = sorted({n for name in names for n in sorted(found[name])[:limit]})
    return [(n, lines[n - 1].strip()) for n in numbers]


def fix_problems(prep: Prepared, result: FixResult, fmt: str = "edits", unchanged: bool = False,
                 context_check: bool = False, detail: bool = False) -> list[str]:
    """Problemas de una corrección que se ven sin el dataset, redactados para el LLM (el reintento). Solo si el LLM
    reemplazó algún import: una abstención no tiene nada que revisar. `fmt` es el formato que se le pide en el
    reintento: con "script" no se le explica cómo copiar el "antes", porque ya no va a escribir reemplazos. Con
    `unchanged` (solo la variante uniforme, para no cambiar las otras), también es un problema decir "reemplazado" y
    devolver el script sin cambios. Con `context_check`, también usar nombres de la API nueva que no aparecen en los
    fragmentos. Con `detail` (el reintento v2 y v3, `__iter5` y `__iter6`), muestra las líneas: la que no parsea (y la del original que
    le corresponde) y las que usan los nombres sin definir; sin él, los textos de siempre."""
    if not any(e.get("decision") == REPLACED for e in result.report):
        return []
    out = []
    if unchanged and result.output_format == "script" and result.code == prep.code:
        out.append("Decidiste reemplazar al menos un import, pero devolviste el script sin cambios.")
    for error in result.edit_errors:
        i, n = _EDIT_ERROR.match(error).groups()
        if fmt == "script":
            out.append(f"Tu reemplazo {i} no se pudo aplicar: el \"antes\" "
                       + ("no está en el script tal cual." if n is None else f"aparece {n} veces en el script."))
            continue
        out.append(f"Reemplazo {i}: el \"antes\" no está en el script tal cual. Copialo exacto de la sección "
                   "\"Script\": líneas consecutivas, con su sangría, sin saltear ni agregar líneas. Si las líneas a "
                   "cambiar no están juntas, hacé un reemplazo por cada lugar." if n is None else
                   f"Reemplazo {i}: el \"antes\" aparece {n} veces en el script. Agregá líneas vecinas para que "
                   "aparezca una sola vez.")
    if result.code != prep.code:
        try:
            ast.parse(result.code)
        except SyntaxError as e:
            problem = f"El script corregido no es Python válido: línea {e.lineno}: {e.msg}."
            lines = result.code.splitlines()
            if detail and e.lineno and 0 < e.lineno <= len(lines):
                problem += f"\n  Tu línea {e.lineno}: {lines[e.lineno - 1].strip()}"
                if (before := _original_line(prep.code, result.code, e.lineno)) is not None:
                    problem += f"\n  En el original: {before.strip()}"
                problem += "\n  Copiá sin cambios las líneas que no hace falta migrar."
            out.append(problem)
        else:
            if names := new_undefined_names(prep.code, result.code):
                problem = ("El script corregido usa nombres que no están definidos ni importados: "
                           f"{', '.join(f'`{n}`' for n in names)}. Migrá también esos usos a la librería nueva, con la "
                           "API que aparece en los fragmentos.")
                if detail and (where := _name_lines(result.code, names)):
                    problem += "\n  Dónde los usa tu script:" + "".join(f"\n  - línea {n}: {t}" for n, t in where)
                out.append(problem)
            if context_check and (unfaithful := faithfulness(prep.code, result.code, context_text(prep))["unfaithful"]):
                out.append("El script corregido usa nombres de la librería nueva que no aparecen en los fragmentos: "
                           f"{', '.join(f'`{n}`' for n in unfaithful)}. Usá solo la API que aparece en los fragmentos; "
                           "si no alcanza para hacer la migración, respondé \"sin_reemplazo_confiable\".")
    return out


def abstain(prep: Prepared, left: list[str]) -> FixResult:
    """La corrección siguió con problemas después del reintento: el script original y "sin reemplazo confiable" en los
    imports que llegaron al LLM, con el motivo."""
    result = finish(prep, None)
    result.llm_called = True
    reason = "la corrección no pasó la post-validación después de un reintento: " + " ".join(left)
    for e in result.report:
        if e.get("decision") == NO_REPLACEMENT and e.get("reason", "").startswith("el LLM no devolvió"):
            e["reason"] = reason
    return result


def _ask(prep: Prepared, llm: LLM, system: str, user: str) -> FixResult:
    try:
        return finish(prep, llm(system, user))
    except TruncatedOutput as e:
        return finish(prep, None, llm_error="length", llm_raw=e.raw)


def usages(prep: Prepared) -> dict[str, list[tuple[int, str]]]:
    """Por cada import que llegó al LLM, las líneas del script que lo usan (el import y sus usos, las mismas de la
    consulta A2 del retrieval), con su número."""
    lines = prep.code.splitlines()
    out = {}
    for ctx in prep.contexts:
        if ctx.passed:
            numbers = sorted({n for r in target_refs(prep.code, ctx.module) for n in r.context_lines})
            out[ctx.module] = [(n, lines[n - 1].strip()) for n in numbers if lines[n - 1].strip()]
    return out


def first_prompt(prep: Prepared, fmt: str = "edits", version: str = "v1") -> tuple[str, str] | None:
    """El prompt del primer intento en formato `fmt` ("edits" y "v1": el de siempre, `prep.prompt`). `version`: la del
    primer prompt (`prompt.PROMPTS`; "v2" es `__iter5`)."""
    return (PROMPTS[version][fmt], prep.user) if prep.user is not None else None


def retry_prompt(prep: Prepared, first: FixResult, fmt: str = "edits", uniform: bool = False,
                 context_check: bool = False, version: str = "v1") -> tuple[str, str] | None:
    """El prompt del reintento en formato `fmt` para esta primera respuesta, o None si no hay problemas (no se
    reintenta). Con `uniform` (la primera respuesta fue el script completo), en "Tu respuesta anterior" va ese script."""
    found = fix_problems(prep, first, fmt, unchanged=uniform, context_check=context_check, detail=version != "v1")
    if not found:
        return None
    previous = first.code if uniform else None
    return PROMPTS[version][fmt], retry_message(prep.user, first.edits, found, fmt, usages(prep) if fmt == "script" else None,
                                       previous)


def answer(prep: Prepared, llm: LLM, retry: str | None = None, retry_llm: LLM | None = None,
           first: str = "edits", context_check: bool = False, version: str = "v1") -> FixResult:
    """Llama al LLM si algún import pasó el umbral y arma el resultado; una respuesta cortada por el tope de salida
    deja el script igual (`llm_error="length"`) en vez de cortar todo.

    Con `retry` (la iteración, "edits" o "script"): si la primera respuesta tiene `fix_problems`, un reintento con esos
    problemas, pidiendo la respuesta en ese formato (con "script", a `retry_llm`, que devuelve `ScriptOutput`); si el
    reintento también los tiene, `abstain`. `FixResult.retry` guarda el formato, los problemas de la primera respuesta,
    sus reemplazos, los que quedaron (vacío si el reintento salió bien o el LLM se abstuvo) y, si el reintento quedó,
    cuántas líneas del original cambió.

    Con `first="script"` (variante uniforme): el primer intento también pide el script completo, a `llm`, que en ese
    caso devuelve `ScriptOutput`; el reintento es "script" y se le muestra el script anterior. `context_check`: ver
    `fix_problems`. Con `version="v2"` o `"v3"` (`__iter5`, `__iter6`), el primer intento y el reintento usan ese
    primer prompt (`prompt.PROMPTS`) y el reintento muestra las líneas de los problemas; los LLM tienen que devolver
    los esquemas `*V2`."""
    if prep.prompt is None:
        return finish(prep, None)
    uniform = first == "script"
    result = _ask(prep, llm, *first_prompt(prep, first, version))
    if retry is not None and (prompt := retry_prompt(prep, result, retry, uniform, context_check, version)) is not None:
        result = _retry(prep, result, prompt, retry, retry_llm if retry == "script" else llm, uniform, context_check,
                        version != "v1")
    if uniform:
        result.changed_lines = changed_lines(prep.code, result.code)
    return result


def _retry(prep: Prepared, first_result: FixResult, prompt: tuple[str, str], retry: str, llm: LLM,
           uniform: bool, context_check: bool = False, detail: bool = False) -> FixResult:
    """El reintento de `answer`: la segunda respuesta si no tiene problemas; si no, `abstain`."""
    second = _ask(prep, llm, *prompt)
    left = fix_problems(prep, second, retry, unchanged=uniform, context_check=context_check, detail=detail)
    result = abstain(prep, left) if left else second
    result.retry = {"format": retry, "problems": fix_problems(prep, first_result, retry, unchanged=uniform,
                                                             context_check=context_check, detail=detail),
                    "first_edits": first_result.edits, "remaining": left}
    if uniform:
        result.retry["first_code"] = first_result.code
    if retry == "script" and not left:
        result.retry["changed_lines"] = changed_lines(prep.code, result.code)
    return result


def fix(code: str, store: Chroma, fetch: Fetch, llm: LLM, threshold: float, budget: int = CONTEXT_BUDGET,
        reranker: Reranker | None = None) -> FixResult:
    return answer(prepare(code, store, fetch, threshold, budget, reranker), llm)
