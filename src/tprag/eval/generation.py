"""Evaluación de la generación, nivel 1: chequeos automáticos sin API (generador.md §2.6).

Por ítem: los reemplazos se aplicaron, el script corregido parsea, el validador ya no marca el target y no aparece
ningún import nuevo inexistente, el import nuevo es de un paquete aceptable, la decisión es la esperada (positivos →
reemplazado; negativos → sin reemplazo y el código igual) y las fuentes citadas están entre los fragmentos recibidos.
"""

import ast
import re
from statistics import mean

from tprag.generate.pipeline import FixResult, Prepared, import_key
from tprag.generate.schema import NO_REPLACEMENT, REPLACED
from tprag.validate.packages import normalize
from tprag.validate.pypi import Fetch
from tprag.validate.validate import ImportResult, validate

STDLIB = "python-stdlib"
_SOURCE_PREFIX = re.compile(r"^(-{3}\s*)?(\[\d+\]\s*)?")


def acceptable_import(r: ImportResult, acceptable: list[str]) -> bool:
    """¿El import es de un paquete aceptable? Para la stdlib se mira el módulo (`zoneinfo`, `unittest.mock`): todos
    los chunks de la stdlib son del paquete `python-stdlib` y no alcanza con eso."""
    if r.status == "stdlib":
        names = {r.module} | {f"{r.module}.{n}" for n in r.names}
        modules = [a for a in acceptable if a != STDLIB]
        return any(n == a or n.startswith(a + ".") for n in names for a in modules)
    return r.package is not None and normalize(r.package) in {normalize(a) for a in acceptable}


def _source_ok(source: str, headers: set[str]) -> bool:
    return _SOURCE_PREFIX.sub("", source.strip()) in headers


def check_item(item: dict, prep: Prepared, fix: FixResult, fetch: Fetch) -> dict:
    original = {import_key(r) for r in prep.results}
    try:
        ast.parse(fix.code)
        new_results = validate(fix.code, fetch)
    except SyntaxError:
        new_results = None
    new_imports = [r for r in new_results or [] if import_key(r) not in original]
    hallucinated = [r.module for r in new_imports if r.status == "not_found"]
    headers = {c["header"] for ctx in prep.contexts for c in ctx.chunks}
    sources = [s for e in fix.report for s in e.get("sources", [])]
    by_module = {e["module"]: e for e in fix.report}
    contexts = {c.module: c for c in prep.contexts}

    targets = []
    for t in item["targets"]:
        entry, ctx = by_module.get(t["module"], {}), contexts.get(t["module"])
        expected = REPLACED if t["acceptable_packages"] else NO_REPLACEMENT
        out = {"module": t["module"], "expected": expected, "decision": entry.get("decision"),
               "replacement_package": entry.get("replacement_package"),
               "replacement_package_llm": entry.get("replacement_package_llm"),
               "retrieval_score": ctx.score if ctx else None, "passed_threshold": bool(ctx and ctx.passed)}
        out["decision_ok"] = out["decision"] == expected
        if expected == REPLACED:
            out["target_gone"] = new_results is not None and not any(
                r.module == t["module"] and r.to_rag for r in new_results)
            out["package_ok"] = any(acceptable_import(r, t["acceptable_packages"]) for r in new_imports)
            out["success"] = (out["decision_ok"] and not fix.edit_errors and new_results is not None
                              and out["target_gone"] and out["package_ok"] and not hallucinated)
        else:
            out["success"] = out["decision_ok"] and fix.code == item["code"]
        targets.append(out)

    return {
        "id": item["id"], "subset": item["subset"], "split": item.get("split"), "targets": targets,
        "llm_called": fix.llm_called, "applicable": not fix.edit_errors, "edit_errors": fix.edit_errors,
        "parses": new_results is not None, "hallucinated_imports": hallucinated,
        "sources_ok": all(_source_ok(s, headers) for s in sources),
        "success": all(t["success"] for t in targets),
    }


def _rate(values: list[bool]) -> float | None:
    return round(mean(values), 4) if values else None


def summarize(records: list[dict]) -> dict:
    """Tasas sobre targets positivos (con reemplazo) y negativos; `success` es el éxito de punta a punta."""
    pos = [(r, t) for r in records for t in r["targets"] if t["expected"] == REPLACED]
    neg = [(r, t) for r in records for t in r["targets"] if t["expected"] == NO_REPLACEMENT]
    called = [r for r in records if r["llm_called"]]
    passed_neg = [t for _, t in neg if t["passed_threshold"]]
    return {
        "items": len(records), "llm_calls": len(called),
        "positives": {
            "targets": len(pos),
            "success": _rate([t["success"] for _, t in pos]),
            "passed_threshold": _rate([t["passed_threshold"] for _, t in pos]),
            "decision_ok": _rate([t["decision_ok"] for _, t in pos]),
            "applicable": _rate([r["applicable"] for r, _ in pos]),
            "parses": _rate([r["parses"] for r, _ in pos]),
            "target_gone": _rate([t["target_gone"] for _, t in pos]),
            "package_ok": _rate([t["package_ok"] for _, t in pos]),
        },
        "negatives": {
            "targets": len(neg),
            "success": _rate([t["success"] for _, t in neg]),
            "passed_threshold": len(passed_neg),
            "rejected_by_llm": sum(t["decision"] == NO_REPLACEMENT for t in passed_neg),
        },
        "hallucinated_imports": sum(len(r["hallucinated_imports"]) for r in records),
        "sources_ok": _rate([r["sources_ok"] for r in called]),
    }
