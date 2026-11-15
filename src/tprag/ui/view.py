"""Lo que muestra la interfaz, sin Streamlit: tablas, diff, reporte y ejemplos. Funciones puras (se testean solas)."""

import difflib
import json
from pathlib import Path

from tprag.generate.context import TargetContext
from tprag.generate.pipeline import FixResult
from tprag.generate.schema import REPLACED
from tprag.validate.validate import ImportResult

STATUS = {  # estado del validador → (etiqueta, va al RAG)
    "stdlib": ("biblioteca estándar", False),
    "ok": ("existe en PyPI", False),
    "deprecated": ("deprecado", True),
    "not_found": ("no existe en PyPI", True),
    "unknown": ("sin datos de PyPI", False),
}
EXAMPLE_SUBSETS = ("manual_deprecated", "negative")  # scripts realistas escritos a mano (el sintético es más fácil)


def import_rows(results: list[ImportResult]) -> list[dict]:
    """Una fila por import: qué dijo el validador (el preprocesamiento: decide qué va al RAG)."""
    return [{"línea": r.line, "import": r.module, "paquete": r.package or "-",
             "estado": STATUS.get(r.status, (r.status, False))[0],
             "va al RAG": "sí" if r.to_rag else "no",
             "motivo": r.reason or "",
             "advertencias": ", ".join(w["code"] for w in r.warnings)} for r in results]


def target_rows(contexts: list[TargetContext], threshold: float) -> list[dict]:
    """Una fila por import problemático: confianza del retrieval contra el umbral y qué llega al LLM."""
    rows = []
    for c in contexts:
        packages = list(dict.fromkeys(ch["package"] for ch in c.chunks))
        rows.append({"import": c.module, "estado": STATUS.get(c.status, (c.status, False))[0],
                     "confianza": None if c.score is None else round(c.score, 4), "umbral": round(threshold, 4),
                     "pasa": "sí" if c.passed else "no",
                     "fragmentos al LLM": len(c.chunks),
                     "tokens de contexto": sum(ch.get("n_tokens", 0) for ch in c.chunks),
                     "librerías en el contexto": ", ".join(packages) or "-"})
    return rows


def decision_rows(report: list[dict]) -> list[dict]:
    """Una fila por import que pasó por el generador (los que tienen decisión). El reemplazo es el paquete que quedó
    importado (post-validación); si es de la biblioteca estándar no hay paquete, y se muestra el que nombró el LLM."""
    return [{"import": e["module"], "decisión": "reemplazado" if e["decision"] == REPLACED else "sin reemplazo confiable",
             "reemplazo": e.get("replacement_package") or e.get("replacement_package_llm") or "-",
             "motivo": e.get("reason", ""),
             "fuentes": len(e.get("sources", []))} for e in report if "decision" in e]


def unified_diff(original: str, fixed: str, name: str = "script.py") -> str:
    """Diff unificado del script original contra el corregido ("" si no cambió)."""
    lines = difflib.unified_diff(original.splitlines(keepends=True), fixed.splitlines(keepends=True),
                                 fromfile=f"a/{name}", tofile=f"b/{name}")
    return "".join(line if line.endswith("\n") else line + "\n" for line in lines)


def report_json(result: FixResult) -> str:
    """El mismo reporte que escribe `fix_script.py` (`<script>.report.json`)."""
    return json.dumps({"imports": result.report, "edit_errors": result.edit_errors}, indent=2,
                      ensure_ascii=False) + "\n"


def fixed_name(name: str) -> str:
    stem = name[:-3] if name.endswith(".py") else name
    return f"{stem}.fixed.py"


def load_examples(eval_dir: Path, subsets: tuple[str, ...] = EXAMPLE_SUBSETS) -> dict[str, dict]:
    """Scripts del dataset de evaluación para probar la app: id → {"task", "code", "subset", "split"}."""
    out = {}
    for path in sorted(eval_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if item.get("subset") in subsets:
                    out[item["id"]] = {k: item.get(k) for k in ("task", "code", "subset", "split")}
    return out
