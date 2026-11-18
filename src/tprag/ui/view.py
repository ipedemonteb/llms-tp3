"""Lo que muestra la interfaz, sin Streamlit: el resultado, la tabla de decisiones, el diff y el reporte. Funciones puras (se testean solas)."""

import difflib
import json

from tprag.generate.pipeline import FixResult
from tprag.generate.schema import REPLACED


def decision_rows(report: list[dict]) -> list[dict]:
    """Una fila por import que pasó por el generador (los que tienen decisión). El reemplazo es el paquete que quedó
    importado (post-validación); si es de la biblioteca estándar no hay paquete, y se muestra el que nombró el LLM."""
    return [{"Import": e["module"], "Decisión": "reemplazado" if e["decision"] == REPLACED else "sin reemplazo confiable",
             "Reemplazo": e.get("replacement_package") or e.get("replacement_package_llm") or "-",
             "Motivo": e.get("reason", ""),
             "Fuentes": len(e.get("sources", []))} for e in report if "decision" in e]


def outcome(result: FixResult) -> list[tuple[str, str]]:
    """El resultado en una o dos líneas para la persona: [(nivel, texto)], con nivel "success" o "warning"."""
    decided = [e for e in result.report if "decision" in e]
    if not decided:
        return [("success", "No hay dependencias deprecadas ni inexistentes: el script no necesita cambios.")]
    if result.llm_error or result.edit_errors:
        return [("warning", "No se pudo aplicar la corrección: el script queda igual.")]
    out = []
    replaced = [f"`{e['module']}` por `{e.get('replacement_package') or e.get('replacement_package_llm')}`"
                for e in decided if e["decision"] == REPLACED]
    if replaced:
        out.append(("success", "Se reemplazó " + _join(replaced) + "."))
    if kept := [f"`{e['module']}`" for e in decided if e["decision"] != REPLACED]:
        out.append(("warning", "No se encontró un reemplazo confiable para " + _join(kept) + ": queda igual en el "
                    "script."))
    return out


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " y " + parts[-1]


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
