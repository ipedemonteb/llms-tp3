"""Chequeos automáticos del dataset de evaluación (dataset_eval.md §2.3, §3.2).

Por ítem: (1) cada gold sigue siendo el bloque `block` de su documento normalizado, y si no, dónde quedó; (2) el
validador da el `expected_status` de cada target y `ok`/`stdlib` para el resto; (3) el código no nombra los paquetes
de reemplazo (pistas). Solo informa: no reescribe el dataset.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from tprag.split.markdown import parse
from tprag.split.store import ROOT
from tprag.validate.validate import ImportResult

EVAL_DIR = ROOT / "data" / "eval"
EVAL_FILES = ("manual.jsonl", "negatives.jsonl", "synthetic.jsonl")
_FENCE_LINE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*$")


@dataclass
class ItemReport:
    id: str
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # avisos que no fallan (un gold que se corrió y se encontró)


def load_items(eval_dir: Path = EVAL_DIR) -> list[dict]:
    items = []
    for name in EVAL_FILES:
        path = eval_dir / name
        if path.exists():
            items += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return items


def code_blocks(text: str) -> list[str]:
    """Contenido de cada bloque de código, numerados como los splitters (también los indentados dentro de listas)."""
    out = []
    for b in parse(text).blocks:
        if b.kind != "code":
            continue
        lines = text[b.start:b.end].split("\n")[1:]  # sin la línea que abre
        if lines and _FENCE_LINE.match(lines[-1]):
            lines = lines[:-1]  # sin la que cierra (un bloque sin cerrar llega hasta el final)
        out.append("\n".join(lines))
    return out


def check_gold(gold: dict, docs: dict[str, str]) -> tuple[str | None, str | None]:
    """(error, nota). `docs`: doc_id → texto normalizado."""
    where = f"{gold.get('source_path')} bloque {gold['block']}"
    text = docs.get(gold["doc_id"])
    if text is None:
        return f"gold {where}: el documento {gold['doc_id']} no está en la doc normalizada", None
    blocks = code_blocks(text)
    if gold["block"] < len(blocks) and blocks[gold["block"]] == gold["code"]:
        return None, None
    found = [i for i, b in enumerate(blocks) if b == gold["code"]]
    if found:
        return None, f"gold {where} → bloque {found[0]} (se corrió: actualizar `block`)"
    return f"gold {where}: el código guardado no está en el documento (revisar a mano)", None


def check_validator(item: dict, results: list[ImportResult]) -> list[str]:
    errors = []
    by_module = {r.module: r for r in results}
    targets = {t["module"] for t in item["targets"]}
    for t in item["targets"]:
        r = by_module.get(t["module"])
        if r is None:
            errors.append(f"target {t['module']}: el validador no encontró ese import")
        elif r.status != t["expected_status"]:
            errors.append(f"target {t['module']}: da {r.status}, se esperaba {t['expected_status']}")
    for r in results:
        if r.module not in targets and r.status not in ("ok", "stdlib"):
            errors.append(f"import {r.module}: da {r.status} y no es target (falsa alarma o target faltante)")
    return errors


def hint_patterns(package: str) -> list[str]:
    """`google-api-python-client` → también `google_api_python_client` y `google.api.python.client`."""
    return sorted({package, package.replace("-", "_"), package.replace("-", ".")})


def check_hints(item: dict) -> list[str]:
    code = item["code"].lower()
    errors = []
    for t in item["targets"]:
        for pkg in t["acceptable_packages"]:
            for pat in hint_patterns(pkg.lower()):
                if re.search(rf"(?<![\w.]){re.escape(pat)}(?![\w])", code):
                    errors.append(f"pista: el código menciona {pat!r} (paquete de reemplazo)")
                    break
    return errors


def verify_item(item: dict, docs: dict[str, str], results: list[ImportResult] | None) -> ItemReport:
    """`results`: salida del validador para `item['code']`, o None si no parsea."""
    report = ItemReport(item["id"])
    for t in item["targets"]:
        for g in t["gold"]:
            error, note = check_gold(g, docs)
            report.errors += [error] if error else []
            report.notes += [note] if note else []
    if results is None:
        report.errors.append("el código no parsea")
    else:
        report.errors += check_validator(item, results)
    report.errors += check_hints(item)
    return report
