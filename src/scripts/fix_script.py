"""Corrige las dependencias deprecadas o inexistentes de un script con el sistema final (`tprag.generate.system`: la
Variante 7 del Ej. 4, la misma que corre la interfaz). No tiene opciones de configuración. USA LA API (el modelo de
chat solo si algún import problemático pasa el umbral; un reintento, si hace falta, se paga aparte).

Uso:
    uv run --group local python src/scripts/fix_script.py mi_script.py

Escribe al lado del script <nombre>.fixed.py (el script corregido; igual al original si no hubo cambios) y
<nombre>.report.json (una entrada por import: estado del validador, score del retrieval, decisión, paquete de
reemplazo, motivo y fuentes).
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from tprag.generate import system
from tprag.generate.context import load_reranker
from tprag.generate.pipeline import prepare
from tprag.split.store import ROOT
from tprag.validate.pypi import CachedPyPI


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    args = parser.parse_args()

    store = system.open_index()
    if store._collection.count() == 0:  # open_collection crea la colección vacía si no existe
        print(f"la colección {system.COLLECTION} no existe o está vacía: armarla con build_index.py")
        return 1

    load_dotenv(ROOT / ".env")
    code = args.script.read_text(encoding="utf-8")
    try:
        prepared = prepare(code, store, CachedPyPI(), system.threshold(), system.BUDGET,
                           load_reranker(system.RERANKER))
    except SyntaxError as e:
        print(f"{args.script}: el script no parsea ({e})")
        return 1
    result = system.correct(prepared, system.chat_llm())

    fixed = args.script.with_name(f"{args.script.stem}.fixed.py")
    report = args.script.with_name(f"{args.script.stem}.report.json")
    fixed.write_text(result.code, encoding="utf-8")
    report.write_text(json.dumps({"imports": result.report, "edit_errors": result.edit_errors}, indent=2,
                                 ensure_ascii=False) + "\n", encoding="utf-8")
    if result.retry is not None:
        print("la primera corrección no pasó la post-validación: " +
              ("el reintento sigue mal y el sistema se abstiene" if result.retry["remaining"] else
               "el reintento la corrigió"))
    for e in result.report:
        if "decision" in e:
            target = f" -> {e['replacement_package']}" if e["replacement_package"] else ""
            print(f"{e['module']} ({e['status']}): {e['decision']}{target}")
    for msg in result.edit_errors:
        print(f"no se aplicaron los cambios: {msg}")
    print(f"-> {fixed}\n-> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
