"""A2 (validate/imports.py) sobre un ítem real del dataset de evaluación (data/eval/, versionado)."""

import json

from tprag.eval.verify import EVAL_DIR
from tprag.validate.imports import extract_imports


def test_a2_dep_fuzzywuzzy_match():
    item = next(json.loads(l) for l in (EVAL_DIR / "manual.jsonl").read_text(encoding="utf-8").splitlines()
                if json.loads(l)["id"] == "dep-fuzzywuzzy-match-01")
    lines = item["code"].splitlines()
    ref = next(r for r in extract_imports(item["code"]) if r.module == "fuzzywuzzy")
    assert [lines[i - 1].strip() for i in ref.context_lines] == [
        "from fuzzywuzzy import fuzz, process",
        'resultado = process.extractOne(linea["descripcion"], nombres, scorer=fuzz.token_set_ratio,',
        "score_cutoff=UMBRAL)",  # sentencia de dos líneas: A1 solo tomaba la primera
        "if resultado is None:",
        "nombre, puntaje = resultado",
        'asociadas.append({**linea, "sku": catalogo[nombre], "producto": nombre, "puntaje": puntaje})',
    ]
