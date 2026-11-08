"""Una corrida de la evaluación del retrieval (paso 7, docs/agents/evaluacion.md).

Por cada target de los ítems del split: arma la consulta (`query.py`), busca 50 chunks en la colección de Chroma y
calcula las métricas (`metrics.py`). El resultado son registros por target (el `.targets.jsonl`) y un resumen.
"""

import hashlib
import io
import re
import subprocess
import tokenize

from langchain_chroma import Chroma

from tprag.split.store import ROOT

from .metrics import Span, target_metrics
from .query import QUERIES
from .verify import EVAL_DIR, EVAL_FILES, code_block_spans

N_RESULTS = 50

_WORD = re.compile(r"[a-záéíóúñü]+")
_ES = {"de", "la", "el", "los", "las", "que", "del", "por", "para", "con", "una", "un", "se", "sin", "cada", "si",
       "archivo", "página", "usuario", "hay", "lee", "devuelve"}
_EN = {"the", "of", "and", "to", "for", "with", "from", "is", "this", "that", "each", "if", "file", "page", "user",
       "returns", "read", "an", "in", "on"}


def item_lang(code: str) -> str:
    """`es` o `en` según los comentarios y strings del código (heurística: palabras frecuentes y acentos; opción b de
    evaluacion.md). Se guarda en el detalle para poder revisarla."""
    texts = []
    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING) or tok.type == getattr(tokenize, "FSTRING_MIDDLE", -1):
            texts.append(tok.string.lower())
    words = _WORD.findall(" ".join(texts))
    es = sum(w in _ES for w in words) + sum(any(c in w for c in "áéíóúñ") for w in words)
    en = sum(w in _EN for w in words)
    return "es" if es > en else "en"


def select_items(items: list[dict], split: str) -> list[dict]:
    return [it for it in items if it.get("split") == split]


def build_targets(items: list[dict], query: str) -> list[dict]:
    """Un registro por target, todavía sin resultados de búsqueda."""
    make = QUERIES[query]
    records = []
    for it in items:
        lang = item_lang(it["code"])
        for t in it["targets"]:
            records.append({
                "id": it["id"], "subset": it["subset"], "split": it["split"], "lang": lang, "module": t["module"],
                "acceptable_packages": t["acceptable_packages"], "query": make(it["code"], t["module"]),
            })
    return records


def gold_spans(items: list[dict], docs: dict[str, str]) -> dict[tuple[str, str], list[Span]]:
    """(id del ítem, módulo) → posiciones de sus bloques gold en la doc normalizada. Ubica el bloque por su texto
    (como `verify.check_gold`), así un número de bloque corrido no rompe la métrica; si no está, error."""
    out = {}
    for it in items:
        for t in it["targets"]:
            spans = []
            for g in t["gold"]:
                text = docs.get(g["doc_id"])
                if text is None:
                    raise ValueError(f"{it['id']}: falta el documento {g['doc_id']} (¿corriste normalize_docs.py?)")
                blocks = code_block_spans(text)
                found = [i for i, (_, _, code) in enumerate(blocks) if code == g["code"]]
                if not found:
                    raise ValueError(f"{it['id']}: el gold {g['source_path']} bloque {g['block']} no está en la doc "
                                     "(correr verify_eval.py)")
                i = g["block"] if g["block"] in found else found[0]
                spans.append((g["doc_id"], blocks[i][0], blocks[i][1]))
            out[(it["id"], t["module"])] = spans
    return out


def search(store: Chroma, query: str, k: int = N_RESULTS) -> list[dict]:
    """Chunks más parecidos; Chroma devuelve distancia coseno, se guarda la similitud (1 - distancia)."""
    return [
        {"id": doc.id, "package": doc.metadata["package"], "score": round(1 - distance, 6),
         "doc_id": doc.metadata["doc_id"], "start": doc.metadata["start"], "end": doc.metadata["end"],
         "n_tokens": doc.metadata["n_tokens"]}
        for doc, distance in store.similarity_search_with_score(query, k=k)
    ]


def run(store: Chroma, records: list[dict], spans: dict[tuple[str, str], list[Span]]) -> list[dict]:
    for r in records:
        r["hits"] = search(store, r["query"])
        r["metrics"] = target_metrics(r["hits"], r["acceptable_packages"], spans.get((r["id"], r["module"]), []))
    return records


def dataset_info() -> dict:
    """Commit del repo y sha256 de cada archivo de data/eval/: identifica la versión del dataset (los resultados no se
    versionan y se pasan entre integrantes, evaluacion.md §2.4)."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    files = {name: hashlib.sha256((EVAL_DIR / name).read_bytes()).hexdigest()
             for name in EVAL_FILES if (EVAL_DIR / name).exists()}
    return {"commit": commit, "sha256": files}
