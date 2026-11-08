"""Edición del dataset de evaluación (dataset_eval.md §3.2): los JSONL de data/eval/ son la única fuente.

`export_items` saca cada ítem a dos archivos editables: `<id>.py` (el script) y `<id>.json` (el resto, con cada gold
reducido a paquete + archivo + número de bloque). `import_items` los vuelve a juntar y recalcula `doc_id` y el texto
de cada gold desde la doc normalizada, numerando los bloques con el parser de los splitters (como `verify.py`).
"""

import json
from pathlib import Path

from tprag.eval.verify import code_blocks


def _gold_ref(g: dict) -> dict:
    return {"package": g["package"], "source_path": g["source_path"], "block": g["block"]}


def export_items(items: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for it in items:
        (out_dir / f"{it['id']}.py").write_text(it["code"], encoding="utf-8")
        meta = {**it, "code": None,
                "targets": [{**t, "gold": [_gold_ref(g) for g in t["gold"]]} for t in it["targets"]]}
        (out_dir / f"{it['id']}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
                                                  encoding="utf-8")


def resolve_gold(ref: dict, docs: dict[tuple[str, str], tuple[str, str]]) -> dict:
    """`docs`: (paquete, archivo) → (doc_id, texto normalizado)."""
    key = (ref["package"], ref["source_path"])
    if key not in docs:
        raise ValueError(f"no está en la doc normalizada: {ref['package']}/{ref['source_path']}")
    doc_id, text = docs[key]
    blocks = code_blocks(text)
    if not 0 <= ref["block"] < len(blocks):
        raise ValueError(f"{ref['package']}/{ref['source_path']} tiene {len(blocks)} bloques, no {ref['block']}")
    return {"doc_id": doc_id, **ref, "code": blocks[ref["block"]]}


def import_items(src_dir: Path, docs: dict[tuple[str, str], tuple[str, str]], order: list[str]) -> list[dict]:
    """Ítems de `src_dir`, en el orden de `order` (los ids que ya estaban) y después los nuevos por id."""
    metas = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in src_dir.glob("*.json")}
    ids = [i for i in order if i in metas] + sorted(set(metas) - set(order))
    items = []
    for id_ in ids:
        meta = metas[id_]
        code = (src_dir / f"{id_}.py").read_text(encoding="utf-8")
        targets = [{**t, "gold": [resolve_gold(g, docs) for g in t["gold"]]} for t in meta["targets"]]
        items.append({**meta, "code": code, "targets": targets})
    return items
