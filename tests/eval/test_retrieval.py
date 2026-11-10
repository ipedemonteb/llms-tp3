"""Selección de ítems para el retrieval (eval/retrieval.py)."""

from tprag.eval.retrieval import select_items


def test_select_items_drops_targets_outside_packages():
    def item(id, *golds):
        return {"id": id, "split": "dev", "targets": [{"module": m, "gold": [{"package": p}] if p else []}
                                                      for m, p in golds]}

    items = [item("a", ("x", "pandas")), item("b", ("y", "rapidfuzz")), item("c", ("z", None)),
             item("d", ("u", "pandas"), ("v", "rapidfuzz"))]
    kept = select_items(items, "dev", ["pandas"])
    assert [it["id"] for it in kept] == ["a", "c", "d"]  # b no tiene respuesta en el índice; c es negativo
    assert [t["module"] for t in kept[2]["targets"]] == ["u"]  # d conserva solo el target contestable
    assert select_items(items, "dev") == items and select_items(items, "test") == []
