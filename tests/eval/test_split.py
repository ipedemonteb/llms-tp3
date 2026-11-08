"""Split dev/test agrupado por librería de reemplazo (eval/split.py)."""

from collections import Counter

from tprag.eval.split import assign_splits, group_key


def item(id_, module, subset="manual_deprecated", kind=None, status="deprecated"):
    it = {"id": id_, "subset": subset, "targets": [{"module": module, "expected_status": status}]}
    if kind:
        it["negative_kind"] = kind
    return it


def test_group_key_por_reemplazo():
    assert group_key(item("a", "PyPDF2")) == group_key(item("b", "PyPDF4")) == "reemplazo:pypdf"
    assert group_key(item("c", "oauth2client.service_account")) == group_key(item("d", "oauth2client.file"))
    assert group_key(item("e", "jose")) == "reemplazo:authlib"  # import → paquete como el validador
    assert group_key(item("f", "inventadolib", status="not_found")) == "item:f"


ITEMS = (
    [item(f"pdf2-{i}", "PyPDF2") for i in range(4)] + [item(f"pdf4-{i}", "PyPDF4") for i in range(2)]
    + [item(f"fuzz-{i}", "fuzzywuzzy") for i in range(3)] + [item(f"nose-{i}", "nose.tools") for i in range(3)]
    + [item(f"toml-{i}", "toml") for i in range(2)] + [item(f"pytz-{i}", "pytz") for i in range(3)]
    + [item("neg-toml", "toml", "negative", "deprecated_no_replacement")]
    + [item(f"neg-far-{i}", f"inventado{i}", "negative", "far", "not_found") for i in range(4)]
    + [item(f"neg-near-{i}", f"cercano{i}", "negative", "near", "not_found") for i in range(4)]
)


def test_determinista_y_grupos_enteros():
    a, b = assign_splits(ITEMS), assign_splits(list(reversed(ITEMS)))
    assert a == b
    assert len({a[f"pdf2-{i}"] for i in range(4)} | {a[f"pdf4-{i}"] for i in range(2)}) == 1
    assert a["neg-toml"] == a["toml-0"] == a["toml-1"]  # el negativo va con los manuales de su reemplazo


def test_balance_por_conjunto_y_tipo():
    s = assign_splits(ITEMS)
    manual = Counter(s[it["id"]] for it in ITEMS if it["subset"] == "manual_deprecated")
    assert abs(manual["dev"] - manual["test"]) <= 6  # el grupo más grande (pypdf) tiene 6
    for kind in ("far", "near"):
        c = Counter(s[it["id"]] for it in ITEMS if it.get("negative_kind") == kind)
        assert c["dev"] == c["test"] == 2


def test_otra_semilla_puede_cambiar():
    assert any(assign_splits(ITEMS, seed=k) != assign_splits(ITEMS) for k in range(5))
