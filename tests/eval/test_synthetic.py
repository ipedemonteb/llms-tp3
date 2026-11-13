from tprag.eval.synthetic import forbidden_names, shared_names

GOLD = """from alembic.config import Config
from alembic import command
cfg = Config("/path/alembic.ini")
command.stamp(cfg, "head", sql=True)
data = load(path)
"""


def test_forbidden_names_api_and_example_data():
    names = forbidden_names(GOLD)
    assert {"Config", "alembic", "command", "stamp", "sql", "/path/alembic.ini", "head"} <= set(names)
    # genéricos, builtins y nombres cortos no identifican a la librería
    assert not {"data", "path", "load", "cfg", "print"} & set(names)


def test_forbidden_names_unparseable_block_uses_regex():
    names = forbidden_names("client.fetch_rows(limit=5)\n...\n  >>> broken(")
    assert {"fetch_rows", "limit", "broken"} <= set(names)


def test_shared_names_only_usage_lines_and_identifiers():
    query = """import os
from quillcfg import Config, command
# head of file
cfg = Config("x")
command.stamp(cfg)
os.stamp = 1  # no es una línea de uso del módulo inventado
"""
    assert shared_names(query, GOLD, "quillcfg") == ["Config", "command", "stamp"]


def test_shared_names_ignores_stdlib_api():
    gold = "import foo\nfoo.mkdir(p)\nfoo.Engine(uuid4())"
    query = "import barlib\nbarlib.mkdir(p)\nbarlib.Motor(uuid4())"
    assert shared_names(query, gold, "barlib") == []
