"""Inclusiones (includes.py), con un corpus en memoria. No usan pandoc ni data/raw."""

import sys
from pathlib import Path

from tprag.normalize import includes
from tprag.normalize.includes import Corpus, included_paths, resolve_include, resolve_includes

FILES = {
    # fastapi
    "docs/en/docs/tutorial/first-steps.md": "The simplest FastAPI file:\n\n{* ../../docs_src/first_steps/t1.py *}\n",
    "docs_src/first_steps/t1.py": "from fastapi import FastAPI\n\napp = FastAPI()\n\n\n@app.get('/')\nasync def root():\n"
                                  "    return {'message': 'Hello'}\n",
    # Sphinx
    "doc/charts/bar.rst": "Bar charts\n==========\n\n.. literalinclude:: bar.py\n   :language: python\n"
                          "   :start-after: # start\n   :end-before: # end\n\nDone.\n",
    "doc/charts/bar.py": "import openpyxl\n# start\nchart = BarChart()\nws.add_chart(chart)\n# end\nwb.save('x.xlsx')\n",
    "doc/api.rst": ".. literalinclude:: ../examples/demo.py\n   :pyobject: Demo.run\n",
    "examples/demo.py": "class Demo:\n    def setup(self):\n        pass\n\n    @staticmethod\n    def run():\n"
                        "        return 1\n",
    "doc/index.rst": "Intro.\n\n.. include:: parts/install.rst\n\nEnd.\n",
    "doc/parts/install.rst": "Install with pip.\n\n.. include:: cmd.txt\n   :literal:\n",
    "doc/parts/cmd.txt": "pip install openpyxl\n",
    "doc/loop_a.rst": ".. include:: loop_b.rst\n",
    "doc/loop_b.rst": ".. include:: loop_a.rst\n",
    # textual
    "docs/guide/app.md": 'Code:\n\n```python title="app.py"\n--8<-- "docs/examples/app.py"\n```\n\n--8<-- "docs/examples/app.tcss"\n',
    "docs/examples/app.py": "from textual.app import App\n\nApp().run()\n",
}
CORPUS = Corpus(paths=set(FILES), read=FILES.__getitem__, doc_dirs=["doc", "docs"])


def test_fastapi_include_is_replaced_by_the_code():
    res = resolve_includes(FILES["docs/en/docs/tutorial/first-steps.md"], "md",
                           "docs/en/docs/tutorial/first-steps.md", CORPUS)
    assert "```python\nfrom fastapi import FastAPI" in res.text and "{*" not in res.text
    assert res.used == {"docs_src/first_steps/t1.py"}


def test_fastapi_line_range():
    res = resolve_includes("{* ../../docs_src/first_steps/t1.py ln[1:3] hl[3] *}\n", "md",
                           "docs/en/docs/tutorial/x.md", CORPUS)
    assert "app = FastAPI()" in res.text and "async def root" not in res.text


def test_literalinclude_with_start_after_end_before():
    res = resolve_includes(FILES["doc/charts/bar.rst"], "rst", "doc/charts/bar.rst", CORPUS)
    assert ".. code-block:: python\n\n   chart = BarChart()\n   ws.add_chart(chart)\n" in res.text
    assert "wb.save" not in res.text and "Done." in res.text


def test_literalinclude_pyobject():
    res = resolve_includes(FILES["doc/api.rst"], "rst", "doc/api.rst", CORPUS)
    assert "   @staticmethod\n   def run():\n       return 1" in res.text and "setup" not in res.text


def test_nested_rst_include_relative_to_included_file():
    res = resolve_includes(FILES["doc/index.rst"], "rst", "doc/index.rst", CORPUS)
    assert "Install with pip." in res.text and "   pip install openpyxl" in res.text
    assert res.used == {"doc/parts/install.rst", "doc/parts/cmd.txt"}


def test_include_cycle_stops():
    res = resolve_includes(FILES["doc/loop_a.rst"], "rst", "doc/loop_a.rst", CORPUS)
    assert any("límite" in u for u in res.unresolved)


def test_snippet_inside_code_fence_and_unresolved():
    res = resolve_includes(FILES["docs/guide/app.md"], "md", "docs/guide/app.md", CORPUS)
    assert '```python title="app.py"\nfrom textual.app import App\n\nApp().run()\n```' in res.text
    assert res.unresolved == ["docs/guide/app.md -> docs/examples/app.tcss"]  # no está en el corpus
    assert "--8<--" not in res.text


def test_myst_literalinclude():
    res = resolve_includes("```{literalinclude} ../examples/demo.py\n:language: python\n:lines: 1-3\n```\n", "md",
                           "doc/page.md", CORPUS)
    assert res.text == "```python\nclass Demo:\n    def setup(self):\n        pass\n```\n"


def test_included_paths():
    pages = {p: t for p, t in FILES.items() if p.endswith((".md", ".rst"))}
    found = included_paths(pages, CORPUS)
    assert {"docs_src/first_steps/t1.py", "doc/charts/bar.py", "examples/demo.py", "docs/examples/app.py"} <= found


def test_resolve_include_absolute_sphinx_path():
    # pytest: la raíz de la doc (conf.py) es doc/en, debajo de la carpeta de docs del manifest (doc)
    assert resolve_include("/example/xfail_demo.py", "doc/en/how-to/skipping.rst",
                           {"doc/en/example/xfail_demo.py"}, ["doc"]) == "doc/en/example/xfail_demo.py"


def test_fetch_script_uses_the_same_functions():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "scripts"))
    import fetch_raw_docs
    assert fetch_raw_docs.include_targets is includes.include_targets
    assert fetch_raw_docs.resolve_include is includes.resolve_include
