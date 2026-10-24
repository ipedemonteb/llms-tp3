"""Normalización de una librería completa, escritura, índice y carga (store.py). Usan pandoc y una carpeta temporal."""

import json

import pytest

from tprag.normalize.prepass import clean_readme
from tprag.normalize.store import load_normalized, mode_dir, normalize_package, write_index, write_package, RAW_DIR

FILES = {
    "README.md": "# fastlib\n\nFast web APIs.\n\n```python\nfrom fastlib import App\n```\n\n"
                 "## Changelog\n\n### 1.0\n\n- First release.\n\n## License\n\nMIT\n",
    "docs/en/docs/tutorial/first-steps.md": "# First Steps\n\nThe simplest file:\n\n{* ../../docs_src/t1.py *}\n\n"
                                            "{* ../../docs_src/missing.py *}\n",
    "docs_src/t1.py": "from fastlib import App\n\napp = App()\n",
    "docs_src/t1_variant.py": "from fastlib import App\n\napp = App(debug=True)\n",  # ninguna página la nombra
    "docs/guide.rst": "Guide\n=====\n\nUse :func:`fastlib.run`::\n\n    >>> fastlib.run()\n    ok\n",
    "examples/plot_demo.py": '"""\nDemo\n====\n\nA demo.\n"""\nimport fastlib\n\nfastlib.run()\n',
}
MANIFEST = {"package": "fastlib", "category": "web_apis", "docs_dirs": ["docs"], "examples_dirs": ["examples"],
            "files": [{"path": "README.md", "kind": "readme", "format": "md"},
                      {"path": "docs/en/docs/tutorial/first-steps.md", "kind": "quickstart", "format": "md"},
                      {"path": "docs_src/t1.py", "kind": "examples", "format": "py"},
                      {"path": "docs_src/t1_variant.py", "kind": "examples", "format": "py"},
                      {"path": "docs/guide.rst", "kind": "guide", "format": "rst"},
                      {"path": "examples/plot_demo.py", "kind": "examples", "format": "py"}]}


@pytest.fixture
def pkg_dir(tmp_path):
    d = tmp_path / "raw" / "fastlib"
    for path, text in FILES.items():
        (d / path).parent.mkdir(parents=True, exist_ok=True)
        (d / path).write_text(text, encoding="utf-8")
    (d / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    return d


def test_normalize_package(pkg_dir):
    results = {e.source_path: (e, md) for e, md in normalize_package(pkg_dir)}
    # 3b: el archivo incluido no es un documento propio; su código aparece en la página que lo incluye
    assert "docs_src/t1.py" not in results
    assert "docs_src/t1_variant.py" not in results  # docs_src: lo que ninguna página incluye tampoco va
    entry, md = results["docs/en/docs/tutorial/first-steps.md"]
    assert "app = App()" in md and entry.included == ["docs_src/t1.py"]
    assert entry.unresolved == ["docs/en/docs/tutorial/first-steps.md -> ../../docs_src/missing.py"]
    # README: sin changelog ni licencia
    readme = results["README.md"][1]
    assert "Fast web APIs." in readme and "First release" not in readme and "MIT" not in readme
    # rst y galería
    entry, md = results["docs/guide.rst"]
    assert "`fastlib.run`" in md and entry.block_langs == ["python"]
    assert results["examples/plot_demo.py"][0].block_langs == ["python"]
    assert all(e.status == "ok" for e, _ in results.values())


def test_write_index_and_load(pkg_dir, tmp_path):
    base = mode_dir("outputs_as_comments", tmp_path / "normalized")
    base.mkdir(parents=True)
    results = normalize_package(pkg_dir)
    write_package(results, "fastlib", base)
    write_index([e for e, _ in results], base, {"fastlib"})
    assert (base / "fastlib" / "docs" / "guide.rst.md").exists()

    docs = load_normalized(out_dir=tmp_path / "normalized")
    assert {d.metadata["source_path"] for d in docs} == {e.source_path for e, _ in results}
    doc = next(d for d in docs if d.metadata["source_path"] == "docs/guide.rst")
    assert doc.metadata["package"] == "fastlib" and doc.metadata["kind"] == "guide" and doc.id
    assert doc.page_content == (base / doc.metadata["md_path"]).read_text(encoding="utf-8")
    assert "package" not in doc.page_content.split("\n")[0]  # la metadata no va dentro del .md


def test_write_index_keeps_other_packages(tmp_path):
    base = tmp_path / "mode"
    base.mkdir()
    (base / "index.jsonl").write_text(json.dumps({"package": "other", "source_path": "a.md"}) + "\n")
    write_index([], base, {"fastlib"})
    assert json.loads((base / "index.jsonl").read_text().splitlines()[0])["package"] == "other"


def test_output_inside_raw_is_refused():
    with pytest.raises(ValueError):
        mode_dir("outputs_as_comments", RAW_DIR / "requests")


def test_clean_readme_changelog_with_version_titles():
    # pydantic: "## Changelog" seguido de "## v2.14.1 (2026-10-11)", "## v2.14.0 …", todos del mismo nivel
    md = ("# pydantic\n\nValidation.\n\n## Reporting a Security Vulnerability\n\nSee policy.\n\n"
          "## Changelog\n\n## v2.14.1 (2026-10-11)\n\n- Fix.\n\n## v2.14.0 (2026-10-08)\n\n- Feature.\n\n"
          "## Usage\n\nMore text.\n")
    out = clean_readme(md, "md")
    assert "Validation." in out and "More text." in out
    assert "policy" not in out and "v2.14" not in out and "Fix." not in out and "Feature." not in out


def test_clean_readme_rst_and_code():
    rst = "lib\n===\n\nUsage.\n\nChangelog\n---------\n\n1.0 first.\n\nExamples\n--------\n\nMore usage.\n"
    out = clean_readme(rst, "rst")
    assert "Usage." in out and "More usage." in out and "1.0 first." not in out
    md = "# lib\n\n```python\n# License: this comment is code\nx = 1\n```\n"
    assert clean_readme(md, "md") == md  # un `# License` dentro de un bloque de código no es un título
