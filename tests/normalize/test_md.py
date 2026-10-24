"""Paso previo y transformaciones para Markdown (MkDocs, fastapi, MyST, Docusaurus/MDX). Usan pandoc."""

from tprag.normalize.pandoc_tree import normalize_md


def md(text: str) -> str:
    return normalize_md(text).markdown


# --- MkDocs ---------------------------------------------------------------------------------------

def test_mkdocs_admonition_with_code_inside():
    # pydantic: el contenido indentado no tiene que leerse como bloque de código
    out = md('!!! warning "Careful"\n    Mutating the context is risky.\n\n    ```python\n    x = 1\n    ```\n')
    assert "**Careful:** Mutating the context is risky." in out
    assert "``` python\nx = 1\n```" in out


def test_mkdocs_admonition_without_title():
    assert "**Note:** Some text." in md("!!! note\n    Some text.\n")


def test_mkdocs_tabs_drop_python_variants():
    # pydantic: el mismo ejemplo para varias versiones de Python → se queda la primera
    text = ('=== "Python 3.10 and above"\n\n    ```python\n    x: int | None\n    ```\n\n'
            '=== "Python 3.9 and above"\n\n    ```python\n    x: Optional[int]\n    ```\n')
    norm = normalize_md(text)
    assert "x: int | None" in norm.markdown and "Optional[int]" not in norm.markdown
    assert norm.block_langs == ["python"]


def test_mkdocs_complementary_tabs_are_kept():
    # textual: el código y su hoja de estilos son pestañas complementarias
    text = '=== "app.py"\n\n    ```python\n    App().run()\n    ```\n\n=== "app.tcss"\n\n    ```css\n    Screen {}\n    ```\n'
    out = md(text)
    assert "**Tab: app.py**" in out and "**Tab: app.tcss**" in out
    assert "App().run()" in out and "Screen {}" in out


def test_mkdocstrings_reference_is_dropped():
    out = md("Intro.\n\n::: pydantic.BaseModel\n    options:\n      show_root_heading: true\n\nAfter.\n")
    assert "pydantic.BaseModel" not in out and "show_root_heading" not in out
    assert "Intro." in out and "After." in out


def test_header_attributes_are_removed():
    assert md("# First Steps { #first-steps }\n\nText.\n").startswith("# First Steps\n")


# --- fastapi --------------------------------------------------------------------------------------

def test_fastapi_blocks_and_tabs():
    text = ("/// tip\n\nUse the extension.\n\n///\n\n"
            "//// tab | Python 3.10+\n\n```Python\ndef f(q: str | None = None): ...\n```\n\n////\n\n"
            "//// tab | Python 3.10+ non-Annotated\n\n```Python\ndef f(q=None): ...\n```\n\n////\n")
    norm = normalize_md(text)
    assert "**Tip:** Use the extension." in norm.markdown
    assert "str | None" in norm.markdown and "q=None" not in norm.markdown
    assert norm.block_langs == ["python"]


def test_fastapi_terminal_html_tags_are_stripped():
    text = ('<div class="termy">\n\n```console\n$ <font color="#4E9A06">uv run fastapi</font> dev\n```\n\n</div>\n')
    out = md(text)
    assert "$ uv run fastapi dev" in out and "<font" not in out and "<div" not in out


# --- MyST -----------------------------------------------------------------------------------------

def test_myst_admonition_and_code_with_options():
    text = ("```{note}\nThis is a note.\n```\n\n"
            "```{code-block} python\n:caption: example.py\n:linenos:\n\nimport attrs\n```\n")
    out = md(text)
    assert "**Note:** This is a note." in out
    assert "``` python\nimport attrs\n```" in out and "caption" not in out


def test_myst_eval_rst_is_processed_as_rst():
    out = md("```{eval-rst}\nUse :func:`attrs.define` here.\n```\n")
    assert "`attrs.define`" in out and "eval-rst" not in out


def test_myst_colon_fence_and_version():
    out = md(":::{versionadded} 22.2.0\nNew API.\n:::\n")
    assert "**New in version 22.2.0:** New API." in out


def test_myst_dropped_directives():
    out = md("```{toctree}\n:maxdepth: 2\n\nintro\nguide\n```\n\nText.\n")
    assert "maxdepth" not in out and "guide" not in out and "Text." in out


# --- Docusaurus / MDX (playwright) ----------------------------------------------------------------

PLAYWRIGHT = """---
id: locators
title: "Locators"
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

## Introduction

:::note
Locators are strict.
:::

<Tabs
  groupId="python-flavor"
  defaultValue="sync"
  values={[
    {label: 'Sync', value: 'sync'},
    {label: 'Async', value: 'async'}
  ]
}>
<TabItem value="sync">

```py
page.get_by_role("button").click()
```

</TabItem>
<TabItem value="async">

```py
await page.get_by_role("button").click()
```

</TabItem>
</Tabs>
"""


def test_playwright_mdx():
    norm = normalize_md(PLAYWRIGHT)
    out = norm.markdown
    assert out.startswith("# Locators\n")  # el title del encabezado YAML pasa a ser el H1
    assert "import Tabs" not in out and "groupId" not in out and "<TabItem" not in out
    assert "**Note:** Locators are strict." in out
    assert "**Tab: sync**" in out and "**Tab: async**" in out  # sync / async: complementarias
    assert norm.block_langs == ["python", "python"]


def test_docusaurus_admonition_with_bracket_title():
    out = md(":::note[When to use role locators]\nPrefer role locators.\n:::\n")
    assert "**When to use role locators:** Prefer role locators." in out and ":::" not in out


def test_htmlcard_keeps_only_the_code():
    # playwright: la vista previa del HTML repite el bloque de código que viene abajo
    out = md("<HTMLCard>\n<div>\n<button>Sign in</button>\n</div>\n\n```html\n<button>Sign in</button>\n```\n"
             "</HTMLCard>\n")
    assert out.count("Sign in") == 1 and "``` html" in out


def test_title_from_front_matter_when_code_has_python_comments():
    out = md('---\ntitle: "Locators"\n---\n\n## Intro\n\n```py\n# a comment\nx = 1\n```\n')
    assert out.startswith("# Locators\n")


# --- Casos del piloto -----------------------------------------------------------------------------

def test_fence_with_only_options_has_no_language():
    # fastapi: ```hl_lines="4" solo resalta líneas; el lenguaje lo decide la regla C
    norm = normalize_md('```hl_lines="4"\nfrom fastapi import FastAPI\napp = FastAPI()\n```\n')
    assert norm.block_langs == ["python"]


def test_textual_screenshot_blocks_are_dropped():
    norm = normalize_md('Text.\n\n```{.textual path="docs/examples/guide/dom1.py"}\n```\n\n```python\nApp().run()\n```\n')
    assert "dom1.py" not in norm.markdown
    assert norm.block_langs == ["python"]  # la captura no entra en la numeración


# --- HTML -----------------------------------------------------------------------------------------

def test_raw_html_keeps_visible_text():
    out = md('<p align="center"><img src="logo.png"></p>\n\n<details>\n<summary>More info</summary>\n\nHidden text.\n\n</details>\n')
    assert "More info" in out and "Hidden text." in out
    assert "<" not in out and "logo.png" not in out


def test_straight_quotes_and_dollar_signs_are_kept():
    out = md('Say "hello" and it costs $5 or $10.\n')
    assert '"hello"' in out and "$5 or $10" in out
