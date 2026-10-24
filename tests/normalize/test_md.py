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


def test_myst_doctest_argument_is_not_the_language():
    # cattrs: ```{doctest} namedtuples → el bloque es consola de Python, "namedtuples" es un grupo de tests
    norm = normalize_md("```{doctest} namedtuples\n>>> from cattrs import structure\n>>> structure(1, int)\n1\n```\n")
    assert norm.block_langs == ["python"] and "namedtuples" not in norm.markdown


def test_myst_directive_inside_list_item_keeps_indentation():
    # attrs: un ```{doctest} dentro de un ítem de lista se rompía y se tragaba el título siguiente
    text = ("- Expose `field` wrappers:\n\n   ```{doctest}\n   >>> def typed(cls):\n   ...     return cls\n"
            "   ```\n\n## Automatic Field Transformation\n\nText.\n")
    norm = normalize_md(text)
    assert norm.block_langs == ["python"]
    # el bloque queda dentro del ítem de la lista (indentado) y el título siguiente, fuera del bloque
    assert "  def typed(cls):\n      return cls\n  ```\n\n## Automatic Field Transformation" in norm.markdown


def test_myst_roles_and_labels():
    out = md("(extending-metadata)=\n\n## Metadata\n\nUse {func}`attrs.define`, {class}`~attrs.Attribute` "
             "and see {pep}`681` or {ref}`the docs <metadata>`.\n")
    assert "`attrs.define`" in out and "`attrs.Attribute`" in out and "PEP 681" in out and "the docs" in out
    assert "{func}" not in out and "(extending-metadata)=" not in out


def test_mkdocs_autorefs_keep_text_but_not_indexing():
    out = md("Use [`Annotated`][typing.Annotated] and [model_validate()][pydantic.BaseModel.model_validate].\n\n"
             "Index with `x[0][1]` or d[a][b].\n")
    assert "`Annotated` and model_validate()." in out and "typing.Annotated" not in out
    assert "`x[0][1]`" in out and "d[a][b]" in out


def test_empty_code_block_is_dropped():
    # textual: snippet .tcss que no se pudo resolver → bloque vacío
    norm = normalize_md("Text.\n\n```css\n```\n\n```python\nApp().run()\n```\n")
    assert norm.block_langs == ["python"] and "``` css" not in norm.markdown


def test_html_code_block_keeps_its_tags():
    # playwright: en un bloque html, <span> es el ejemplo; en la terminal de fastapi, <font> es ruido
    out = md('```html\n<span title="Issues count">25 issues</span>\n```\n\n```console\n$ <font color="red">run</font>\n```\n')
    assert '<span title="Issues count">25 issues</span>' in out and "$ run" in out


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


# --- Hugging Face (transformers) ------------------------------------------------------------------

def test_hf_tabs_tips_links_and_autodoc():
    text = ("[[open-in-colab]]\n\nUse the [`Pipeline`] or [`~GenerationMixin.generate`] API.\n\n"
            '<hfoptions id="install">\n<hfoption id="pip">\n```bash\npip install transformers\n```\n</hfoption>\n'
            '<hfoption id="uv">\n```bash\nuv pip install transformers\n```\n</hfoption>\n</hfoptions>\n\n'
            "<Tip warning={true}>\n\nLarge models need a GPU.\n\n</Tip>\n\n## AutoModel\n\n[[autodoc]] AutoModel\n"
            "    - from_pretrained\n    - save_pretrained\n\nEnd.\n")
    norm = normalize_md(text)
    out = norm.markdown
    assert "open-in-colab" not in out and "[[autodoc]]" not in out and "from_pretrained" not in out
    assert "`Pipeline`" in out and "`GenerationMixin.generate`" in out and "[`" not in out
    assert "**Tab: pip**" in out and "**Tab: uv**" in out and "<hfoption" not in out
    assert norm.block_langs == ["bash", "bash"]
    assert "**Warning:** Large models need a GPU." in out and "End." in out


# --- streamlit, gradio ----------------------------------------------------------------------------

def test_streamlit_components():
    text = ('<InlineCalloutContainer>\n    <InlineCallout color="red" href="/x">Go somewhere</InlineCallout>\n'
            "</InlineCalloutContainer>\n\n"
            "<Important>\n\n    Forms rerun the app <i style={{ verticalAlign: \"-.25em\" }} "
            'className={{ class: "material-icons-sharp" }}>warning</i> on submit.\n\n</Important>\n\n'
            '<Collapse title="Example">\n\n```python\nst.form("f")\n```\n\n</Collapse>\n\n'
            '<Image src="/img.png" clean />\n\nEnd.\n')
    norm = normalize_md(text)
    out = norm.markdown
    assert "Go somewhere" not in out and "<" not in out.replace("<!--", "")
    assert "**Important:** Forms rerun the app on submit." in out  # sin bloque de código falso
    assert "**Example**" in out and norm.block_langs == ["python"] and "End." in out


def test_gradio_metadata_lines_are_dropped():
    out = md("# Quickstart\n\nTags: GETTING STARTED, INTRO\nRelated spaces: https://x\n\nGradio is a library.\n")
    assert "Tags" not in out and "Related spaces" not in out and "Gradio is a library." in out


# --- Fórmulas y artefactos de pandoc --------------------------------------------------------------

def test_math_is_not_mangled():
    # transformers, torch: `z^*` se volvía <sup>; y un párrafo con solo una fórmula ($$…$$) desaparecía
    out = md("The loss is $z^*_i = x^*$ and\n\n$$\\left\\{ a \\right\\}$$\n\nwith a\\\nhard break.\n")
    assert "<sup>" not in out and "z^*_i = x^*" in out  # en Markdown de GitHub: $`z^*_i = x^*`$
    assert "\\left\\{ a \\right\\}" in out
    assert "a\\\n" not in out and "with a\nhard break." in out


# --- HTML -----------------------------------------------------------------------------------------

def test_raw_html_keeps_visible_text():
    out = md('<p align="center"><img src="logo.png"></p>\n\n<details>\n<summary>More info</summary>\n\nHidden text.\n\n</details>\n')
    assert "More info" in out and "Hidden text." in out
    assert "<" not in out and "logo.png" not in out


def test_straight_quotes_and_dollar_signs_are_kept():
    out = md('Say "hello" and it costs $5 or $10.\n')
    assert '"hello"' in out and "$5 or $10" in out


def test_markdown_escapes_are_removed_outside_code():
    # rapidfuzz: pandoc escribía `Sequence\[Hashable\]` y `ratio \< score_cutoff`
    out = md("s1 : Sequence[Hashable] and ratio < score_cutoff, a_b.\n\n```python\nx = '\\\\['\n```\n")
    assert "Sequence[Hashable]" in out and "ratio < score_cutoff" in out and "a_b" in out
    assert "x = '\\\\['" in out  # dentro del código no se toca
