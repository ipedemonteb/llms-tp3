"""Paso previo y transformaciones del árbol para rst (loaders.md §3.5, §3.6). Usan pandoc (pypandoc-binary)."""

from tprag.normalize.pandoc_tree import normalize_rst
from tprag.normalize.prepass import prepass_rst


def md(text: str, mode: str = "outputs_as_comments") -> str:
    return normalize_rst(text, mode).markdown


# --- Paso previo ----------------------------------------------------------------------------------

def pre(text: str) -> str:
    return prepass_rst(text).rstrip("\n")


def test_ipython_becomes_code_block():
    # pandas: sin el paso previo, pandoc junta estas líneas en un párrafo
    text = ".. ipython:: python\n   :suppress:\n\n   import datetime\n   @savefig x.png\n   pd.Timedelta('1 days')\n"
    assert pre(text) == ".. code-block:: python\n\n   import datetime\n   pd.Timedelta('1 days')"


def test_api_directive_keeps_signature():
    # flask/docs/config.rst: sin el paso previo se pierde el nombre DEBUG
    text = ".. data:: DEBUG\n\n   Whether debug mode is enabled.\n"
    assert pre(text) == "``DEBUG``\n\nWhether debug mode is enabled."


def test_nested_api_directives():
    text = ".. class:: Session()\n\n   A session.\n\n   .. method:: get(url)\n\n      Sends a GET.\n"
    assert pre(text) == "``Session()``\n\nA session.\n\n``get(url)``\n\nSends a GET."


def test_directives_shown_as_examples_are_not_touched():
    text = "Write this::\n\n    .. function:: foo()\n\n       Doc.\n"
    assert prepass_rst(text) == text


def test_native_doctest_block_becomes_code():
    # Párrafo que empieza con >>> sin "::" (numpydoc, dateutil…): pandoc lo leería como texto
    text = "Example:\n\n>>> ratio('a', 'b')\n0.0\n\nAfter.\n"
    assert pre(text) == "Example:\n\n.. code-block:: pycon\n\n   >>> ratio('a', 'b')\n   0.0\n\nAfter."
    norm = normalize_rst(text)
    assert norm.block_langs == ["python"] and "ratio('a', 'b')\n# 0.0" in norm.markdown


def test_hidden_directives_are_removed():
    assert pre("Text.\n\n.. testsetup::\n\n   import os\n\nMore.") == "Text.\n\n\nMore."


def test_code_tab_becomes_tab_with_code():
    text = ".. code-tab:: python Synchronous\n\n   run()\n"
    assert pre(text) == ".. tab:: Synchronous\n\n   .. code-block:: python\n\n      run()"


# --- Árbol ----------------------------------------------------------------------------------------

def test_sphinx_roles():
    out = md("Now, we have a :class:`Response <requests.Response>` object.\n\n"
             "Use :meth:`~requests.Session.get` or see :ref:`installed <install>` and :pep:`8`.")
    assert "`requests.Response`" in out
    assert "`requests.Session.get`" in out
    assert "installed" in out and "<install>" not in out
    assert "PEP 8" in out


def test_literal_block_gets_language_and_console_conversion():
    norm = normalize_rst("Example::\n\n    >>> r = requests.get(url)\n    >>> r.status_code\n    200\n")
    assert norm.block_langs == ["python"]
    assert "``` python\nr = requests.get(url)\nr.status_code\n# 200\n```" in norm.markdown


def test_console_mode_as_is():
    out = md("Example::\n\n    >>> 1 + 1\n    2\n", "as_is")
    assert "``` pycon\n>>> 1 + 1\n2\n```" in out


def test_admonition_and_version_notes():
    out = md(".. note::\n\n   Be careful.\n\n.. deprecated:: 2.0\n   Use :func:`new.thing` instead.\n\n"
             ".. versionadded:: 3.1\n")
    assert "**Note:** Be careful." in out
    assert "**Deprecated since version 2.0:** Use `new.thing` instead." in out
    assert "**New in version 3.1:**" in out


def test_dropped_directives_and_links():
    norm = normalize_rst(".. module:: requests.models\n\nSee `the docs <https://example.com>`_.\n\n"
                         ".. image:: logo.png\n")
    assert "requests.models" not in norm.markdown
    assert "https://example.com" not in norm.markdown and "the docs" in norm.markdown
    assert "<div" not in norm.markdown
    assert norm.dropped["module"] == 1


def test_tabs_keep_all_with_title():
    text = (".. tabs::\n\n   .. group-tab:: Bash\n\n      .. code-block:: text\n\n         $ export X=1\n\n"
            "   .. group-tab:: Fish\n\n      .. code-block:: text\n\n         $ set -x X 1\n")
    out = md(text)
    assert "**Tab: Bash**" in out and "**Tab: Fish**" in out
    assert "export X=1" in out and "set -x X 1" in out


def test_tabs_drop_other_languages_when_python_exists():
    text = (".. tabs::\n\n   .. tab:: Python\n\n      .. code-block:: python\n\n         page.goto(url)\n\n"
            "   .. tab:: Java\n\n      .. code-block:: java\n\n         page.navigate(url);\n")
    norm = normalize_rst(text)
    assert "page.goto(url)" in norm.markdown and "page.navigate" not in norm.markdown
    assert norm.block_langs == ["python"]  # el bloque descartado no entra en la numeración


def test_template_placeholder_paragraph_is_dropped_but_not_in_code():
    # pandas: `{{ header }}` lo reemplaza Sphinx; dentro de un bloque de código es parte del ejemplo
    norm = normalize_rst("{{ header }}\n\nText.\n\nTemplate::\n\n    {{ header }}\n")
    assert norm.markdown.startswith("Text.")
    assert "``` text\n{{ header }}\n```" in norm.markdown
    assert norm.dropped["variable de plantilla"] == 1


def test_default_role_is_code():
    # flask, scikit-learn: `texto` sin rol se ve como código en el sitio
    out = md("Use `TimeSeriesSplit` with `extrapolation=\"periodic\"`.\n")
    assert '`TimeSeriesSplit`' in out and '`extrapolation="periodic"`' in out


def test_sqlalchemy_reference_aliases():
    out = md("Use :class:`_orm.Session` and :meth:`_sql.Select.where`, or :class:`~sqlalchemy.Integer`.\n")
    assert "`sqlalchemy.orm.Session`" in out and "`sqlalchemy.sql.Select.where`" in out
    assert "`sqlalchemy.Integer`" in out and "_orm" not in out
    # :paramref: (parámetros) y `texto` sin rol también llevan alias
    out = md("See :paramref:`_orm.relationship.foreign_keys` and `_orm.Session`.\n")
    assert "`sqlalchemy.orm.relationship.foreign_keys`" in out and "`sqlalchemy.orm.Session`" in out


def test_superscript_as_text():
    # pandas: `2\ :sup:`31`` bytes → 2^31 (pandoc lo escribía como <sup>31</sup>)
    out = md("Chunks up to 2\\ :sup:`31` bytes and H\\ :sub:`2`\\ O.\n")
    assert "2^31 bytes" in out and "H_2O" in out and "<sup>" not in out


def test_admonition_with_custom_title():
    # django: `.. admonition:: Django's Time Zone` → el título es la etiqueta
    out = md(".. admonition:: Django's Time Zone\n\n   Django includes a TIME_ZONE setting.\n")
    assert "**Django's Time Zone:** Django includes a TIME_ZONE setting." in out


def test_complex_table_is_written_as_markdown_not_html():
    # scikit-learn svm.rst, pandas io.rst, torch: celdas con varios párrafos, fórmulas o código → pandoc escribía HTML
    text = ("+------+--------------------+\n"
            "| Name | Description        |\n"
            "+======+====================+\n"
            "| x    | para one           |\n"
            "|      |                    |\n"
            "|      | para two           |\n"
            "+------+--------------------+\n"
            "| y    | uses :math:`a^2`   |\n"
            "+------+--------------------+\n"
            "| z    | ::                 |\n"
            "|      |                    |\n"
            "|      |     f(1)           |\n"
            "+------+--------------------+\n")
    norm = normalize_rst(text)
    assert "<table" not in norm.markdown and "<td" not in norm.markdown
    assert "| x" in norm.markdown and "para one para two" in norm.markdown and "a^2" in norm.markdown
    assert "`f(1)`" in norm.markdown and norm.block_langs == []  # el código de la celda no se numera


def test_definition_list_becomes_bold_term():
    out = md("term\n   Its definition.\n")
    assert "**term**" in out and "Its definition." in out
