"""Casos tomados del corpus (data/raw) al medir la regla C y los ejemplos de consola (loaders.md §3.2, §3.4)."""

import pytest

from tprag.normalize.codeblocks import convert_console, detect_lang, normalize_lang, process_block, resolve_lang

REQUESTS_PYCON = """\
>>> payload = {'key1': 'value1', 'key2': 'value2'}
>>> r = requests.get('https://httpbin.org/get', params=payload)
>>> print(r.url)
https://httpbin.org/get?key2=value2&key1=value1"""


# --- Lenguaje declarado ---------------------------------------------------------------------------

@pytest.mark.parametrize("declared, expected", [
    ("python", "python"), ("py", "python"), ("python3", "python"), ("ipython3", "python"),
    ("pycon", "pycon"), ("pycon+sql", "pycon"), ("doctest", "pycon"),
    ("shell", "bash"), ("sh", "bash"), ("console", "console"), ("pytest", "console"),
    ("none", "text"), ("yml", "yaml"), ("html+django", "html"),
    ("{code-block} python", "python"), ("{ipython}", "python"), ("{doctest}", "pycon"),  # MyST
    ("{code-block}", ""),  # MyST sin lenguaje: lo decide la regla C
    ("{note}", "note"),  # directiva de MyST que no es código
    ("{doctest} namedtuples", "pycon"),  # cattrs: el argumento es un grupo de tests, no el lenguaje
    ("python+sql", "python"),  # sqlalchemy
    ('{.textual path="docs/examples/app.py"', "textual"),  # atributos de pandoc sin reconocer
    ("python title='ejemplo.py'", "python"),  # info string con atributos
    ("", ""), (None, ""),
])
def test_normalize_lang(declared, expected):
    assert normalize_lang(declared) == expected


def test_declared_python_with_prompts_is_console():
    assert resolve_lang("python", REQUESTS_PYCON) == "pycon"


def test_declared_language_wins_over_content():
    assert resolve_lang("bash", "x = 1") == "bash"


# --- Regla C (sin lenguaje declarado) ---------------------------------------------------------------

@pytest.mark.parametrize("code, expected", [
    # requests/docs/user/authentication.rst
    (">>> from requests.auth import HTTPDigestAuth\n>>> requests.get(url)\n<Response [200]>", "pycon"),
    # invoke/sites/docs/concepts/namespaces.rst
    ("$ invoke --list\nAvailable tasks:\n\n    release.release", "console"),
    ('{"name": "x", "version": 1}', "json"),
    # xlsxwriter/.../working_with_conditional_formats.rst
    ("worksheet.conditional_format('A1:A4', {'type': 'text',\n"
     "                                       'criteria': 'containing'})", "python"),
    # aiohttp/docs/web_quickstart.rst: incompleto, lo salva la heurística
    ("app.add_routes([web.get('/path1', get_1),\n                web.post('/path1', post_1)]", "python"),
    # invoke: función con `# ...` como cuerpo
    ("@task(name='dir')\ndef dir_(c):\n    # ...", "python"),
    # pymupdf/docs/document.rst: Python lo lee, pero es una salida sin forma de código
    ("{\n    '__bookmark_1': {'page': 0, 'to': (0.0, 541.0), 'zoom': 0.0},\n}", "text"),
    # numpy: código C
    ("#if NPY_ABI_VERSION < 0x02000000\n  #define PyDataType_ELSIZE(descr) ((descr)->elsize)\n#endif", "text"),
    ("requests", "text"),  # una palabra suelta es Python válido, pero no tiene forma de código
    ("", "text"),
])
def test_detect_lang(code, expected):
    assert detect_lang(code) == expected


@pytest.mark.parametrize("code, expected", [
    # hypothesis/docs/extensions.rst: el falso positivo que motivó las reglas de formato
    ("[run]\nparallel = True\nsource = ...", "ini"),
    # boto3/docs/source/guide/configuration.rst (con valores indentados)
    ("[default]\nregion = us-east-1\napi_versions =\n    ec2 = 2015-03-01\n    cloudfront = 2015-09-17", "ini"),
    # pytest-django/docs/managing_python_path.rst (se le escapaba a la regla de la medición)
    ('# pyproject.toml\n[build-system]\nrequires = [\n    "setuptools>=61.0.0",\n]\n'
     'build-backend = "setuptools.build_meta"', "toml"),
    # lxml/doc/build.txt: variables de entorno + comando
    ('CFLAGS="-O0"  make inplace', "console"),
    ("STATIC_DEPS=true sudo pip install lxml", "console"),
])
def test_config_rules(code, expected):
    assert detect_lang(code) == expected


def test_traceback_log_is_not_python():
    # torch (revisión manual, ítems 23 y 42): un log de error con traceback lo salvaba la heurística
    log = ("torch._dynamo.convert_frame: [ERROR] WON'T CONVERT test_assertion_error\ndue to:\n"
           "Traceback (most recent call last):\n  File \"symbolic_convert.py\", line 837, in BUILD_MAP\n"
           "    assert isinstance(k, ConstantVariable)\nAssertionError")
    assert detect_lang(log) == "pytb"


def test_python_with_list_line_is_not_ini():
    # Una línea entre corchetes con espacios no es una sección de configuración.
    assert detect_lang("[x for x in items]\ntotal = 1") == "python"


def test_python_constant_with_spaces_is_not_env():
    assert detect_lang('DEBUG = True\nALLOWED_HOSTS = ["*"]') == "python"


# --- Ejemplos de consola --------------------------------------------------------------------------

def test_outputs_as_comments():
    assert convert_console(REQUESTS_PYCON, "outputs_as_comments") == (
        "payload = {'key1': 'value1', 'key2': 'value2'}\n"
        "r = requests.get('https://httpbin.org/get', params=payload)\n"
        "print(r.url)\n"
        "# https://httpbin.org/get?key2=value2&key1=value1")


def test_code_only():
    assert convert_console(REQUESTS_PYCON, "code_only").splitlines()[-1] == "print(r.url)"
    assert "httpbin.org/get?key2" not in convert_console(REQUESTS_PYCON, "code_only")


def test_as_is():
    assert convert_console(REQUESTS_PYCON, "as_is") == REQUESTS_PYCON


def test_continuation_lines_keep_indentation():
    code = ">>> def f(x):\n...     return x + 1\n>>> f(1)\n2"
    assert convert_console(code, "outputs_as_comments") == "def f(x):\n    return x + 1\nf(1)\n# 2"


def test_long_outputs_are_cut():
    code = ">>> print_table()\n" + "\n".join(f"fila {i}" for i in range(20))
    lines = convert_console(code, "outputs_as_comments", max_output_lines=5).splitlines()
    assert lines == ["print_table()", "# fila 0", "# fila 1", "# fila 2", "# fila 3", "# fila 4", "# ..."]


def test_ipython_session():
    code = "In [1]: import pandas as pd\n\nIn [2]: pd.Timedelta('1 days')\nOut[2]: Timedelta('1 days 00:00:00')"
    assert convert_console(code, "outputs_as_comments") == (
        "import pandas as pd\n\npd.Timedelta('1 days')\n# Timedelta('1 days 00:00:00')")


def test_process_block():
    assert process_block(None, REQUESTS_PYCON)[0] == "python"
    assert process_block(None, REQUESTS_PYCON, "as_is") == ("pycon", REQUESTS_PYCON)
    assert process_block("bash", "pip install pypdf") == ("bash", "pip install pypdf")
