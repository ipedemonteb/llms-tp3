"""Lenguaje de cada bloque de código y conversión de los ejemplos de consola.

- Lenguaje (loaders.md §3.4): se respeta el declarado (con alias unificados); si no hay, regla C sobre el
  contenido, con reglas de formato antes de preguntarle a Python.
- Ejemplos de consola (loaders.md §3.2): `>>>` / `...` e `In [n]:` / `Out[n]:` se convierten según
  `console_examples`.
"""

import ast
import json
import re
import textwrap
import warnings
from typing import Literal

ConsoleMode = Literal["as_is", "code_only", "outputs_as_comments"]
CONSOLE_MODES: tuple[ConsoleMode, ...] = ("as_is", "code_only", "outputs_as_comments")
MAX_OUTPUT_LINES = 5  # líneas de salida que se conservan como comentario antes de recortar

LANG_ALIASES = {
    "py": "python", "py3": "python", "python3": "python", "python2": "python", "ipython": "python",
    "ipython3": "python",
    "python+sql": "python",  # sqlalchemy: Python con el SQL generado como salida
    "pycon": "pycon", "python-console": "pycon", "pycon+sql": "pycon", "doctest": "pycon",
    "sh": "bash", "shell": "bash", "zsh": "bash",
    "console": "console", "shell-session": "console", "sh-session": "console", "pytest": "console",
    "none": "text", "plain": "text", "plaintext": "text", "txt": "text", "default": "text",
    "yml": "yaml", "jsonc": "json", "json5": "json",
    "html+django": "html", "html+jinja": "html", "django": "html", "jinja": "html",
    "javascript": "js", "typescript": "ts",
}

# Directivas de MyST que son bloques de código, con el lenguaje por defecto si no lo declaran.
MYST_CODE = {"code-block": "", "code": "", "sourcecode": "", "ipython": "python", "testcode": "python",
             "jupyter-execute": "python", "doctest": "pycon"}

# Prompts de consola: `>>> x` / `... x` y los de IPython (`In [1]: x`, `   ...: x`, `Out[1]: valor`).
PROMPT = re.compile(r"^(?:>>>|\.\.\.)(?: |$)")
IPY_IN = re.compile(r"^In \[\d*\]:(?: |$)")
IPY_CONT = re.compile(r"^\s*\.\.\.+:(?: |$)")
IPY_OUT = re.compile(r"^Out\[\d*\]:\s?")

# Reglas de formato (loaders.md §3.4): se aplican antes de `ast.parse`, porque Python acepta estos textos.
CONFIG_SECTION = re.compile(r'^\s*\[{1,2}[\w.\-]+(?:\s+"[^"]*")?\]{1,2}\s*$')  # [run], [tool.x], [[tabla]]
CONFIG_LINE = re.compile(r"^\s*(?:[#;].*|[\w.\-]+\s*[=:].*|\s+\S.*|[\]\}],?)\s*$")
ENV_LINE = re.compile(r"^\s*(?:export\s+)?[A-Z][A-Z0-9_]*=\S*")

# "Forma de código": sin esto, salidas como `{'page': 0}` o una palabra suelta contarían como Python.
CODE_SHAPE = (ast.Import, ast.ImportFrom, ast.Assign, ast.AugAssign, ast.AnnAssign, ast.FunctionDef,
              ast.AsyncFunctionDef, ast.ClassDef, ast.Call, ast.For, ast.AsyncFor, ast.While, ast.With,
              ast.AsyncWith, ast.If, ast.Try, ast.Return, ast.Raise, ast.Assert, ast.Delete)
# Para snippets que Python no puede leer (incompletos, con `# ...`). Mismo criterio que PY_HINT de
# src/scripts/fetch_raw_docs.py, así lo que la descarga contó como Python coincide con lo que se etiqueta.
PY_HINT = re.compile(r"^\s*(>>> |import \w|from [\w.]+ import |def \w|class \w|@\w|[\w.\[\]'\"]+\s*=[^=]|[\w.]+\()",
                     re.M)


def normalize_lang(declared: str | None) -> str:
    """Lenguaje declarado con alias unificados ("" si no hay).

    Acepta directivas de MyST: `{code-block} python` → python; `{ipython}` → python; `{doctest}` → pycon.
    Otra directiva (`{note}`) devuelve su nombre: no es un bloque de código y lo decide quien llama.
    """
    text = (declared or "").strip().lower()
    myst = re.match(r"\{([\w-]+)\}\s*(\S*)", text)
    if myst:
        name, arg = myst.groups()
        if name not in MYST_CODE:
            return name
        text = (arg or "") if MYST_CODE[name] == "" else MYST_CODE[name]  # lenguaje fijo: el argumento es otra cosa
    # Atributos de pandoc que no se reconocieron como tales: ```{.textual path="x.py"} → textual
    text = re.sub(r"^\{\s*\.", "", text)
    words = text.split()
    if not words or "=" in words[0]:  # ```hl_lines="4" (fastapi): solo una opción de resaltado, sin lenguaje
        return ""
    return LANG_ALIASES.get(words[0], words[0])


def is_console(code: str) -> bool:
    """Alguna línea arranca con `>>>` o `In [n]:` (`...` solo no alcanza: también es Ellipsis o relleno)."""
    return any(re.match(r">>>(?: |$)", l.lstrip()) or IPY_IN.match(l.lstrip()) for l in code.splitlines())


def detect_lang(code: str) -> str:
    """Regla C, para bloques sin lenguaje declarado."""
    lines = [l for l in code.splitlines() if l.strip()]
    if not lines:
        return "text"
    if is_console(code):
        return "pycon"
    if lines[0].lstrip().startswith(("$ ", "% ")):
        return "console"
    stripped = code.strip()
    if stripped[:1] in "{[":
        try:
            json.loads(stripped)
            return "json"
        except ValueError:
            pass
    config = _config_lang(lines)
    if config:
        return config
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # SyntaxWarning por escapes inválidos en strings
            tree = ast.parse(textwrap.dedent(code))
        return "python" if any(isinstance(n, CODE_SHAPE) for n in ast.walk(tree)) else "text"
    except (SyntaxError, ValueError):
        pass
    return "python" if PY_HINT.search(code) else "text"


def _config_lang(lines: list[str]) -> str | None:
    """INI/TOML (`[sección]` + `clave = valor`) o variables de entorno (`NOMBRE=valor ...`)."""
    if any(CONFIG_SECTION.match(l) for l in lines) and all(CONFIG_SECTION.match(l) or CONFIG_LINE.match(l)
                                                           for l in lines):
        return "toml" if any(re.search(r"=\s*[\"\[\{]", l) for l in lines) else "ini"
    commands = [l for l in lines if not l.lstrip().startswith("#")]
    if commands and all(ENV_LINE.match(l) for l in commands):
        return "console"  # en la doc son comandos con variables adelante: `CFLAGS="-O0" make inplace`
    return None


def resolve_lang(declared: str | None, code: str) -> str:
    lang = normalize_lang(declared) or detect_lang(code)
    # Un bloque declarado como Python que es una sesión de consola se trata como consola.
    return "pycon" if lang == "python" and is_console(code) else lang


def convert_console(code: str, mode: ConsoleMode, max_output_lines: int = MAX_OUTPUT_LINES) -> str:
    """Quita los prompts; las salidas se dejan (as_is), se eliminan (code_only) o pasan a comentario."""
    if mode == "as_is":
        return code
    out, run = [], 0  # run: líneas de salida seguidas ya emitidas
    for line in textwrap.dedent(code).splitlines():
        bare = line.lstrip()
        if PROMPT.match(bare) or IPY_IN.match(bare) or IPY_CONT.match(line):
            out.append(re.sub(r"^\s*(?:>>>|\.\.\.+:?|In \[\d*\]:) ?", "", line, count=1))
            run = 0
        elif not bare:
            out.append("")
            run = 0
        elif mode == "outputs_as_comments":
            value = IPY_OUT.sub("", bare) if IPY_OUT.match(bare) else line.rstrip()
            if run < max_output_lines:
                out.append(f"# {value}")
            elif run == max_output_lines:
                out.append("# ...")
            run += 1
    return "\n".join(out).strip("\n")


def process_block(declared: str | None, code: str, console_mode: ConsoleMode = "outputs_as_comments"
                  ) -> tuple[str, str]:
    """(lenguaje final, código final) de un bloque."""
    lang = resolve_lang(declared, code)
    if lang == "pycon" and console_mode != "as_is":
        return "python", convert_console(code, console_mode)
    return lang, code
