"""Pool de librerías candidatas para el dataset crudo (ver docs/agents/plan_dataset_crudo.md §5).

De este pool, select_and_verify.py elige las 100 finales según la calidad medida de su documentación.
No incluye librerías deprecadas (van solo al catálogo del validador).
"""

CANDIDATES = {
    "http_redes": [
        "requests", "httpx", "aiohttp", "urllib3", "websockets", "websocket-client", "requests-oauthlib",
        "httplib2", "dnspython", "requests-toolbelt", "pycurl", "treq", "twisted", "scapy", "zeep", "paramiko",
        "fabric", "pyzmq", "asyncssh", "httpcore",
    ],
    "web_apis": [
        "fastapi", "flask", "django", "starlette", "uvicorn", "werkzeug", "falcon", "bottle", "tornado",
        "pyramid", "litestar", "quart", "sanic", "gunicorn", "djangorestframework", "flask-sqlalchemy",
        "flask-login", "flask-wtf", "channels", "strawberry-graphql", "graphene", "connexion", "wtforms",
    ],
    "datos_numerico": [
        "numpy", "pandas", "polars", "pyarrow", "scipy", "sympy", "networkx", "narwhals", "xarray", "dask",
        "statsmodels", "duckdb", "numba", "h5py", "shapely", "geopandas", "pint",
    ],
    "ml_ia": [
        "scikit-learn", "torch", "transformers", "huggingface-hub", "datasets", "accelerate", "peft", "diffusers",
        "sentence-transformers", "xgboost", "lightgbm", "spacy", "gensim", "nltk", "joblib", "optuna", "mlflow",
        "jax", "tokenizers", "tiktoken", "keras", "tensorflow", "lightning", "timm", "evaluate",
    ],
    "visualizacion": [
        "matplotlib", "plotly", "seaborn", "bokeh", "altair", "folium", "streamlit", "gradio", "pyvista",
        "plotnine", "pygal",
    ],
    "html_xml_scraping": [
        "beautifulsoup4", "lxml", "scrapy", "html5lib", "soupsieve", "xmltodict", "defusedxml", "selenium",
        "playwright", "parsel", "mechanicalsoup", "pyquery", "feedparser", "trafilatura", "cssselect", "w3lib",
    ],
    "documentos": [
        "pypdf", "openpyxl", "xlsxwriter", "python-docx", "pymupdf", "python-pptx", "pdfplumber", "fpdf2",
        "pdfminer.six", "weasyprint", "xlwings", "docxtpl", "camelot-py", "tabula-py", "reportlab",
    ],
    "imagenes": [
        "pillow", "opencv-python", "scikit-image", "imageio", "wand", "pyvips", "qrcode", "moviepy", "pydub",
        "python-barcode", "pytesseract",
    ],
    "cli_terminal": [
        "click", "typer", "rich", "tqdm", "colorama", "tabulate", "prompt-toolkit", "questionary", "fire",
        "textual", "argcomplete", "docopt", "cmd2", "plumbum", "invoke", "blessed", "urwid", "asciimatics",
    ],
    "fechas": [
        "python-dateutil", "arrow", "pendulum", "babel", "isodate", "tzlocal", "dateparser", "humanize",
        "croniter", "holidays",
    ],
    "config_serializacion": [
        "pyyaml", "tomli", "tomlkit", "python-dotenv", "orjson", "msgpack", "pydantic-settings", "jsonpatch",
        "jsonpointer", "simplejson", "dynaconf", "hydra-core", "omegaconf", "cattrs", "ujson", "jsonpickle",
        "configobj", "cbor2", "fastavro", "ruamel.yaml",
    ],
    "validacion_modelos": [
        "pydantic", "attrs", "marshmallow", "jsonschema", "email-validator", "fastjsonschema", "cerberus",
        "voluptuous", "schema", "msgspec", "typeguard", "beartype", "validators",
    ],
    "bases_de_datos": [
        "sqlalchemy", "alembic", "psycopg", "psycopg2", "asyncpg", "pymysql", "pymongo", "redis", "peewee",
        "sqlparse", "sqlmodel", "motor", "tortoise-orm", "elasticsearch", "pony", "databases", "aiosqlite",
        "sqlite-utils", "cassandra-driver", "neo4j", "influxdb-client", "mysqlclient", "oracledb", "mongoengine",
    ],
    "testing": [
        "pytest", "coverage", "hypothesis", "pytest-cov", "pytest-asyncio", "pytest-mock", "tox", "nox",
        "factory-boy", "faker", "responses", "freezegun", "behave", "locust", "pytest-django", "pytest-xdist",
        "vcrpy", "pytest-benchmark", "time-machine", "respx", "testcontainers",
    ],
    "seguridad_cripto": [
        "cryptography", "pyjwt", "bcrypt", "pynacl", "pycryptodome", "argon2-cffi", "authlib", "itsdangerous",
        "pyotp", "keyring", "pyopenssl", "rsa", "pwdlib", "oauthlib", "jwcrypto",
    ],
    "async_reintentos_tareas": [
        "tenacity", "backoff", "anyio", "aiofiles", "celery", "uvloop", "trio", "rq", "dramatiq", "apscheduler",
        "schedule", "huey", "arq", "stamina", "gevent", "aiocache", "kombu",
    ],
    "texto": [
        "regex", "rapidfuzz", "charset-normalizer", "jinja2", "markdown", "mako", "chardet", "jellyfish",
        "unidecode", "ftfy", "mistune", "markdown-it-py", "pygments", "textblob", "inflect", "emoji",
        "python-slugify", "levenshtein",
    ],
    "logging": [
        "loguru", "structlog", "python-json-logger", "colorlog", "sentry-sdk", "logbook", "eliot",
    ],
    "sistema_archivos": [
        "psutil", "filelock", "platformdirs", "watchfiles", "gitpython", "docker", "pexpect", "cachetools", "sh",
        "watchdog", "fsspec", "send2trash", "py7zr", "pyinstaller", "diskcache", "pathspec", "pyautogui",
    ],
    "sdks": [
        "boto3", "openai", "anthropic", "google-auth", "msal", "google-cloud-storage", "kubernetes", "pygithub",
        "tweepy", "discord.py", "python-telegram-bot", "praw", "slack-sdk", "google-api-python-client", "jira",
        "atlassian-python-api", "hvac", "minio",
    ],
}

# Repos que no se pueden deducir bien de PyPI, o donde la doc vive en otro repo ("host/owner/repo").
REPO_OVERRIDES = {
    "beautifulsoup4": "git.launchpad.net/beautifulsoup",
    "opencv-python": "github.com/opencv/opencv",  # opencv-python es solo el empaquetado
    "openpyxl": "foss.heptapod.net/openpyxl/openpyxl",  # Mercurial: se baja por la API de Heptapod
    "playwright": "github.com/microsoft/playwright",  # la doc de todos los lenguajes vive acá
    "streamlit": "github.com/streamlit/docs",
    "keras": "github.com/keras-team/keras-io",
    "tensorflow": "github.com/tensorflow/docs",
    "hydra-core": "github.com/facebookresearch/hydra",
    "spacy": "github.com/explosion/spaCy",
    "duckdb": "github.com/duckdb/duckdb-web",
    "numba": "github.com/numba/numba",
    "moviepy": "github.com/Zulko/moviepy",
    "cmd2": "github.com/python-cmd2/cmd2",
    "jellyfish": "github.com/jamesturk/jellyfish",
}

# Dónde está la doc cuando no es docs/, doc/ o documentation/:
#   docs: carpetas con .md/.rst/.ipynb.
#   examples: carpetas con .py que la doc incluye o publica como galería de ejemplos.
#   exclude: subcadenas de ruta a descartar además de las reglas generales.
SOURCES = {
    "paramiko": {"docs": ["sites/docs"]},
    "fabric": {"docs": ["sites/docs"]},
    "invoke": {"docs": ["sites/docs"]},
    "xlsxwriter": {"docs": ["dev/docs/source"], "examples": ["examples"]},
    "hypothesis": {"docs": ["hypothesis/docs"]},
    "pyarrow": {"docs": ["docs/source/python"]},  # apache/arrow también tiene la doc de C++, R, Java...
    "opencv-python": {"docs": ["doc/py_tutorials"], "exclude": ["py_setup"]},
    "openpyxl": {"docs": ["doc"]},
    "fastapi": {"docs": ["docs/en"], "examples": ["docs_src"]},
    "typer": {"docs": ["docs"], "examples": ["docs_src"]},
    "sqlmodel": {"docs": ["docs"], "examples": ["docs_src"]},
    "scikit-learn": {"docs": ["doc"], "examples": ["examples"]},
    "matplotlib": {"docs": ["doc"], "examples": ["galleries"]},
    "scikit-image": {"docs": ["doc/source"], "examples": ["doc/examples"]},
    "networkx": {"docs": ["doc"], "examples": ["examples"]},
    "statsmodels": {"docs": ["docs/source", "examples/notebooks"]},
    "optuna": {"docs": ["docs/source"], "examples": ["tutorial"]},
    "lightgbm": {"docs": ["docs"], "examples": ["examples/python-guide"]},
    "xgboost": {"docs": ["doc"], "examples": ["demo/guide-python"]},
    "gensim": {"docs": ["docs/src"], "examples": ["docs/src/gallery"]},
    "pyvista": {"docs": ["doc/source"], "examples": ["examples"]},
    "textual": {"docs": ["docs"], "examples": ["docs/examples"]},
    "rich": {"docs": ["docs"], "examples": ["examples"]},
    "plotly": {"docs": ["doc/python"]},
    "playwright": {"docs": ["docs/src"], "exclude": ["-csharp.", "-java.", "-js."]},  # solo Python
    "streamlit": {"docs": ["content"]},
    "keras": {"docs": ["templates"], "examples": ["guides"]},
    "tensorflow": {"docs": ["site/en/guide", "site/en/tutorials"]},
    "hydra-core": {"docs": ["website/docs"]},
    "spacy": {"docs": ["website/docs"]},
    "duckdb": {"docs": ["docs/current/clients/python", "docs/current/guides/python"]},
    "polars": {"docs": ["docs/source"], "examples": ["docs/source/src/python"]},  # {{code_block(...)}}
    "gradio": {"docs": ["guides"]},
    "transformers": {"docs": ["docs/source/en"]},
    "diffusers": {"docs": ["docs/source/en"]},
    "datasets": {"docs": ["docs/source"]},
    "accelerate": {"docs": ["docs/source"]},
    "peft": {"docs": ["docs/source"]},
    "evaluate": {"docs": ["docs/source"]},
    "timm": {"docs": ["hfdocs/source"]},
    "huggingface-hub": {"docs": ["docs/source/en"]},
    "lightning": {"docs": ["docs/source-pytorch"]},
    "mlflow": {"docs": ["docs/docs"]},
    "kubernetes": {"docs": ["doc", "examples"], "examples": ["examples"]},
    "pygithub": {"docs": ["doc"]},
    "boto3": {"docs": ["docs/source/guide"]},
    "pyautogui": {"exclude": ["chinese"]},
    #   docstrings: módulos cuya doc publicada es solo autodoc (los ejemplos viven en los docstrings).
    "rapidfuzz": {"docstrings": ["src/rapidfuzz/fuzz_py.py", "src/rapidfuzz/process_py.py",
                                 "src/rapidfuzz/utils_py.py", "src/rapidfuzz/distance/*_py.py"]},
}

# Nombre de import (y alias habituales) para medir si los bloques de código usan la librería.
# Si no figura, se deriva del nombre del paquete.
IMPORT_NAMES = {
    "numpy": ["numpy", "np"], "pandas": ["pandas", "pd"], "polars": ["polars", "pl"],
    "pyarrow": ["pyarrow", "pa", "pq", "pc"], "matplotlib": ["matplotlib", "plt", "mpl"],
    "seaborn": ["seaborn", "sns"], "plotly": ["plotly", "px", "go"], "altair": ["altair", "alt"],
    "scikit-learn": ["sklearn"], "scikit-image": ["skimage"], "beautifulsoup4": ["bs4", "BeautifulSoup", "soup"],
    "pillow": ["PIL", "Image"], "opencv-python": ["cv", "cv2"], "pyyaml": ["yaml"],
    "python-dateutil": ["dateutil"], "pymupdf": ["fitz", "pymupdf"], "python-docx": ["docx", "Document"],
    "python-pptx": ["pptx", "Presentation"], "pyjwt": ["jwt"], "pynacl": ["nacl"], "pycryptodome": ["Crypto"],
    "gitpython": ["git", "Repo"], "prompt-toolkit": ["prompt_toolkit"], "python-dotenv": ["dotenv"],
    "email-validator": ["email_validator"], "huggingface-hub": ["huggingface_hub"], "torch": ["torch", "nn"],
    "tensorflow": ["tensorflow", "tf"], "keras": ["keras", "layers"], "jax": ["jax", "jnp"],
    "sentence-transformers": ["sentence_transformers", "SentenceTransformer"], "pdfminer.six": ["pdfminer"],
    "fpdf2": ["fpdf", "FPDF"], "camelot-py": ["camelot"], "tabula-py": ["tabula"], "pyzmq": ["zmq"],
    "dnspython": ["dns"], "websocket-client": ["websocket"], "requests-oauthlib": ["requests_oauthlib", "OAuth"],
    "djangorestframework": ["rest_framework", "serializers"], "flask-sqlalchemy": ["flask_sqlalchemy", "db"],
    "flask-login": ["flask_login"], "flask-wtf": ["flask_wtf"], "strawberry-graphql": ["strawberry"],
    "argon2-cffi": ["argon2"], "pyopenssl": ["OpenSSL"], "python-json-logger": ["pythonjsonlogger"],
    "sentry-sdk": ["sentry_sdk"], "send2trash": ["send2trash"], "pygithub": ["github", "Github"],
    "discord.py": ["discord"], "python-telegram-bot": ["telegram"], "slack-sdk": ["slack_sdk"],
    "google-api-python-client": ["googleapiclient"], "atlassian-python-api": ["atlassian"],
    "google-cloud-storage": ["google.cloud", "storage"], "google-auth": ["google.auth", "google.oauth2"],
    "ruamel.yaml": ["ruamel", "YAML"], "factory-boy": ["factory"], "faker": ["faker", "fake", "Faker"],
    "time-machine": ["time_machine"], "pytest-mock": ["mocker"], "pytest-asyncio": ["pytest", "asyncio"],
    "pytest-cov": ["pytest", "cov"], "pytest-django": ["pytest", "django"], "pytest-xdist": ["pytest", "xdist"],
    "pytest-benchmark": ["benchmark"], "hydra-core": ["hydra", "cfg"], "markdown-it-py": ["markdown_it"],
    "python-slugify": ["slugify"], "levenshtein": ["Levenshtein"], "cassandra-driver": ["cassandra"],
    "influxdb-client": ["influxdb_client"], "mysqlclient": ["MySQLdb"], "pyvips": ["pyvips"],
    "python-barcode": ["barcode"], "tortoise-orm": ["tortoise"], "pydantic-settings": ["pydantic_settings"],
    "charset-normalizer": ["charset_normalizer"], "pymongo": ["pymongo", "MongoClient", "db"],
    "spacy": ["spacy", "nlp"], "duckdb": ["duckdb", "con"], "streamlit": ["streamlit", "st"],
    "msgpack": ["msgpack"], "elasticsearch": ["elasticsearch", "es"], "lightning": ["lightning", "L"],
    "redis": ["redis", "r"], "sqlalchemy": ["sqlalchemy", "session"],
    "statsmodels": ["statsmodels", "sm", "smf"], "soupsieve": ["soupsieve", "sv"], "gradio": ["gradio", "gr"],
    "vcrpy": ["vcr"], "cerberus": ["cerberus", "Validator"], "tenacity": ["tenacity", "retry"],
    "celery": ["celery", "app", "shared_task"], "omegaconf": ["omegaconf", "OmegaConf"],
    "python-dateutil": ["dateutil", "parse", "relativedelta", "rrule", "tz"], "peewee": ["peewee", "Model", "db"],
    "configobj": ["configobj", "ConfigObj"], "mongoengine": ["mongoengine", "Document", "connect"],
    "sqlite-utils": ["sqlite_utils", "db"], "xlsxwriter": ["xlsxwriter", "workbook", "worksheet"],
    "scapy": ["scapy", "IP", "TCP", "UDP", "sniff", "send", "sr1"], "playwright": ["playwright", "page"],
    "pyinstaller": ["PyInstaller"], "lxml": ["lxml", "etree", "html"], "parsel": ["parsel", "Selector"],
    "dask": ["dask", "dd", "da"], "geopandas": ["geopandas", "gpd"], "xarray": ["xarray", "xr"],
}
