"""Import → paquete de PyPI (validador.md §2.4): tabla manual → tabla de pipreqs → variantes de nombre.

`data/pipreqs_mapping` es una copia **sin modificar** de `pipreqs/mapping` del proyecto bndr/pipreqs
(https://github.com/bndr/pipreqs), commit d9040b7ad5afafade325fee41762b6d6166c8597, licencia Apache 2.0 (copia en
`data/pipreqs_LICENSE`). Las correcciones van en `MANUAL`, que se consulta primero.
"""

import re
import sys
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# Solo donde el import no coincide con el paquete. Las claves con punto se prueban de la más larga a la más corta.
MANUAL = {
    # namespaces: dos niveles (pipreqs mapea `google` a protobuf)
    "google.auth": "google-auth", "google.oauth2": "google-auth", "google_auth_oauthlib": "google-auth-oauthlib",
    "google.cloud.storage": "google-cloud-storage", "google.cloud.bigquery": "google-cloud-bigquery",
    "google.protobuf": "protobuf", "googleapiclient": "google-api-python-client",
    # librerías indexadas
    "bs4": "beautifulsoup4", "factory": "factory-boy", "github": "pygithub", "hydra": "hydra-core",
    "cv2": "opencv-python", "PIL": "pillow", "fitz": "pymupdf", "dateutil": "python-dateutil",
    "docx": "python-docx", "pptx": "python-pptx", "skimage": "scikit-image", "sklearn": "scikit-learn",
    "websocket": "websocket-client", "Crypto": "pycryptodome", "Cryptodome": "pycryptodomex", "attr": "attrs",
    "mpl_toolkits": "matplotlib", "pytest_django": "pytest-django",
    # comunes fuera del índice
    "yaml": "pyyaml", "dotenv": "python-dotenv", "jwt": "pyjwt", "nacl": "pynacl", "OpenSSL": "pyopenssl",
    "MySQLdb": "mysqlclient", "jose": "python-jose",
}
VARIANT_PREFIXES = ("", "python-")  # PyPI ya trata igual `_`, `-` y `.` (§2.7)


def normalize(name: str) -> str:
    """Nombre de PyPI normalizado (PEP 503): `PyPDF2` y `pypdf_2` dan lo mismo que en la URL."""
    return re.sub(r"[-_.]+", "-", name).lower()


def is_stdlib(module: str) -> bool:
    return module.split(".")[0] in sys.stdlib_module_names


@lru_cache(maxsize=1)
def pipreqs_mapping() -> dict[str, str]:
    """`import:paquete`; con un import repetido gana la primera línea (68 casos, validador.md §2.4)."""
    mapping = {}
    for line in (DATA_DIR / "pipreqs_mapping").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            module, package = line.strip().split(":", 1)
            mapping.setdefault(module, package)
    return mapping


def candidates(module: str) -> tuple[list[str], str]:
    """Paquetes a probar en PyPI, en orden, y de dónde salieron (`manual`, `pipreqs` o `variants`)."""
    parts = module.split(".")
    for i in range(len(parts), 0, -1):
        if (key := ".".join(parts[:i])) in MANUAL:
            return [MANUAL[key]], "manual"
    top = parts[0]
    if top in pipreqs_mapping():
        return [pipreqs_mapping()[top]], "pipreqs"
    return [p + top for p in VARIANT_PREFIXES], "variants"
