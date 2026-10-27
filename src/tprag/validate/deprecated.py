"""Lista manual de paquetes deprecados (validador.md §2.2, §2.6). Las claves son nombres de PyPI normalizados.

El reemplazo esperado no es la respuesta del sistema (la propone el RAG): marca el import y sirve para evaluar.
Origen: los pares de plan_dataset_crudo.md §5.3, sin retrying (se retomó en 2025, §2.6).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Deprecation:
    replacement: list[str]  # paquetes de PyPI o módulos de la stdlib (`tomllib`)
    reason: str


DEPRECATED = {
    "pypdf2": Deprecation(["pypdf"], "abandonado; el proyecto siguió como pypdf"),
    "pypdf4": Deprecation(["pypdf"], "abandonado; el proyecto siguió como pypdf"),
    "fuzzywuzzy": Deprecation(["rapidfuzz"], "sin mantenimiento"),
    "nose": Deprecation(["pytest"], "abandonado"),
    "xlrd": Deprecation(["openpyxl"], "desde la 2.0 solo lee .xls; para .xlsx, openpyxl"),
    "python-jose": Deprecation(["authlib"], "sin mantenimiento"),
    "passlib": Deprecation(["cryptography"], "sin mantenimiento"),
    "oauth2client": Deprecation(["google-auth"], "deprecado por Google"),
    "toml": Deprecation(["tomllib"], "reemplazado por la stdlib (tomllib, Python 3.11)"),
    "pytz": Deprecation(["zoneinfo"], "reemplazado por la stdlib (zoneinfo, Python 3.9)"),
    "mock": Deprecation(["unittest.mock"], "integrado a la stdlib (unittest.mock)"),
    # Nunca se activan desde un import (validador.md §2.2): `sklearn` resuelve a scikit-learn y `Crypto` a
    # pycryptodome. Quedan como registro y por si el validador lee requirements.txt más adelante.
    "sklearn": Deprecation(["scikit-learn"], "paquete vacío en PyPI; el correcto es scikit-learn"),
    "pycrypto": Deprecation(["pycryptodome", "cryptography"], "abandonado y con vulnerabilidades conocidas"),
}
