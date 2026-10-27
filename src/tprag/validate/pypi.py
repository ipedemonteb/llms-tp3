"""Consulta a la API JSON de PyPI con caché en disco (validador.md §2.7). Gratis; no usa la API de OpenAI.

Una consulta devuelve `PackageInfo` si el paquete existe, `None` si da 404, o levanta `PyPIError` si no se pudo
verificar (sin red, timeout, PyPI caído): el validador lo convierte en `unknown`, nunca en `not_found`.
"""

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass

from tprag.split.store import ROOT

from .packages import normalize

CACHE_DIR = ROOT / "data" / "validate" / "pypi_cache"
URL = "https://pypi.org/pypi/{}/json"
TTL_FOUND = 7 * 86400
TTL_MISSING = 86400  # un nombre que hoy no existe puede registrarse mañana (slopsquatting)


class PyPIError(Exception):
    pass


@dataclass
class PackageInfo:
    name: str
    classifiers: list[str]
    last_release: str | None  # fecha ISO de la última subida
    vulnerabilities: list[str]  # ids de los avisos de la última versión


Fetch = Callable[[str], PackageInfo | None]


def parse(data: dict) -> PackageInfo:
    """Se queda solo con los campos que usa el validador."""
    dates = [f["upload_time_iso_8601"] for files in data.get("releases", {}).values() for f in files]
    return PackageInfo(name=data["info"]["name"], classifiers=data["info"].get("classifiers") or [],
                       last_release=max(dates)[:10] if dates else None,
                       vulnerabilities=[v["id"] for v in data.get("vulnerabilities") or []])


def fetch_pypi(name: str, timeout: float = 10, retries: int = 1) -> PackageInfo | None:
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(URL.format(normalize(name)), timeout=timeout) as resp:
                return parse(json.load(resp))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            error = e
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            error = e
        if attempt < retries:
            time.sleep(1)
    raise PyPIError(f"{name}: {error}")


class CachedPyPI:
    """Envuelve una función de consulta con caché en disco (un JSON por paquete). Con `offline`, lo que no está en el
    caché (o venció) levanta `PyPIError`."""

    def __init__(self, fetch: Fetch = fetch_pypi, cache_dir=CACHE_DIR, offline: bool = False,
                 now: Callable[[], float] = time.time):
        self._fetch, self._dir, self.offline, self._now = fetch, cache_dir, offline, now

    def __call__(self, name: str) -> PackageInfo | None:
        path = self._dir / f"{normalize(name)}.json"
        if path.exists():
            entry = json.loads(path.read_text(encoding="utf-8"))
            ttl = TTL_FOUND if entry["info"] else TTL_MISSING
            if self._now() - entry["fetched_at"] < ttl:
                return PackageInfo(**entry["info"]) if entry["info"] else None
        if self.offline:
            raise PyPIError(f"{name}: no está en el caché (modo offline)")
        info = self._fetch(name)
        self._dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_at": self._now(), "info": asdict(info) if info else None}),
                        encoding="utf-8")
        return info
