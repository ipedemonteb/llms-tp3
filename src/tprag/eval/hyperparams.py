"""Grilla del estudio de hiperparámetros (etapa 3 de docs/agents/plan_experimentos_indice.md) y regla para elegir los
finalistas. Es la fuente única: la usan el runner (`run_hyperparam_study.py`) y el análisis (`analyze_hyperparams.py`).

Diseño: un factor a la vez (OAT) alrededor de la configuración base (S3, 1024 tokens, sin mínimo, tope 150, R1, E4).
Cada variante cambia un solo factor; la base es el centro y pertenece a todos los grupos. Con un dev de ~116 ítems una
grilla completa sería sobreajustar; si un factor mueve algo, se combinan después los mejores niveles.
"""

from dataclasses import dataclass

from tprag.index.build import IndexConfig
from tprag.split.splitters import SplitConfig
from tprag.split.store import config_name

SPLITTER = "s3"
BUDGETS = (500, 1000, 2000, 3000, 4000)  # tokens de contexto para la curva de Span Hit
MAIN_BUDGET = 2000  # el del generador y la métrica primaria


@dataclass(frozen=True)
class Variant:
    id: str
    factor: str  # "base" o el factor que cambia: size, min_tokens, cap, representation
    label: str
    size: int = 1024
    min_tokens: int = 0
    cap: int | None = 150
    representation: str = "r1"
    model: str = "e4"

    @property
    def chunks(self) -> str:
        """Carpeta de data/processed/<modo>/ (`s3-512`, `s3-1024-min64`)."""
        return config_name(SPLITTER, SplitConfig(max_tokens=self.size, min_tokens=self.min_tokens))

    def index_config(self, packages: list[str]) -> IndexConfig:
        return IndexConfig(chunks=self.chunks, model=self.model, representation=self.representation, cap=self.cap,
                           packages=packages)


# Orden de ejecución: primero lo que reusa el caché de vectores de la base (tope, mínimo), después lo que embebe textos
# nuevos, y al final el tamaño 2048 (el más pesado: ventana de 3072 tokens).
VARIANTS = (
    Variant("base", "base", "S3 · 1024 · R1 · tope 150"),
    Variant("cap75", "cap", "tope 75", cap=75),
    Variant("cap300", "cap", "tope 300", cap=300),
    Variant("nocap", "cap", "sin tope", cap=None),
    Variant("raw", "representation", "sin prefijo (raw)", representation="raw"),
    Variant("min64", "min_tokens", "mínimo 64", min_tokens=64),
    Variant("min128", "min_tokens", "mínimo 128", min_tokens=128),
    Variant("size512", "size", "tamaño 512", size=512),
    Variant("size256", "size", "tamaño 256", size=256),
    Variant("size2048", "size", "tamaño 2048", size=2048, model="e4l"),
)
BY_ID = {v.id: v for v in VARIANTS}
BASE = BY_ID["base"]

# Grupos para comparar niveles de un mismo factor (en el orden en que se dibujan); la base está en todos.
FACTORS = {
    "size": ("Tamaño de chunk (tokens)", ["size256", "size512", "base", "size2048"],
             {"size256": "256", "size512": "512", "base": "1024", "size2048": "2048"}),
    "min_tokens": ("Mínimo de tokens", ["base", "min64", "min128"], {"base": "0", "min64": "64", "min128": "128"}),
    "cap": ("Tope por librería", ["cap75", "base", "cap300", "nocap"],
            {"cap75": "75", "base": "150", "cap300": "300", "nocap": "sin tope"}),
    "representation": ("Representación", ["base", "raw"], {"base": "R1", "raw": "raw"}),
}


def finalists(rows: dict[str, dict], n: int = 3) -> list[str]:
    """Configuraciones que pasan a la evaluación con generación.

    `rows[id]` = {"span": Span Hit@2000, "pkg_verdict": veredicto de Pkg Hit@1 contra la base ("mejor", "empate" o
    "peor"; la base va como "empate"), "span_vs_best": veredicto de Span Hit@2000 contra la mejor ("empate" o "peor";
    la mejor va como "empate"), "tokens": tokens a embeber (costo)}.

    Regla (fijada antes de correr): solo entran las que no son peores que la base en Pkg Hit@1. De esas, el mejor en
    Span Hit@2000 va primero y lo siguen las que empatan con él, de la más barata a la más cara; si faltan para
    completar `n`, se suman las siguientes por Span Hit@2000.
    """
    eligible = {k: r for k, r in rows.items() if r["pkg_verdict"] != "peor"}
    if not eligible:
        return []
    best = max(eligible, key=lambda k: eligible[k]["span"])
    tied = sorted((k for k, r in eligible.items() if k != best and r["span_vs_best"] == "empate"),
                  key=lambda k: (eligible[k]["tokens"], -eligible[k]["span"]))
    rest = sorted((k for k in eligible if k != best and k not in tied), key=lambda k: -eligible[k]["span"])
    return ([best] + tied + rest)[:n]
