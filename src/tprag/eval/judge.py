"""Answer Relevance con LLM (metricas.md §3.3): ¿el script corregido sigue haciendo la misma tarea?

- **(a) Técnica de clase (RAGAS):** el juez escribe 3 descripciones de qué hace el script corregido; Answer Relevance =
  coseno promedio entre sus embeddings (e1, `text-embedding-3-small`) y el de `task` del ítem.
- **(b) Juez directo (principal):** el juez compara el script original con el modificado y dice si conserva la
  funcionalidad (sí = 1, parcial = 0,5, no = 0), qué se perdió y por qué.

Juez: gpt-4o-mini, temperatura 0, salida estructurada, con caché propio (`judge.sqlite`): juzgar de nuevo no vuelve a
pagar. Mitigación de la autopreferencia (el generador es el mismo modelo): el juez no sabe que el script modificado lo
escribió un LLM. Los mismos prompts se exportan para validar con Claude Opus 5.5 (metricas.md §3.8).
"""

import hashlib
import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from tprag.generate.llm import CACHE_PATH, ChatModel, CachedLLM
from tprag.split.tokens import token_counter

JUDGE_CACHE = CACHE_PATH.with_name("judge.sqlite")


class Descriptions(BaseModel):
    descriptions: list[str] = Field(description="Tres descripciones distintas, de una oración cada una, de la tarea "
                                                "que resuelve el script (qué hace, no cómo)")


class Preservation(BaseModel):
    """El orden importa: primero lo que cambió, después el veredicto (como en el generador, schema.py)."""

    lost_or_changed: list[str] = Field(description="Funcionalidades del script original que el modificado pierde, "
                                                   "cambia o deja sin implementar, con el número de línea; vacía si "
                                                   "no hay ninguna")
    preserves: Literal["si", "parcial", "no"] = Field(description="¿El script modificado hace la misma tarea que el "
                                                                  "original?")
    reason: str = Field(description="Motivo en castellano, una o dos oraciones")


PRESERVE_SCORE = {"si": 1.0, "parcial": 0.5, "no": 0.0}

DESCRIBE_SYSTEM = (
    "Vas a leer un script de Python. Escribí tres descripciones distintas, de una oración cada una, de la tarea que "
    "resuelve: qué hace para quien lo usa, no cómo está programado ni qué librerías usa. Escribí en el mismo idioma "
    "que los comentarios y docstrings del script (castellano o inglés).")

PRESERVE_SYSTEM = (
    "Vas a comparar dos versiones de un script de Python. En la versión modificada se reemplazó un import (el módulo "
    "indicado) por otra librería y se adaptó el código que lo usaba. No evalúes si la librería elegida es buena ni el "
    "estilo: evaluá si el script modificado sigue haciendo la misma tarea que el original. Fijate si alguna "
    "funcionalidad se perdió, cambió de comportamiento o quedó sin implementar (por ejemplo, código borrado, `pass`, "
    "`TODO`, valores fijos en lugar de cálculos). Respondé `si` si hace lo mismo, `parcial` si hace lo principal pero "
    "pierde o cambia algo, y `no` si ya no hace la tarea.")


def describe_prompt(fixed: str) -> tuple[str, str]:
    return DESCRIBE_SYSTEM, f"```python\n{fixed}\n```"


def preserve_prompt(original: str, fixed: str, modules: list[str]) -> tuple[str, str]:
    return PRESERVE_SYSTEM, (f"Módulo reemplazado: {', '.join(modules)}\n\n"
                             f"Script original:\n```python\n{original}\n```\n\n"
                             f"Script modificado:\n```python\n{fixed}\n```")


@dataclass(frozen=True)
class Judge:
    name: str  # a | b
    schema: type[BaseModel]
    output_tokens_estimate: int  # metricas.md §5: ~100 (a), ~180 (b)


JUDGES = {"a": Judge("a", Descriptions, 100), "b": Judge("b", Preservation, 180)}


class JudgeLLM(CachedLLM):
    """El caché del generador (`CachedLLM`) con otro esquema de salida. La clave incluye el esquema, así que (a) y (b)
    no se mezclan entre sí ni con el generador; además va en otro archivo."""

    def __init__(self, model: ChatModel, judge: Judge, path: Path = JUDGE_CACHE):
        super().__init__(model, path)
        self.judge = judge

    def _key(self, system: str, user: str) -> str:
        schema = json.dumps(self.judge.schema.model_json_schema(), sort_keys=True)
        return hashlib.sha256("\x00".join((system, user, schema)).encode()).hexdigest()

    def estimate(self, prompts: list[tuple[str, str]]) -> dict:
        count = token_counter(self.model.tokenizer)
        input_tokens = sum(count(s) + count(u) for s, u in prompts)
        output_tokens = self.judge.output_tokens_estimate * len(prompts)
        return {"calls": len(prompts), "input_tokens": input_tokens, "output_tokens_estimate": output_tokens,
                "usd": self.model.usd(input_tokens, output_tokens)}

    def _call(self, system: str, user: str) -> dict:
        if self._client is None:
            from langchain_openai import ChatOpenAI  # noqa: PLC0415  (pide la API key al construirse)

            chat = ChatOpenAI(model=self.model.api_name, temperature=0)
            self._client = chat.with_structured_output(self.judge.schema, method="json_schema", strict=True,
                                                       include_raw=True)
        out = self._client.invoke([("system", system), ("human", user)])
        if out["parsed"] is None:
            raise ValueError(f"respuesta del juez que no respeta el esquema: {out['parsing_error']}")
        usage = out["raw"].usage_metadata or {}
        return {"output": out["parsed"].model_dump(),
                "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)}}

    def __call__(self, system: str, user: str) -> BaseModel:
        key = self._key(system, user)
        response = self._get(key)
        if response is None:
            response = self._call(system, user)
            self._db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?)",
                             (self.model.name, key, json.dumps(response, ensure_ascii=False)))
            self._db.commit()
            self.misses += 1
            self.input_tokens += response["usage"]["input_tokens"]
            self.output_tokens += response["usage"]["output_tokens"]
        else:
            self.hits += 1
        return self.judge.schema.model_validate(response["output"])


def cosine(u: list[float], v: list[float]) -> float:
    dot = sum(a * b for a, b in zip(u, v))
    return dot / (sqrt(sum(a * a for a in u)) * sqrt(sum(b * b for b in v)))


def relevance_a(descriptions: list[str], task: str, embed) -> float:
    """Coseno promedio entre cada descripción y `task`. `embed`: lista de textos → lista de vectores."""
    vectors = embed([task, *descriptions])
    return sum(cosine(vectors[0], v) for v in vectors[1:]) / len(descriptions)


def judged_records(records: list[dict], originals: dict[str, str], ids: set[str] | None = None) -> list[dict]:
    """Solo los scripts cambiados (Answer Relevance no aplica a una abstención); con `ids`, solo esos."""
    return [r for r in records if r["code"] != originals[r["id"]] and (ids is None or r["id"] in ids)]
