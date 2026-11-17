"""Modelo de chat del generador (generador.md §2.1) con caché en disco.

Temperatura 0 y salida estructurada (esquema JSON estricto, `schema.py`). El caché guarda la respuesta por (modelo,
hash de instrucciones + mensaje + esquema), como el de embeddings: repetir una corrida no vuelve a pagar, y `missing`
dice qué llamadas se pagarían antes de hacerlas (dry-run).
"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from tprag.split.store import ROOT
from tprag.split.tokens import token_counter

from .schema import GeneratorOutput

CACHE_PATH = ROOT / "data" / "generate" / "cache" / "llm.sqlite"
OUTPUT_TOKENS_ESTIMATE = 700  # reemplazos ~400 + reporte ~300 (generador.md §2.5)
# Tope de salida (2026-11-14): sin él, una respuesta en bucle llegó a 16.384 tokens (neg-useragent-01). No entra en la
# clave del caché: las 458 respuestas guardadas hasta entonces (sin tope) tienen como máximo 921 tokens, así que con
# el tope habrían salido idénticas. Las respuestas nuevas guardan `max_tokens`; las viejas no lo tienen (= sin tope).
MAX_OUTPUT_TOKENS = 2000


class TruncatedOutput(Exception):
    """La respuesta del LLM llegó a `MAX_OUTPUT_TOKENS` y no se puede leer (queda en el caché: no se vuelve a pagar)."""

    def __init__(self, raw: str):
        super().__init__("la respuesta del LLM se cortó por el tope de tokens de salida")
        self.raw = raw


@dataclass(frozen=True)
class ChatModel:
    id: str
    name: str
    api_name: str  # nombre en la API de OpenAI
    usd_per_m_input: float  # precios verificados en developers.openai.com/api/docs/pricing el 2026-11-08
    usd_per_m_output: float
    tokenizer: str = "o200k_base"

    def usd(self, input_tokens: int, output_tokens: int) -> float:
        return round((input_tokens * self.usd_per_m_input + output_tokens * self.usd_per_m_output) / 1e6, 6)


CHAT_MODELS = {
    "gpt-4o-mini": ChatModel("gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini", 0.15, 0.60),
}


def _schema(schema: type[BaseModel] = GeneratorOutput) -> str:
    return json.dumps(schema.model_json_schema(), sort_keys=True)


class CachedLLM:
    """Llama al modelo con (instrucciones, mensaje) y devuelve `GeneratorOutput`. El cliente se crea recién cuando
    hace falta (el dry-run funciona sin API key). Cuenta tokens reales de lo que se pagó. El esquema de salida y el
    tope son atributos: el juez (eval/judge.py) los cambia y hereda el resto."""

    output_schema: type[BaseModel] = GeneratorOutput
    max_tokens: int = MAX_OUTPUT_TOKENS

    def __init__(self, model: ChatModel, path: Path = CACHE_PATH):
        self.model = model
        self._client = None
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS responses (model TEXT, key TEXT, response TEXT, "
                         "PRIMARY KEY (model, key))")
        self.hits = self.misses = self.input_tokens = self.output_tokens = 0

    def _key(self, system: str, user: str) -> str:
        """El esquema de salida entra en la clave: una respuesta en formato "script" (`ScriptOutput`) no se confunde
        con una de reemplazos aunque el mensaje fuera el mismo. Con `GeneratorOutput`, la clave de siempre."""
        return hashlib.sha256("\x00".join((system, user, _schema(self.output_schema))).encode()).hexdigest()

    def _get(self, key: str) -> dict | None:
        row = self._db.execute("SELECT response FROM responses WHERE model = ? AND key = ?",
                               (self.model.name, key)).fetchone()
        return json.loads(row[0]) if row else None

    def missing(self, prompts: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Prompts distintos que todavía no están en el caché (los que se pagarían)."""
        unique = dict.fromkeys(prompts)
        return [p for p in unique if self._get(self._key(*p)) is None]

    def estimate(self, prompts: list[tuple[str, str]]) -> dict:
        """Tokens de entrada contados con el tokenizer del modelo (sin el esquema ni el formato de los mensajes, que
        suman poco) y salida estimada en `OUTPUT_TOKENS_ESTIMATE` por llamada."""
        count = token_counter(self.model.tokenizer)
        input_tokens = sum(count(s) + count(u) for s, u in prompts)
        output_tokens = OUTPUT_TOKENS_ESTIMATE * len(prompts)
        return {"calls": len(prompts), "input_tokens": input_tokens, "output_tokens_estimate": output_tokens,
                "usd": self.model.usd(input_tokens, output_tokens)}

    def _call(self, system: str, user: str) -> dict:
        if self._client is None:
            from langchain_openai import ChatOpenAI  # noqa: PLC0415  (pide la API key al construirse)

            chat = ChatOpenAI(model=self.model.api_name, temperature=0, max_tokens=self.max_tokens)
            self._client = chat.with_structured_output(self.output_schema, method="json_schema", strict=True,
                                                       include_raw=True)
        from openai import LengthFinishReasonError  # noqa: PLC0415

        try:
            out = self._client.invoke([("system", system), ("human", user)])
        except LengthFinishReasonError as e:  # llegó al tope: se guarda igual (se pagó) con el texto cortado
            usage = e.completion.usage
            return {"output": None, "error": "length", "raw": e.completion.choices[0].message.content or "",
                    "max_tokens": self.max_tokens,
                    "usage": {"input_tokens": usage.prompt_tokens if usage else 0,
                              "output_tokens": usage.completion_tokens if usage else 0}}
        if out["parsed"] is None:
            raise ValueError(f"respuesta del LLM que no respeta el esquema: {out['parsing_error']}")
        usage = out["raw"].usage_metadata or {}
        return {"output": out["parsed"].model_dump(), "max_tokens": self.max_tokens,
                "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)}}

    def peek(self, system: str, user: str) -> BaseModel | None:
        """La respuesta guardada en el caché, sin llamar a la API; None si no está. Para el dry-run del reintento, que
        necesita la primera respuesta para saber qué se reintentaría. Levanta `TruncatedOutput` como `__call__`."""
        response = self._get(self._key(system, user))
        if response is None:
            return None
        if response.get("error") == "length":
            raise TruncatedOutput(response["raw"])
        return self.output_schema.model_validate(response["output"])

    def __call__(self, system: str, user: str) -> BaseModel:
        """Levanta `TruncatedOutput` si la respuesta (nueva o del caché) se cortó por el tope de salida."""
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
        if response.get("error") == "length":
            raise TruncatedOutput(response["raw"])
        return self.output_schema.model_validate(response["output"])

    @property
    def usd(self) -> float:
        return self.model.usd(self.input_tokens, self.output_tokens)
