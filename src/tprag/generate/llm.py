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

from tprag.split.store import ROOT
from tprag.split.tokens import token_counter

from .schema import GeneratorOutput

CACHE_PATH = ROOT / "data" / "generate" / "cache" / "llm.sqlite"
OUTPUT_TOKENS_ESTIMATE = 700  # reemplazos ~400 + reporte ~300 (generador.md §2.5)


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


def _schema() -> str:
    return json.dumps(GeneratorOutput.model_json_schema(), sort_keys=True)


class CachedLLM:
    """Llama al modelo con (instrucciones, mensaje) y devuelve `GeneratorOutput`. El cliente se crea recién cuando
    hace falta (el dry-run funciona sin API key). Cuenta tokens reales de lo que se pagó."""

    def __init__(self, model: ChatModel, path: Path = CACHE_PATH):
        self.model = model
        self._client = None
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.execute("CREATE TABLE IF NOT EXISTS responses (model TEXT, key TEXT, response TEXT, "
                         "PRIMARY KEY (model, key))")
        self.hits = self.misses = self.input_tokens = self.output_tokens = 0

    def _key(self, system: str, user: str) -> str:
        return hashlib.sha256("\x00".join((system, user, _schema())).encode()).hexdigest()

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

            chat = ChatOpenAI(model=self.model.api_name, temperature=0)
            self._client = chat.with_structured_output(GeneratorOutput, method="json_schema", strict=True,
                                                       include_raw=True)
        out = self._client.invoke([("system", system), ("human", user)])
        if out["parsed"] is None:
            raise ValueError(f"respuesta del LLM que no respeta el esquema: {out['parsing_error']}")
        usage = out["raw"].usage_metadata or {}
        return {"output": out["parsed"].model_dump(),
                "usage": {"input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0)}}

    def __call__(self, system: str, user: str) -> GeneratorOutput:
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
        return GeneratorOutput.model_validate(response["output"])

    @property
    def usd(self) -> float:
        return self.model.usd(self.input_tokens, self.output_tokens)
