"""El sistema final (la Variante 7, elegida en el Ej. 4): una sola configuración, la misma para `fix_script.py` y la
interfaz (`app.py`). No tiene opciones: el usuario corre el sistema tal como quedó.

- Índice `s3-1024__e4__r1__cap150` (plan_cierre.md §0) con el reranker k1-rr30 (Qwen3 reordena los 30 primeros).
- Umbral sobre el puntaje del reranker, calibrado en dev (pasa ≥ 95% de los positivos).
- Contexto: hasta 2.000 tokens por import.
- gpt-4.1-mini devuelve el script completo (prompt v3); si la corrección no pasa la post-validación, un reintento y,
  si sigue mal, abstención.

Usa los mismos prompts y esquemas que `eval_generation.py --rerank k1-rr30 --llm gpt-4.1-mini --first-format script
--retry-format script --first-prompt v3`, así que comparte su caché de respuestas. Con `cache=False` (la interfaz), el
LLM y los embeddings de la consulta no usan los cachés de los estudios: cada corrección es una llamada real.
"""

from tprag.eval.metrics import CONTEXT_BUDGET
from tprag.generate.context import rerank_threshold
from tprag.generate.llm import CHAT_MODELS, CachedLLM, LiveLLM
from tprag.generate.pipeline import FixResult, Prepared, answer
from tprag.generate.schema import ScriptOutputV2
from tprag.index.build import IndexConfig, open_collection
from tprag.index.embeddings import MODELS, cached, model_embeddings

INDEX = IndexConfig(chunks="s3-1024", console_mode="outputs_as_comments", model="e4", representation="r1", cap=150,
                    strategy="round_robin")
COLLECTION = INDEX.collection_name()
RERANKER = "k1-rr30"
BUDGET = CONTEXT_BUDGET
LLM = "gpt-4.1-mini"
FORMAT = "script"  # el primer intento y el reintento devuelven el script completo
PROMPT = "v3"


def threshold() -> float:
    return rerank_threshold(COLLECTION, RERANKER)


def open_index(cache: bool = True):
    """El índice; sin `cache`, la consulta se embebe con el modelo cada vez."""
    return open_collection(COLLECTION, cached(INDEX.model) if cache else model_embeddings(MODELS[INDEX.model]))


def chat_llm(cache: bool = True) -> CachedLLM:
    llm = CachedLLM(CHAT_MODELS[LLM]) if cache else LiveLLM(CHAT_MODELS[LLM])
    llm.output_schema = ScriptOutputV2
    return llm


def correct(prepared: Prepared, llm) -> FixResult:
    """La llamada al LLM, la post-validación y, si hace falta, el reintento y la abstención."""
    return answer(prepared, llm, retry=FORMAT, retry_llm=llm, first=FORMAT, version=PROMPT)
