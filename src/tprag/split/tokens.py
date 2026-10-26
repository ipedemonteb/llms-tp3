"""Conteo de tokens (splitters.md §3.6). Default `cl100k_base`, el tokenizer de text-embedding-3 (E1).

Con el mismo tokenizer, los chunks son iguales para todos los modelos de embeddings. Con `hf:<modelo>` se cuenta con
el tokenizer de ese modelo (necesita `transformers`, del grupo opcional): los chunks pasan a depender del modelo.
"""

from collections.abc import Callable
from functools import lru_cache

DEFAULT_TOKENIZER = "cl100k_base"

TokenCounter = Callable[[str], int]


@lru_cache(maxsize=None)
def token_counter(name: str = DEFAULT_TOKENIZER) -> TokenCounter:
    if name.startswith("hf:"):
        from transformers import AutoTokenizer  # noqa: PLC0415  (opcional)

        tok = AutoTokenizer.from_pretrained(name[3:])
        return lambda text: len(tok.encode(text, add_special_tokens=False))
    import tiktoken  # noqa: PLC0415

    enc = tiktoken.get_encoding(name)
    # disallowed_special=(): la doc puede contener textos como "<|endoftext|>" (tokenizers de transformers)
    return lambda text: len(enc.encode(text, disallowed_special=()))
