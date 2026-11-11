"""Estadísticas de los chunks de una configuración de índice (tamaños, bloques de código partidos, truncamiento).
No usa API ni embeddings; con un modelo local carga solo su tokenizer.

Uso (mismos flags que build_index.py):
    uv run python src/scripts/chunk_stats.py --chunks s3-1024 --model e4 --packages requests httpx

Escribe results/study/<colección>__chunks.json (versionado, es chico).
"""

import argparse
import json
from dataclasses import asdict
from datetime import datetime

from tprag.eval.chunk_stats import chunk_stats, cut_code_stats, model_token_stats
from tprag.index.build import REPRESENTATIONS, IndexConfig, select_chunks, to_document
from tprag.index.embeddings import MODELS
from tprag.normalize.store import load_normalized
from tprag.split.select import DEFAULT_CAP, STRATEGIES
from tprag.split.store import PROCESSED_DIR, ROOT, load_chunks

STUDY_DIR = ROOT / "results" / "study"


def tokenizer_counter(name: str):
    """Cuenta tokens con el tokenizer del modelo, o None si `transformers` no está instalado."""
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415  (grupo opcional `local`)
    except ImportError:
        return None
    tokenizer = AutoTokenizer.from_pretrained(name)
    return lambda texts: [len(ids) for ids in tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="s3-1024", help="carpeta de data/processed/<modo>/")
    parser.add_argument("--console-examples", default="outputs_as_comments", help="carpeta de data/processed")
    parser.add_argument("--model", choices=sorted(MODELS), default="e1")
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="r1")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP, help="0 = sin tope")
    parser.add_argument("--strategy", choices=STRATEGIES, default="round_robin")
    parser.add_argument("--packages", nargs="*", help="solo estas librerías")
    args = parser.parse_args()

    cfg = IndexConfig(chunks=args.chunks, console_mode=args.console_examples, model=args.model,
                      representation=args.representation, cap=args.cap or None, strategy=args.strategy,
                      packages=args.packages)
    model, name = MODELS[cfg.model], cfg.collection_name()
    every = [c for c in load_chunks(PROCESSED_DIR / cfg.console_mode / cfg.chunks / "chunks.jsonl")
             if not cfg.packages or c.package in cfg.packages]
    indexed = select_chunks(cfg)
    docs = load_normalized(cfg.console_mode, packages=cfg.packages)

    counter = tokenizer_counter(model.name) if model.provider == "local" else None
    model_tokens = None
    if counter:
        model_tokens = model_token_stats([to_document(c).page_content for c in indexed], counter, model.max_seq_length)

    report = {
        "collection": name, "date": datetime.now().isoformat(timespec="seconds"), "config": asdict(cfg),
        "all_chunks": chunk_stats(every),  # lo que produce el splitter, antes del filtro y el tope
        "indexed": chunk_stats(indexed),  # lo que entra al índice
        "code_blocks": cut_code_stats(docs, every),
        "model_tokens": model_tokens,  # tokens del embedder sobre lo indexado (prefijo + chunk)
    }
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    out = STUDY_DIR / f"{name}__chunks.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cb = report["code_blocks"]["python"]
    print(f"{name}: {report['indexed']['chunks']} chunks indexados, mediana {report['indexed']['n_tokens']['median']} "
          f"tokens; bloques Python no enteros {cb['not_intact_rate']:.1%}"
          + (f"; truncados por el modelo {model_tokens['over_max']:.1%}" if model_tokens else "") + f" -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
