"""Corrige las dependencias deprecadas o inexistentes de un script (paso 8, docs/agents/generador.md). USA LA API
(el modelo de chat solo si algún import problemático pasa el umbral).

Uso:
    uv run python src/scripts/fix_script.py mi_script.py
    uv run python src/scripts/fix_script.py mi_script.py --chunks s3-1024 --threshold 0.42
    uv run --group local python src/scripts/fix_script.py mi_script.py --model e4 --rerank k1-rr30

Escribe al lado del script <nombre>.fixed.py (el script corregido; igual al original si no hubo cambios) y
<nombre>.report.json (una entrada por import: estado del validador, score del retrieval, decisión, paquete de
reemplazo, motivo y fuentes).
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from tprag.generate.context import dev_threshold, load_reranker, rerank_threshold
from tprag.generate.llm import CHAT_MODELS, CachedLLM
from tprag.generate.pipeline import fix
from tprag.index.build import REPRESENTATIONS, IndexConfig, open_collection
from tprag.index.embeddings import MODELS, cached
from tprag.split.select import DEFAULT_CAP, STRATEGIES
from tprag.split.store import ROOT
from tprag.validate.pypi import CachedPyPI


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("script", type=Path)
    parser.add_argument("--chunks", default="s3-1024", help="carpeta de data/processed/<modo>/")
    parser.add_argument("--console-examples", default="outputs_as_comments", help="carpeta de data/processed")
    parser.add_argument("--model", choices=sorted(MODELS), default="e1", help="modelo de embeddings del índice")
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="r1")
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP, help="0 = sin tope")
    parser.add_argument("--strategy", choices=STRATEGIES, default="round_robin")
    parser.add_argument("--llm", choices=sorted(CHAT_MODELS), default="gpt-4o-mini")
    parser.add_argument("--threshold", type=float, help="umbral de confianza (default: el calibrado en dev)")
    parser.add_argument("--budget", type=int, default=2000, help="tokens de contexto por import")
    parser.add_argument("--rerank", help="variante del reranker (p. ej. k1-rr30; necesita el grupo `local`); el umbral "
                                         "pasa a ser el de su puntaje, calibrado en dev")
    args = parser.parse_args()

    cfg = IndexConfig(chunks=args.chunks, console_mode=args.console_examples, model=args.model,
                      representation=args.representation, cap=args.cap or None, strategy=args.strategy)
    name = cfg.collection_name()
    reranker = load_reranker(args.rerank) if args.rerank else None
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = rerank_threshold(name, args.rerank) if args.rerank else dev_threshold(name)
    store = open_collection(name, cached(cfg.model))
    if store._collection.count() == 0:  # open_collection crea la colección vacía si no existe
        print(f"la colección {name} no existe o está vacía: armarla con build_index.py")
        return 1

    load_dotenv(ROOT / ".env")
    code = args.script.read_text(encoding="utf-8")
    try:
        result = fix(code, store, CachedPyPI(), CachedLLM(CHAT_MODELS[args.llm]), threshold, args.budget,
                     reranker)
    except SyntaxError as e:
        print(f"{args.script}: el script no parsea ({e})")
        return 1

    fixed = args.script.with_name(f"{args.script.stem}.fixed.py")
    report = args.script.with_name(f"{args.script.stem}.report.json")
    fixed.write_text(result.code, encoding="utf-8")
    report.write_text(json.dumps({"imports": result.report, "edit_errors": result.edit_errors}, indent=2,
                                 ensure_ascii=False) + "\n", encoding="utf-8")
    for e in result.report:
        if "decision" in e:
            target = f" -> {e['replacement_package']}" if e["replacement_package"] else ""
            print(f"{e['module']} ({e['status']}): {e['decision']}{target}")
    for msg in result.edit_errors:
        print(f"no se aplicaron los cambios: {msg}")
    print(f"-> {fixed}\n-> {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
