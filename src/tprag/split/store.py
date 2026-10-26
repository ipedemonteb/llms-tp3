"""Escritura y lectura de `data/processed/<modo>/<config>/chunks.jsonl` (splitters.md §3.7)."""

import json
from dataclasses import asdict
from pathlib import Path

from .chunk import Chunk
from .splitters import SplitConfig
from .tokens import DEFAULT_TOKENIZER

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"


def config_name(splitter: str, cfg: SplitConfig) -> str:
    """`s3-1024`; con mínimo, `s3-1024-min50`; con otro tokenizer, `...-<tokenizer>` (los chunks cambian, §3.6)."""
    name = f"{splitter}-{cfg.max_tokens}"
    if cfg.min_tokens and splitter == "s3":
        name += f"-min{cfg.min_tokens}"
    if cfg.tokenizer != DEFAULT_TOKENIZER:
        name += "-" + cfg.tokenizer.replace("hf:", "").replace("/", "_")
    return name


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = sorted(chunks, key=lambda c: (c.package, c.source_path, c.start, c.end))
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
