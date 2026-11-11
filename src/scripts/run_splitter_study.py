"""Etapa 1 del plan de experimentos (docs/agents/plan_experimentos_indice.md): para cada splitter arma el índice con
un embedder local sobre las librerías de data/eval/corpus_subset.json, evalúa el retrieval en dev y calcula las
estadísticas de sus chunks. No usa API.

Uso (hace falta el grupo de dependencias `local`):
    uv run --group local python src/scripts/run_splitter_study.py --dry-run     # solo muestra qué falta
    uv run --group local python src/scripts/run_splitter_study.py               # los seis splitters, en orden
    uv run --group local python src/scripts/run_splitter_study.py --only s3 s2  # solo algunos
    uv run --group local python src/scripts/run_splitter_study.py --force       # rehace aunque ya esté hecho

Cuidados con la Mac (16 GB de memoria compartida con la GPU):
- Cada paso de cada splitter corre en un proceso aparte, así la memoria se libera entre uno y otro.
- Los vectores se calculan de a --warm-batch textos y se guardan en el caché después de cada lote (--pause segundos
  entre lotes); entre splitters hay --cooldown segundos de descanso.
- `caffeinate` evita que la Mac se duerma mientras corre.

Para retomar tras un corte alcanza con volver a lanzarlo: salta los pasos que ya tienen su resultado (el de la
evaluación solo si es del mismo dataset) y del índice a medias solo se rehace el último lote.

Todo queda en results/: reportes de índice (results/index/), evaluación con los 50 chunks de cada target
(results/eval/), estadísticas de chunks y manifiesto (results/study/), registro de cada paso (results/study/logs/).
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from importlib import metadata
from pathlib import Path

from tprag.eval.retrieval import dataset_info
from tprag.eval.verify import EVAL_DIR
from tprag.index.build import IndexConfig
from tprag.index.embeddings import MODELS
from tprag.split.store import ROOT

SPLITTERS = ("s3", "s2", "s0", "s1", "char", "tok")  # primero los baratos; s2 casi sale gratis tras s3 (caché)
SIZE = 1024
SCRIPTS = ROOT / "src" / "scripts"
INDEX_DIR, EVAL_RESULTS, STUDY_DIR = ROOT / "results" / "index", ROOT / "results" / "eval", ROOT / "results" / "study"
MANIFEST = STUDY_DIR / "splitters_manifest.json"
STEPS = ("index", "eval", "stats")


def environment() -> dict:
    env = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version()}
    for package in ("torch", "sentence-transformers", "transformers", "chromadb", "tiktoken"):
        try:
            env[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            env[package] = None
    try:
        import torch  # noqa: PLC0415

        env["mps"] = torch.backends.mps.is_available()
    except ImportError:
        env["mps"] = None
    return env


def git_dirty() -> bool | None:
    """¿Hay cambios sin commitear? Si sí, el commit del manifiesto no alcanza para reproducir la corrida."""
    try:
        return bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
                                   check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def outputs(name: str, split: str) -> dict[str, Path]:
    return {"index": INDEX_DIR / f"{name}.json", "eval": EVAL_RESULTS / f"{name}__qA__{split}.json",
            "stats": STUDY_DIR / f"{name}__chunks.json"}


def is_done(step: str, path: Path, force: bool) -> bool:
    """El paso ya tiene su resultado. La evaluación solo vale si salió del mismo dataset que hay ahora."""
    if force or not path.exists():
        return False
    if step == "eval":
        if not path.with_suffix(".targets.jsonl").exists():  # el detalle se escribe justo después del resumen
            return False
        try:
            return json.loads(path.read_text())["dataset"]["sha256"] == dataset_info()["sha256"]
        except (KeyError, ValueError):
            return False
    return True


def command(step: str, chunks: str, args, packages: list[str]) -> list[str]:
    common = ["--chunks", chunks, "--model", args.model, "--packages", *packages]
    if step == "index":
        return [sys.executable, str(SCRIPTS / "build_index.py"), *common, "--warm-batch", str(args.warm_batch),
                "--pause", str(args.pause)]
    if step == "eval":
        return [sys.executable, str(SCRIPTS / "eval_retrieval.py"), *common, "--split", args.split]
    return [sys.executable, str(SCRIPTS / "chunk_stats.py"), *common]


def save_manifest(manifest: dict) -> None:
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(MANIFEST)  # nunca queda a medias si se corta


def load_manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text())
    except (OSError, ValueError):
        return {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=SPLITTERS, help="solo estos splitters (en el orden dado)")
    parser.add_argument("--model", choices=[m for m in MODELS if MODELS[m].provider == "local"], default="e4")
    parser.add_argument("--packages", nargs="*", help="default: las librerías de data/eval/corpus_subset.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev", help="test solo para el reporte final")
    parser.add_argument("--warm-batch", type=int, default=200, help="textos por lote de vectores (se guarda cada lote)")
    parser.add_argument("--pause", type=float, default=3, help="segundos de pausa entre lotes de vectores")
    parser.add_argument("--cooldown", type=float, default=60, help="segundos de descanso entre splitters")
    parser.add_argument("--force", action="store_true", help="rehacer aunque el resultado ya exista")
    parser.add_argument("--dry-run", action="store_true", help="solo muestra qué pasos faltan")
    args = parser.parse_args()

    packages = args.packages or json.loads((EVAL_DIR / "corpus_subset.json").read_text(encoding="utf-8"))["packages"]
    plan = []
    for splitter in args.only or SPLITTERS:
        chunks = f"{splitter}-{SIZE}"
        name = IndexConfig(chunks=chunks, model=args.model, packages=packages).collection_name()
        plan.append((splitter, chunks, name, outputs(name, args.split)))

    print(f"{len(plan)} splitters con {args.model} ({MODELS[args.model].name}), {len(packages)} librerías, {args.split}")
    for splitter, _, name, paths in plan:
        state = {s: ("hecho" if is_done(s, paths[s], args.force) else "falta") for s in STEPS}
        print(f"  {splitter:<5} {name}: " + ", ".join(f"{s} {v}" for s, v in state.items()))
    if args.dry_run:
        return 0

    if sys.platform == "darwin" and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])  # sin dormir mientras corre este proceso

    manifest = load_manifest()
    manifest.update({
        "study": "splitters", "model": args.model, "model_name": MODELS[args.model].name,
        "model_config": asdict(MODELS[args.model]), "split": args.split, "size": SIZE, "n_packages": len(packages),
        "packages_file": "data/eval/corpus_subset.json" if not args.packages else None, "packages": packages,
        "dataset": dataset_info(), "git_dirty": git_dirty(), "environment": environment(), "settings": {
            "warm_batch": args.warm_batch, "pause": args.pause, "cooldown": args.cooldown},
    })
    manifest.setdefault("steps", {})
    manifest.setdefault("started", datetime.now().isoformat(timespec="seconds"))
    (STUDY_DIR / "logs").mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"}

    for i, (splitter, chunks, name, paths) in enumerate(plan, start=1):
        worked = False
        for step in STEPS:
            if is_done(step, paths[step], args.force):
                continue
            log = STUDY_DIR / "logs" / f"{name}__{step}.log"
            cmd = command(step, chunks, args, packages)
            print(f"[{i}/{len(plan)}] {splitter}: {step} ... (registro: {log.relative_to(ROOT)})", flush=True)
            start = time.time()
            with log.open("a", encoding="utf-8") as fh:
                fh.write(f"\n=== {datetime.now().isoformat(timespec='seconds')} {' '.join(cmd[:3])} ...\n")
                fh.flush()
                code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
            seconds = round(time.time() - start, 1)
            manifest["steps"].setdefault(name, {})[step] = {
                "status": "ok" if code == 0 else f"error {code}", "seconds": seconds,
                "finished": datetime.now().isoformat(timespec="seconds"), "log": str(log.relative_to(ROOT))}
            save_manifest(manifest)
            if code != 0:
                print(f"  falló ({code}) tras {seconds} s; ver {log}. Al relanzar se retoma desde este paso.")
                return 1
            print(f"  ok en {seconds} s", flush=True)
            worked = True
        if worked and i < len(plan) and args.cooldown:
            print(f"  descanso de {args.cooldown:.0f} s", flush=True)
            time.sleep(args.cooldown)

    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    save_manifest(manifest)
    print(f"listo. Manifiesto: {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
