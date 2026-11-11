"""Etapas 1 y 2 del plan de experimentos (docs/agents/plan_experimentos_indice.md): para cada combinación de
embedder local y splitter arma el índice sobre las librerías de data/eval/corpus_subset.json, evalúa el retrieval en
dev y calcula las estadísticas de sus chunks. No usa API.

- `--study splitters` (etapa 1, default): un embedder (e4) y los seis splitters. Manifiesto: splitters_manifest.json.
- `--study embedders` (etapa 2): los embedders e3 y e5 sobre s3 y s2 (e4 ya está hecho en la etapa 1). Manifiesto:
  embedders_manifest.json. Orden: un modelo completo (s3 y después s2, que casi sale gratis por el caché) y recién
  después el siguiente.

Uso (hace falta el grupo de dependencias `local`):
    uv run --group local python src/scripts/run_splitter_study.py --dry-run     # solo muestra qué falta
    uv run --group local python src/scripts/run_splitter_study.py               # etapa 1: los seis splitters, en orden
    uv run --group local python src/scripts/run_splitter_study.py --only s3 s2  # solo algunos
    uv run --group local python src/scripts/run_splitter_study.py --force       # rehace aunque ya esté hecho
    uv run --group local python src/scripts/run_splitter_study.py --study embedders --dry-run
    uv run --group local python src/scripts/run_splitter_study.py --study embedders --smoke   # prueba de humo
    uv run --group local python src/scripts/run_splitter_study.py --study embedders          # las cuatro corridas
    uv run --group local python src/scripts/run_splitter_study.py --study embedders --models e5   # un solo modelo

`--smoke` embebe 200 textos de s3 con cada modelo, mide tiempo y memoria y extrapola al total; no escribe índices ni
toca el caché de los experimentos (usa uno temporal). Sirve para decidir el orden y si bge-m3 entra en un tiempo
razonable antes de lanzar la corrida larga.

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
import re
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
STEPS = ("index", "eval", "stats")
STUDIES = {  # defaults de cada etapa: modelos, splitters y manifiesto
    "splitters": {"models": ["e4"], "splitters": list(SPLITTERS), "manifest": STUDY_DIR / "splitters_manifest.json"},
    "embedders": {"models": ["e3", "e5"], "splitters": ["s3", "s2"],
                  "manifest": STUDY_DIR / "embedders_manifest.json"},
}


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


def timed() -> list[str]:
    """`/usr/bin/time -l` (macOS) imprime al final del registro la memoria máxima usada por el paso."""
    return ["/usr/bin/time", "-l"] if sys.platform == "darwin" and Path("/usr/bin/time").exists() else []


def peak_memory_mb(log: Path) -> float | None:
    """Última 'maximum resident set size' del registro (en macOS está en bytes), en MB."""
    found = re.findall(r"(\d+)\s+maximum resident set size", log.read_text(encoding="utf-8", errors="ignore"))
    return round(int(found[-1]) / 2**20, 1) if found else None


def command(step: str, chunks: str, model: str, args, packages: list[str]) -> list[str]:
    common = ["--chunks", chunks, "--model", model, "--packages", *packages]
    if step == "index":
        return [*timed(), sys.executable, str(SCRIPTS / "build_index.py"), *common, "--warm-batch",
                str(args.warm_batch), "--pause", str(args.pause)]
    if step == "eval":
        return [sys.executable, str(SCRIPTS / "eval_retrieval.py"), *common, "--split", args.split]
    return [sys.executable, str(SCRIPTS / "chunk_stats.py"), *common]


def save_manifest(manifest: dict, path: Path) -> None:
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)  # nunca queda a medias si se corta


def load_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


SMOKE_TEXTS = 200


def smoke_child(model_id: str, chunks: str, packages: list[str], n: int) -> int:
    """Proceso aparte por modelo (así el pico de memoria es el suyo). Embebe n textos repartidos parejo, sin caché."""
    import resource  # noqa: PLC0415

    from tprag.index.build import select_chunks, to_document  # noqa: PLC0415
    from tprag.index.embeddings import LocalEmbeddings  # noqa: PLC0415

    texts = [to_document(c).page_content for c in select_chunks(IndexConfig(chunks=chunks, model=model_id,
                                                                            packages=packages))]
    step = max(1, len(texts) // n)
    sample = texts[::step][:n]
    emb = LocalEmbeddings(MODELS[model_id])
    start = time.time()
    emb.embed_documents(sample[:4])  # carga el modelo y calienta
    load = time.time() - start
    start = time.time()
    emb.embed_documents(sample)
    seconds = time.time() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 2**10)
    print(json.dumps({"model": model_id, "sample": len(sample), "total_texts": len(texts), "load_seconds": round(load, 1),
                      "seconds": round(seconds, 1), "peak_mb": round(peak),
                      "estimated_minutes": round(seconds / len(sample) * len(texts) / 60, 1)}))
    return 0


def smoke(models: list[str], chunks: str, packages: list[str], env: dict) -> int:
    """Prueba de humo: tiempo por texto y memoria de cada modelo, extrapolados al total. No escribe índices."""
    results = []
    for model_id in models:
        print(f"{model_id} ({MODELS[model_id].name}): {SMOKE_TEXTS} textos de {chunks} ...", flush=True)
        done = subprocess.run([sys.executable, str(Path(__file__)), "--smoke-child", model_id, "--chunks", chunks,
                               "--packages", *packages], cwd=ROOT, env=env, capture_output=True, text=True)
        if done.returncode != 0:
            print(done.stderr[-2000:])
            return 1
        results.append(json.loads(done.stdout.strip().splitlines()[-1]))
        print("  ", results[-1], flush=True)
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    out = STUDY_DIR / "embedders_smoke.json"
    out.write_text(json.dumps({"date": datetime.now().isoformat(timespec="seconds"), "environment": environment(),
                               "results": results}, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    local = [m for m in MODELS if MODELS[m].provider == "local"]
    parser.add_argument("--study", choices=sorted(STUDIES), default="splitters")
    parser.add_argument("--only", nargs="*", choices=SPLITTERS, help="solo estos splitters (en el orden dado)")
    parser.add_argument("--models", nargs="*", choices=local, help="default: los de la etapa (e4 / e3 e5)")
    parser.add_argument("--packages", nargs="*", help="default: las librerías de data/eval/corpus_subset.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev", help="test solo para el reporte final")
    parser.add_argument("--warm-batch", type=int, default=200, help="textos por lote de vectores (se guarda cada lote)")
    parser.add_argument("--pause", type=float, default=3, help="segundos de pausa entre lotes de vectores")
    parser.add_argument("--cooldown", type=float, default=60, help="segundos de descanso entre combinaciones")
    parser.add_argument("--force", action="store_true", help="rehacer aunque el resultado ya exista")
    parser.add_argument("--dry-run", action="store_true", help="solo muestra qué pasos faltan")
    parser.add_argument("--smoke", action="store_true", help="prueba de humo de tiempo y memoria (no escribe índices)")
    parser.add_argument("--smoke-child", choices=local, help=argparse.SUPPRESS)
    parser.add_argument("--chunks", default="s3-1024", help=argparse.SUPPRESS)  # solo --smoke-child
    args = parser.parse_args()

    packages = args.packages or json.loads((EVAL_DIR / "corpus_subset.json").read_text(encoding="utf-8"))["packages"]
    if args.smoke_child:
        return smoke_child(args.smoke_child, args.chunks, packages, SMOKE_TEXTS)
    study = STUDIES[args.study]
    manifest_path, models = study["manifest"], args.models or study["models"]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"}
    if args.smoke:
        return smoke(models, f"{(args.only or study['splitters'])[0]}-{SIZE}", packages, env)

    plan = []  # modelo por fuera, splitter por dentro: el segundo splitter reusa casi todo el caché del primero
    for model in models:
        for splitter in args.only or study["splitters"]:
            chunks = f"{splitter}-{SIZE}"
            name = IndexConfig(chunks=chunks, model=model, packages=packages).collection_name()
            plan.append((model, splitter, chunks, name, outputs(name, args.split)))

    print(f"estudio {args.study}: {len(plan)} combinaciones, {len(packages)} librerías, {args.split}")
    for model, splitter, _, name, paths in plan:
        state = {s: ("hecho" if is_done(s, paths[s], args.force) else "falta") for s in STEPS}
        print(f"  {model} {splitter:<5} {name}: " + ", ".join(f"{s} {v}" for s, v in state.items()))
    if args.dry_run:
        return 0

    if sys.platform == "darwin" and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])  # sin dormir mientras corre este proceso

    manifest = load_manifest(manifest_path)
    manifest.update({
        "study": args.study, "models": {m: {"name": MODELS[m].name, "config": asdict(MODELS[m])} for m in models},
        "split": args.split, "size": SIZE, "n_packages": len(packages),
        "packages_file": "data/eval/corpus_subset.json" if not args.packages else None, "packages": packages,
        "dataset": dataset_info(), "git_dirty": git_dirty(), "environment": environment(), "settings": {
            "warm_batch": args.warm_batch, "pause": args.pause, "cooldown": args.cooldown},
    })
    if args.study == "splitters":  # formato de la etapa 1, sin cambios
        manifest.update({"model": models[0], "model_name": MODELS[models[0]].name,
                         "model_config": asdict(MODELS[models[0]])})
    manifest.setdefault("steps", {})
    manifest.setdefault("started", datetime.now().isoformat(timespec="seconds"))
    (STUDY_DIR / "logs").mkdir(parents=True, exist_ok=True)

    for i, (model, splitter, chunks, name, paths) in enumerate(plan, start=1):
        worked = False
        for step in STEPS:
            if is_done(step, paths[step], args.force):
                continue
            log = STUDY_DIR / "logs" / f"{name}__{step}.log"
            cmd = command(step, chunks, model, args, packages)
            print(f"[{i}/{len(plan)}] {model} {splitter}: {step} ... (registro: {log.relative_to(ROOT)})", flush=True)
            start = time.time()
            with log.open("a", encoding="utf-8") as fh:
                fh.write(f"\n=== {datetime.now().isoformat(timespec='seconds')} {' '.join(cmd[:4])} ...\n")
                fh.flush()
                code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
            seconds = round(time.time() - start, 1)
            manifest["steps"].setdefault(name, {})[step] = {
                "status": "ok" if code == 0 else f"error {code}", "seconds": seconds,
                "peak_memory_mb": peak_memory_mb(log) if step == "index" else None,
                "finished": datetime.now().isoformat(timespec="seconds"), "log": str(log.relative_to(ROOT))}
            save_manifest(manifest, manifest_path)
            if code != 0:
                print(f"  falló ({code}) tras {seconds} s; ver {log}. Al relanzar se retoma desde este paso.")
                return 1
            print(f"  ok en {seconds} s", flush=True)
            worked = True
        if worked and i < len(plan) and args.cooldown:
            print(f"  descanso de {args.cooldown:.0f} s", flush=True)
            time.sleep(args.cooldown)

    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    save_manifest(manifest, manifest_path)
    print(f"listo. Manifiesto: {manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
