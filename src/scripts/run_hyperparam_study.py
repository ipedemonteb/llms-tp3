"""Etapa 3 del plan de experimentos (docs/agents/plan_experimentos_indice.md): hiperparámetros de S3 + E4, un factor a
la vez alrededor de la base (grilla en src/tprag/eval/hyperparams.py):

    tamaño de chunk 256 / 512 / 1024 (base) / 2048    mínimo de tokens 0 (base) / 64 / 128
    tope por librería 75 / 150 (base) / 300 / sin tope    representación R1 (base) / raw (sin prefijo)

Por cada variante corre cuatro pasos, cada uno en un proceso aparte (la memoria se libera entre uno y otro):
`split` (chunks), `index` (Chroma + vectores locales), `eval` (retrieval en dev) y `stats` (forma de los chunks).
No usa API. El tamaño 2048 usa el modelo e4l (E4 con ventana de 3072 tokens, así no se trunca).

Uso (hace falta el grupo de dependencias `local`):
    uv run --group local python src/scripts/run_hyperparam_study.py --dry-run           # qué falta
    uv run --group local python src/scripts/run_hyperparam_study.py                     # toda la grilla, en orden
    uv run --group local python src/scripts/run_hyperparam_study.py --only raw cap300   # algunas variantes
    uv run --group local python src/scripts/run_hyperparam_study.py --force             # rehace aunque esté hecho
    uv run python src/scripts/analyze_hyperparams.py                                    # tabla, JSON y gráficos
    uv run --group local python src/scripts/run_hyperparam_study.py --generation --dry-run   # finalistas con LLM
    uv run --group local python src/scripts/run_hyperparam_study.py --generation             # USA LA API (gpt-4o-mini)

Se retoma solo: salta los pasos que ya tienen su resultado (la evaluación solo si es del mismo dataset). La base ya
está hecha de la etapa 1 (misma colección), así que no se rehace. Cuidado con la memoria en `size2048` (ventana de 3072
tokens con la GPU compartida): corre al final; si la Mac swapea, cortar y relanzar con `--warm-batch 100`.

`--generation` toma los finalistas de results/study/hyperparams_analysis.json (los deja analyze_hyperparams.py) y corre
eval_generation.py sobre cada uno: es lo único que gasta API (una llamada por script, con caché de respuestas).

Salidas: results/index/, results/eval/, results/study/ (estadísticas, registros en logs/ y el manifiesto
hyperparams_manifest.json con tiempos y memoria de cada paso).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

from run_splitter_study import (SCRIPTS, STUDY_DIR, environment, git_dirty, is_done, load_manifest, outputs,
                                peak_memory_mb, save_manifest, timed)
from tprag.eval.hyperparams import BY_ID, VARIANTS, Variant
from tprag.eval.retrieval import dataset_info
from tprag.eval.verify import EVAL_DIR
from tprag.index.embeddings import MODELS
from tprag.split.store import PROCESSED_DIR, ROOT

CONSOLE_MODE = "outputs_as_comments"
STEPS = ("split", "index", "eval", "stats")
MANIFEST = STUDY_DIR / "hyperparams_manifest.json"
ANALYSIS = STUDY_DIR / "hyperparams_analysis.json"


def paths(v: Variant, packages: list[str], split: str) -> dict:
    name = v.index_config(packages).collection_name()
    return {"name": name, "split": PROCESSED_DIR / CONSOLE_MODE / v.chunks / "chunks.jsonl", **outputs(name, split)}


def command(step: str, v: Variant, packages: list[str], args) -> list[str]:
    if step == "split":
        return [sys.executable, str(SCRIPTS / "split_docs.py"), "--splitters", "s3", "--max-tokens", str(v.size),
                "--min-tokens", str(v.min_tokens)]
    common = ["--chunks", v.chunks, "--model", v.model, "--representation", v.representation,
              "--cap", str(v.cap or 0), "--packages", *packages]
    if step == "index":
        return [*timed(), sys.executable, str(SCRIPTS / "build_index.py"), *common, "--warm-batch",
                str(args.warm_batch), "--pause", str(args.pause)]
    if step == "eval":
        return [sys.executable, str(SCRIPTS / "eval_retrieval.py"), *common, "--split", args.split]
    return [sys.executable, str(SCRIPTS / "chunk_stats.py"), *common]


def run_step(cmd: list[str], log, env: dict) -> tuple[int, float]:
    start = time.time()
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== {datetime.now().isoformat(timespec='seconds')} {' '.join(cmd[:4])} ...\n")
        fh.flush()
        code = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    return code, round(time.time() - start, 1)


def generation(packages: list[str], args, env: dict) -> int:
    """eval_generation.py sobre los finalistas que dejó el análisis."""
    try:
        finalists = json.loads(ANALYSIS.read_text(encoding="utf-8"))["finalists"]
    except (OSError, KeyError, ValueError):
        print(f"no hay finalistas: correr antes analyze_hyperparams.py (lee {ANALYSIS.relative_to(ROOT)})")
        return 1
    print(f"generación sobre los finalistas: {', '.join(finalists)}" + (" (dry-run)" if args.dry_run else ""))
    for vid in finalists:
        v = BY_ID[vid]
        cmd = [sys.executable, str(SCRIPTS / "eval_generation.py"), "--chunks", v.chunks, "--model", v.model,
               "--representation", v.representation, "--cap", str(v.cap or 0), "--split", args.split,
               "--max-usd", str(args.max_usd), "--packages", *packages] + (["--dry-run"] if args.dry_run else [])
        print(f"  {vid}: {v.index_config(packages).collection_name()}", flush=True)
        code = subprocess.run(cmd, cwd=ROOT, env=env).returncode
        if code != 0:
            print(f"  falló ({code})")
            return code
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", choices=sorted(BY_ID), help="solo estas variantes (en el orden dado)")
    parser.add_argument("--packages", nargs="*", help="default: las librerías de data/eval/corpus_subset.json")
    parser.add_argument("--split", choices=("dev", "test"), default="dev", help="test solo para el reporte final")
    parser.add_argument("--warm-batch", type=int, default=200, help="textos por lote de vectores (se guarda cada lote)")
    parser.add_argument("--pause", type=float, default=3, help="segundos de pausa entre lotes de vectores")
    parser.add_argument("--cooldown", type=float, default=60, help="segundos de descanso entre variantes")
    parser.add_argument("--force", action="store_true", help="rehacer aunque el resultado ya exista")
    parser.add_argument("--dry-run", action="store_true", help="solo muestra qué pasos faltan")
    parser.add_argument("--generation", action="store_true",
                        help="en vez de la grilla, corre eval_generation.py sobre los finalistas (USA LA API)")
    parser.add_argument("--max-usd", type=float, default=0.10, help="tope por finalista con --generation")
    args = parser.parse_args()

    packages = args.packages or json.loads((EVAL_DIR / "corpus_subset.json").read_text(encoding="utf-8"))["packages"]
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"}
    if args.generation:
        return generation(packages, args, env)

    variants = [BY_ID[i] for i in args.only] if args.only else list(VARIANTS)
    plan = [(v, paths(v, packages, args.split)) for v in variants]
    print(f"estudio de hiperparámetros: {len(plan)} variantes, {len(packages)} librerías, {args.split}")
    for v, p in plan:
        state = {s: ("hecho" if is_done(s, p[s], args.force) else "falta") for s in STEPS}
        print(f"  {v.id:<9} {p['name']}: " + ", ".join(f"{s} {x}" for s, x in state.items()))
    if args.dry_run:
        return 0

    if sys.platform == "darwin" and shutil.which("caffeinate"):
        subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])  # sin dormir mientras corre este proceso

    manifest = load_manifest(MANIFEST)
    manifest.update({
        "study": "hyperparams", "split": args.split, "n_packages": len(packages), "packages": packages,
        "variants": {v.id: {"factor": v.factor, "label": v.label, "collection": p["name"], "chunks": v.chunks,
                            "model": v.model, "cap": v.cap, "representation": v.representation}
                     for v, p in plan},
        "models": {m: MODELS[m].name for m in {v.model for v in variants}}, "dataset": dataset_info(),
        "git_dirty": git_dirty(), "environment": environment(),
        "settings": {"warm_batch": args.warm_batch, "pause": args.pause, "cooldown": args.cooldown}})
    manifest.setdefault("steps", {})
    manifest.setdefault("started", datetime.now().isoformat(timespec="seconds"))
    (STUDY_DIR / "logs").mkdir(parents=True, exist_ok=True)

    for i, (v, p) in enumerate(plan, start=1):
        worked = False
        for step in STEPS:
            if is_done(step, p[step], args.force):
                continue
            log = STUDY_DIR / "logs" / f"{p['name']}__{step}.log"
            print(f"[{i}/{len(plan)}] {v.id}: {step} ... (registro: {log.relative_to(ROOT)})", flush=True)
            code, seconds = run_step(command(step, v, packages, args), log, env)
            peak = peak_memory_mb(log) if step == "index" else None
            attempts = manifest["steps"].get(p["name"], {}).get(step, {}).get("attempts", [])
            attempts.append({"status": "ok" if code == 0 else f"error {code}", "seconds": seconds,
                             "peak_memory_mb": peak, "finished": datetime.now().isoformat(timespec="seconds")})
            manifest["steps"].setdefault(p["name"], {})[step] = {
                "status": attempts[-1]["status"], "seconds": seconds,
                "seconds_total": round(sum(a["seconds"] for a in attempts), 1),
                "peak_memory_mb": max((a["peak_memory_mb"] or 0 for a in attempts), default=None) or None,
                "finished": attempts[-1]["finished"], "log": str(log.relative_to(ROOT)), "attempts": attempts}
            save_manifest(manifest, MANIFEST)
            if code != 0:
                print(f"  falló ({code}) tras {seconds} s; ver {log}. Al relanzar se retoma desde este paso.")
                return 1
            print(f"  ok en {seconds} s", flush=True)
            worked = True
        if worked and i < len(plan) and args.cooldown:
            print(f"  descanso de {args.cooldown:.0f} s", flush=True)
            time.sleep(args.cooldown)

    manifest["finished"] = datetime.now().isoformat(timespec="seconds")
    save_manifest(manifest, MANIFEST)
    print(f"listo. Manifiesto: {MANIFEST.relative_to(ROOT)}\nsiguiente: uv run python src/scripts/analyze_hyperparams.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
