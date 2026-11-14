"""Integración del juicio en analyze_generation.py (el script se carga por ruta: src/scripts no es un paquete)."""

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "src" / "scripts" / "analyze_generation.py"
spec = importlib.util.spec_from_file_location("analyze_generation", SCRIPT)
analyze = importlib.util.module_from_spec(spec)
sys.modules["analyze_generation"] = analyze
spec.loader.exec_module(analyze)


def record(item, score):
    return {"id": item, "targets": [{"retrieval_score": score}]}


def test_load_judge(tmp_path):
    items = tmp_path / "corrida.items.jsonl"
    assert analyze.load_judge(items) == {}  # sin juicio
    (tmp_path / "corrida.judge.jsonl").write_text(
        json.dumps({"id": "x", "a": {"score": 0.8}, "b": {"score": 0.5}}) + "\n"
        + json.dumps({"id": "y", "b": {"score": 1.0}}) + "\n")
    assert analyze.load_judge(items) == {"x": {"a": 0.8, "b": 0.5}, "y": {"b": 1.0}}


def test_mean_passing_only_scored_items_over_threshold():
    records = [record("x", 0.7), record("y", 0.5), record("z", 0.9)]
    scores = {"x": 1.0, "y": 0.0, "z": None}  # z sin puntaje (p. ej. no cambió el script)
    assert analyze.mean_passing(records, scores, 0.4) == (0.5, 2)
    assert analyze.mean_passing(records, scores, 0.6) == (1.0, 1)
    assert analyze.mean_passing(records, scores, 0.95) == (None, 0)
