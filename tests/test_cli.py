import os
import subprocess
import sys
from pathlib import Path


PY = [sys.executable, "-m", "umap_atlas.cli"]


def _config_text(out_dir: Path, run_id: str = "cli") -> str:
    return f"""
run:
  seed: 42
  output_dir: "{out_dir.as_posix()}"
  run_id: {run_id}

data:
  n_conversations: 20
  start_date: "2026-01-01"
  days: 3
  multi_intent_ratio: 0.1
  incident_spike_ratio: 0.1

embed:
  model: hash-v1
  dim: 16

project:
  n_neighbors: 5
  min_dist: 0.1
  metric: cosine

cluster:
  min_cluster_size: 2
  min_samples: 1

neighbors:
  k: 3

export:
  format: jsonl
""".strip()


def test_cli_run_happy_path(tmp_path: Path) -> None:
    cfg = tmp_path / "cli.yaml"
    cfg.write_text(_config_text(tmp_path / "out"))
    cmd = [*PY, "run", "-c", str(cfg)]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"})
    assert proc.returncode == 0
    assert "Rows:" in proc.stdout


def test_cli_config_missing(tmp_path: Path) -> None:
    cmd = [*PY, "validate-config", "-c", str(tmp_path / "does_not_exist.yaml")]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"})
    assert proc.returncode != 0


def test_cli_invalid_command() -> None:
    cmd = [*PY, "invalid-cmd"]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": "src"})
    assert proc.returncode != 0
