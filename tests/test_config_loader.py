import pytest

from umap_atlas.config import ConfigError, load_config


def test_load_config_valid(tmp_path) -> None:
    cfg = tmp_path / "ok.yaml"
    cfg.write_text(
        """
run:
  seed: 42
  output_dir: data
  run_id: ok

data:
  n_conversations: 10
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
    )
    conf = load_config(cfg)
    assert conf.export.format == "jsonl"


def test_load_config_missing_key(tmp_path) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(
        """
run:
  seed: 42
  output_dir: data
  run_id: bad

data:
  n_conversations: 10
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
        """.strip()
    )
    with pytest.raises(ConfigError):
        load_config(cfg)
