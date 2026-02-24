from umap_atlas.pipeline import FINAL_COLUMNS, run_pipeline


def test_output_schema_matches_contract(tmp_path) -> None:
    out_dir = (tmp_path / "out").as_posix()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        """
run:
  seed: 7
  output_dir: "{out_dir}"
  run_id: test_schema

data:
  n_conversations: 150
  start_date: "2026-01-01"
  days: 8
  multi_intent_ratio: 0.1
  incident_spike_ratio: 0.05

embed:
  model: hash-v1
  dim: 64

project:
  n_neighbors: 10
  min_dist: 0.1
  metric: cosine

cluster:
  min_cluster_size: 10
  min_samples: 3

neighbors:
  k: 5

export:
  format: jsonl
        """.format(out_dir=out_dir).strip()
    )
    rows, data_path, _ = run_pipeline(cfg)
    assert list(rows[0].keys()) == FINAL_COLUMNS
    assert len(rows) == 150
    assert data_path.suffix == ".jsonl"
    assert rows[0]["projection_method"] in {"umap", "fallback_first2"}
    assert rows[0]["cluster_method"] in {"hdbscan", "fallback_greedy"}
