from pathlib import Path

from umap_atlas.pipeline import run_pipeline
from umap_atlas.sanitize import PHONE_RE, DOC_RE


def test_e2e_pipeline_generates_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    cfg = tmp_path / "cfg_e2e.yaml"
    cfg.write_text(
        f"""
run:
  seed: 42
  output_dir: "{out_dir.as_posix()}"
  run_id: e2e_run

data:
  n_conversations: 220
  start_date: "2026-01-01"
  days: 14
  multi_intent_ratio: 0.2
  incident_spike_ratio: 0.09

embed:
  model: hash-v1
  dim: 96

project:
  n_neighbors: 12
  min_dist: 0.05
  metric: cosine

cluster:
  min_cluster_size: 12
  min_samples: 4

neighbors:
  k: 7

export:
  format: jsonl
        """.strip()
    )

    rows, data_path, report_path = run_pipeline(cfg)

    assert len(rows) == 220
    assert data_path.exists()
    assert data_path.suffix == ".jsonl"
    assert report_path.exists()
    assert all(len(r["neighbors_ids"]) <= 7 for r in rows)
    assert all("@" not in r["sanitized_transcript"] for r in rows)
    assert all(PHONE_RE.search(r["sanitized_transcript"]) is None for r in rows)
    assert all(DOC_RE.search(r["sanitized_transcript"]) is None for r in rows)
    assert all("@" not in r["embedding_text"] for r in rows)
    assert all(PHONE_RE.search(r["embedding_text"]) is None for r in rows)
    assert all(DOC_RE.search(r["embedding_text"]) is None for r in rows)
    assert {r["projection_method"] for r in rows}
    assert {r["cluster_method"] for r in rows}
