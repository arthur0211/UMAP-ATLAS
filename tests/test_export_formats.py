from pathlib import Path

import pytest

from umap_atlas.export import export_outputs


BASE_ROWS = [
    {
        "conversation_id": "c1",
        "created_at": "2026-01-01T00:00:00",
        "channel": "app_chat",
        "language_mix": "pt",
        "customer_segment": "new",
        "journey_stage": "onboarding",
        "product_area": "kyc",
        "resolution_status": "resolved",
        "incident_flag": False,
        "true_intent": "kyc_doc_rejected",
        "synthetic_transcript_raw": "raw",
        "sanitized_transcript": "san",
        "embedding_text": "emb",
        "umap_x": 0.1,
        "umap_y": 0.2,
        "cluster_id": 0,
        "cluster_prob": 0.9,
        "cluster_label": "kyc",
        "cluster_keywords": ["kyc"],
        "neighbors_ids": ["c2"],
        "neighbors_scores": [0.99],
        "projection_method": "fallback_first2",
        "cluster_method": "fallback_greedy",
        "run_id": "r",
        "config_hash": "h",
        "seed": 42,
    }
]


def test_export_jsonl(tmp_path: Path) -> None:
    data_path, report_path = export_outputs(
        rows=BASE_ROWS,
        output_dir=tmp_path,
        run_id="x",
        config_hash="abc",
        export_format="jsonl",
        projection_method="fallback_first2",
        cluster_method="fallback_greedy",
    )
    assert data_path.suffix == ".jsonl"
    assert data_path.exists()
    assert report_path.exists()


def test_export_parquet_if_available(tmp_path: Path) -> None:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ModuleNotFoundError:
        pytest.skip("pyarrow not installed in this environment")

    data_path, _ = export_outputs(
        rows=BASE_ROWS,
        output_dir=tmp_path,
        run_id="y",
        config_hash="abc",
        export_format="parquet",
        projection_method="umap",
        cluster_method="hdbscan",
    )
    assert data_path.suffix == ".parquet"
    table = pq.read_table(data_path)
    assert table.num_rows == 1
