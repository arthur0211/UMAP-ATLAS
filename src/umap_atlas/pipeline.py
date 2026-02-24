from __future__ import annotations

from pathlib import Path

from .cluster import cluster_rows
from .config import Config, load_config
from .embed import create_embeddings
from .export import export_outputs
from .generate import generate_dataset
from .label import label_rows
from .neighbors import compute_neighbors
from .project import project_2d
from .represent import build_embedding_text
from .sanitize import inject_fake_pii, sanitize_rows

FINAL_COLUMNS = [
    "conversation_id",
    "created_at",
    "channel",
    "language_mix",
    "customer_segment",
    "journey_stage",
    "product_area",
    "resolution_status",
    "incident_flag",
    "true_intent",
    "synthetic_transcript_raw",
    "sanitized_transcript",
    "embedding_text",
    "umap_x",
    "umap_y",
    "cluster_id",
    "cluster_prob",
    "cluster_label",
    "cluster_keywords",
    "neighbors_ids",
    "neighbors_scores",
    "projection_method",
    "cluster_method",
    "run_id",
    "config_hash",
    "seed",
]


def _execute(cfg: Config) -> tuple[list[dict], Path, Path]:
    rows = generate_dataset(
        n=cfg.data.n_conversations,
        start_date=cfg.data.start_date,
        days=cfg.data.days,
        multi_intent_ratio=cfg.data.multi_intent_ratio,
        incident_spike_ratio=cfg.data.incident_spike_ratio,
        seed=cfg.run.seed,
    )
    rows = inject_fake_pii(rows, seed=cfg.run.seed)
    rows = sanitize_rows(rows)
    rows = build_embedding_text(rows)

    emb = create_embeddings(rows, dim=cfg.embed.dim)
    points_2d, projection_method = project_2d(
        emb,
        n_neighbors=cfg.project.n_neighbors,
        min_dist=cfg.project.min_dist,
        metric=cfg.project.metric,
        random_state=cfg.run.seed,
    )
    labels, probs, cluster_method = cluster_rows(
        emb,
        min_cluster_size=cfg.cluster.min_cluster_size,
        min_samples=cfg.cluster.min_samples,
    )

    updated: list[dict] = []
    for row, (x, y), label, prob in zip(rows, points_2d, labels, probs):
        nr = row.copy()
        nr["umap_x"] = x
        nr["umap_y"] = y
        nr["cluster_id"] = int(label)
        nr["cluster_prob"] = float(prob)
        updated.append(nr)

    rows = label_rows(updated)

    ids = [r["conversation_id"] for r in rows]
    neigh_ids, neigh_scores = compute_neighbors(emb, ids=ids, k=cfg.neighbors.k)

    final_rows: list[dict] = []
    for row, nids, nscores in zip(rows, neigh_ids, neigh_scores):
        nr = row.copy()
        nr["neighbors_ids"] = nids
        nr["neighbors_scores"] = nscores
        nr["projection_method"] = projection_method
        nr["cluster_method"] = cluster_method
        nr["run_id"] = cfg.run.run_id
        nr["config_hash"] = cfg.config_hash
        nr["seed"] = cfg.run.seed
        ordered = {c: nr[c] for c in FINAL_COLUMNS}
        final_rows.append(ordered)

    data_path, report_path = export_outputs(
        rows=final_rows,
        output_dir=cfg.run.output_dir,
        run_id=cfg.run.run_id,
        config_hash=cfg.config_hash,
        export_format=cfg.export.format,
        projection_method=projection_method,
        cluster_method=cluster_method,
    )
    return final_rows, data_path, report_path


def run_pipeline(config_path: str | Path) -> tuple[list[dict], Path, Path]:
    cfg = load_config(config_path)
    return _execute(cfg)
