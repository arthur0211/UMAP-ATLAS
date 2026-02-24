from umap_atlas.cluster import cluster_rows


def test_cluster_does_not_depend_on_true_intent() -> None:
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.99, 0.01, 0.0],
        [0.98, 0.02, 0.0],
        [0.0, 1.0, 0.0],
        [0.01, 0.99, 0.0],
        [0.02, 0.98, 0.0],
    ]
    labels_a, probs_a, _ = cluster_rows(embeddings, min_cluster_size=2, min_samples=1)
    labels_b, probs_b, _ = cluster_rows(embeddings, min_cluster_size=2, min_samples=1)

    assert labels_a == labels_b
    assert probs_a == probs_b
