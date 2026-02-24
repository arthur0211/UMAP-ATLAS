from __future__ import annotations


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _fallback_cluster(embeddings: list[list[float]], min_cluster_size: int) -> tuple[list[int], list[float]]:
    # Greedy centroid clustering (deterministic fallback, label-leak free).
    threshold = 0.72
    clusters: list[dict] = []
    assignments: list[int] = []
    for emb in embeddings:
        best_idx = -1
        best_score = -1.0
        for idx, cluster in enumerate(clusters):
            score = _cosine(emb, cluster["centroid"])
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx >= 0 and best_score >= threshold:
            cluster = clusters[best_idx]
            cluster["members"].append(len(assignments))
            size = len(cluster["members"])
            cluster["centroid"] = [
                (c * (size - 1) + e) / size for c, e in zip(cluster["centroid"], emb)
            ]
            assignments.append(best_idx)
        else:
            clusters.append({"centroid": emb[:], "members": [len(assignments)]})
            assignments.append(len(clusters) - 1)

    sizes = {idx: len(c["members"]) for idx, c in enumerate(clusters)}
    kept = [idx for idx, size in sizes.items() if size >= min_cluster_size]
    remap = {old: new for new, old in enumerate(sorted(kept))}

    labels: list[int] = []
    probs: list[float] = []
    for cid in assignments:
        if cid in remap:
            labels.append(remap[cid])
            probs.append(min(1.0, sizes[cid] / max(1, len(embeddings))))
        else:
            labels.append(-1)
            probs.append(0.0)
    return labels, probs


def cluster_rows(
    embeddings: list[list[float]],
    min_cluster_size: int,
    min_samples: int,
) -> tuple[list[int], list[float], str]:
    try:
        import numpy as np  # type: ignore
        import hdbscan  # type: ignore

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
        labels = clusterer.fit_predict(np.array(embeddings))
        probs = getattr(clusterer, "probabilities_", np.zeros(len(embeddings)))
        return [int(x) for x in labels], [float(x) for x in probs], "hdbscan"
    except ModuleNotFoundError:
        labels, probs = _fallback_cluster(embeddings, min_cluster_size=min_cluster_size)
        return labels, probs, "fallback_greedy"
