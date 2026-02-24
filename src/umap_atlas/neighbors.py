from __future__ import annotations


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def compute_neighbors(embeddings: list[list[float]], ids: list[str], k: int) -> tuple[list[list[str]], list[list[float]]]:
    all_ids: list[list[str]] = []
    all_scores: list[list[float]] = []
    for i, emb in enumerate(embeddings):
        pairs: list[tuple[float, str]] = []
        for j, other in enumerate(embeddings):
            if i == j:
                continue
            pairs.append((_cosine(emb, other), ids[j]))
        pairs.sort(reverse=True, key=lambda x: x[0])
        top = pairs[:k]
        all_ids.append([p[1] for p in top])
        all_scores.append([float(p[0]) for p in top])
    return all_ids, all_scores
