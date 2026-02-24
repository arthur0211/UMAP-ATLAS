from __future__ import annotations


def _fallback_project(embeddings: list[list[float]]) -> list[tuple[float, float]]:
    points = []
    for emb in embeddings:
        x = emb[0] if emb else 0.0
        y = emb[1] if len(emb) > 1 else 0.0
        points.append((float(x), float(y)))
    return points


def project_2d(
    embeddings: list[list[float]],
    n_neighbors: int,
    min_dist: float,
    metric: str,
    random_state: int,
) -> tuple[list[tuple[float, float]], str]:
    try:
        import numpy as np  # type: ignore
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        arr = np.array(embeddings)
        projected = reducer.fit_transform(arr)
        points = [(float(x), float(y)) for x, y in projected]
        return points, "umap"
    except ModuleNotFoundError:
        return _fallback_project(embeddings), "fallback_first2"
