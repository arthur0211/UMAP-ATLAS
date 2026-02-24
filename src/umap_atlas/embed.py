from __future__ import annotations

import hashlib
import math


def _hash_embedding(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    for tok in text.lower().split():
        h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def create_embeddings(rows: list[dict], dim: int) -> list[list[float]]:
    return [_hash_embedding(r["embedding_text"], dim) for r in rows]
