from __future__ import annotations

from collections import Counter
import re

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]{2,}")
STOPWORDS = {"the", "and", "for", "with", "that", "customer", "problema", "objetivo", "area", "canal", "idioma"}


def label_rows(rows: list[dict]) -> list[dict]:
    by_cluster: dict[int, list[str]] = {}
    for r in rows:
        cid = int(r["cluster_id"])
        by_cluster.setdefault(cid, []).append(r["embedding_text"])

    cluster_label: dict[int, str] = {}
    cluster_keywords: dict[int, list[str]] = {}
    for cid, texts in by_cluster.items():
        if cid == -1:
            cluster_label[cid] = "outliers"
            cluster_keywords[cid] = []
            continue
        c: Counter[str] = Counter()
        for t in texts:
            for tok in TOKEN_RE.findall(t.lower()):
                if tok not in STOPWORDS:
                    c[tok] += 1
        kws = [w for w, _ in c.most_common(5)]
        cluster_keywords[cid] = kws
        cluster_label[cid] = "/".join(kws[:2]) if kws else f"cluster_{cid}"

    out: list[dict] = []
    for r in rows:
        nr = r.copy()
        cid = int(nr["cluster_id"])
        nr["cluster_label"] = cluster_label[cid]
        nr["cluster_keywords"] = cluster_keywords[cid]
        out.append(nr)
    return out
