# Arquitetura

Pipeline offline:
1. generate
2. inject_fake_pii
3. sanitize
4. represent
5. embed
6. project (UMAP)
7. cluster (HDBSCAN)
8. label
9. neighbors
10. export
