# UMAP Atlas — Support Insights Map

Projeto de portfólio para gerar um atlas semântico de conversas de suporte sintéticas (PT/EN) com:
- geração de dados sintéticos
- injeção e sanitização de PII fake
- embeddings determinísticos (hash baseline)
- projeção UMAP (quando disponível) com fallback explícito
- clustering sem vazamento de `true_intent` (HDBSCAN quando disponível + fallback)
- rotulagem automática por cluster
- nearest neighbors pré-computados
- export em `parquet` (quando disponível) ou fallback (`csv`/`jsonl`) documentado no relatório

## Executar

```bash
PYTHONPATH=src python -m umap_atlas.cli validate-config -c configs/base.yaml
PYTHONPATH=src python -m umap_atlas.cli run -c configs/base.yaml
```

## Testes

```bash
PYTHONPATH=src python -m pytest -q
```

## Saídas

- `data/base_run.parquet` (ou `data/base_run.csv` fallback, conforme ambiente)
- `data/base_run_report.md`
