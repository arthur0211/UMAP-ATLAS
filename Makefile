.PHONY: install test run

install:
	pip install -e .[dev]

test:
	pytest -q

run:
	umap-atlas run -c configs/base.yaml
