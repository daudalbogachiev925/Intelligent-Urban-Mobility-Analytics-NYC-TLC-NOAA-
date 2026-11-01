.PHONY: setup download features train serve test lint clean

setup:
	python -m venv .venv
	.venv/bin/pip install -e ".[dev]"

download:
	.venv/bin/python scripts/download_data.py

features:
	.venv/bin/python scripts/build_features.py

train:
	.venv/bin/python scripts/train.py

serve:
	.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000

test:
	.venv/bin/pytest --cov=src

lint:
	.venv/bin/ruff check src tests

clean:
	rm -rf artifacts data/processed/*
