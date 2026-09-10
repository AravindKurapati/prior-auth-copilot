.PHONY: install lint test test-all ingest ingest-rag run demo persistence-test samples all

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests

test:
	pytest -q -m "not slow"

test-all:
	pytest -q

ingest:
	pac ingest

ingest-rag:
	python scripts/ingest_rag.py

demo:
	pac demo

persistence-test:
	pac persistence-test

samples:
	python scripts/make_samples.py

all:
	pac all
