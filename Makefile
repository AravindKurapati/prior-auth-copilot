.PHONY: install lint test test-all ingest run demo persistence-test samples all

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

demo:
	pac demo

persistence-test:
	pac persistence-test

samples:
	python scripts/make_samples.py

all:
	pac all
