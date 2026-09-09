.PHONY: install lint test test-all ingest run demo persistence-test all

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

all:
	pac all
