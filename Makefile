.PHONY: install lint test test-all ingest ingest-rag chunks run demo persistence-test samples all

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

chunks:
	python -c "from pa_copilot.rag import corpus; corpus.write_chunks_jsonl('data/synthetic/clinical_guidance_chunks.jsonl', corpus.load_guidance())"

demo:
	pac demo

# pac persistence-test once PR7 wires the CLI
persistence-test:
	python scripts/run_persistence_test.py

samples:
	python scripts/make_samples.py

all:
	pac all
