"""Build the clinical-guidance Chroma index and refresh the committed evidence.

Run from the repo root:

    python scripts/ingest_rag.py

`sys.path[0]` is `scripts/` when invoked this way, so put the repo root on the
path first (Ruling R3). This is the one in-repo producer of every committed RAG
artifact (I6):

- `data/synthetic/clinical_guidance_chunks.jsonl` — deterministic chunk dump
  from `corpus.load_guidance()` (model-free).
- `traces/rag_index_summary.json` — from `dataclasses.asdict(summary)`,
  deliberately no timestamp field, so it is byte-stable across runs as long as
  the corpus and model are unchanged.

`make chunks` regenerates just the JSONL without touching the model.
"""

import dataclasses
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from pa_copilot.rag import corpus  # noqa: E402
from pa_copilot.rag.index import build_index  # noqa: E402


def main() -> None:
    chunks_path = _REPO_ROOT / "data" / "synthetic" / "clinical_guidance_chunks.jsonl"
    corpus.write_chunks_jsonl(chunks_path, corpus.load_guidance())
    print(f"wrote {chunks_path.relative_to(_REPO_ROOT).as_posix()}")

    summary = build_index()
    payload = dataclasses.asdict(summary)

    out_path = _REPO_ROOT / "traces" / "rag_index_summary.json"
    out_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    print(f"wrote {out_path.relative_to(_REPO_ROOT).as_posix()}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
