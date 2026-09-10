"""Agentic RAG package for the prior-auth copilot.

Corpus loading / clause chunking (`corpus`), local embeddings behind a protocol
(`embedder`), and the Chroma-backed index (`index`, Task 3) live here.

`chromadb` reads `ANONYMIZED_TELEMETRY` at import time and, when unset, spins up a
background telemetry thread that emits noisy warnings under the suite's warning
trap. Disable it here, before any submodule pulls chromadb in.
"""

import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
