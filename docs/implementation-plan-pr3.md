# PR3 — Agentic RAG Tool: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** An agentic-RAG tool (`search_clinical_guidance`) over an in-process Chroma index
of the synthetic clinical-guidance corpus, plus the decision helper the `medical_necessity`
worker will use to decide *when* to call it. Closes AC-11 at the tool level (the full
in-graph "agent decides" evidence is PR5).

**Architecture:** `rag/corpus.py` loads + clause-chunks the committed
`data/synthetic/clinical_guidance/*.md` (pure, deterministic, no model). `rag/embedder.py`
is a thin local-model wrapper behind an `Embedder` protocol so tests inject a fake.
`rag/index.py` builds a persistent Chroma collection (`.pa_chroma/`) and does filtered
similarity search. `rag/tool.py` wraps search as a LangChain `@tool` returning
`CriteriaCitation` objects, with a one-shot corrective query rewrite when retrieval is weak,
plus `should_search_guidance(...)` — the routing predicate PR5's worker calls.

**Tech Stack:** `chromadb` 0.6.3 (raw client, not `langchain-chroma`), `sentence-transformers`
5.2.3 (`BAAI/bge-small-en-v1.5`, CPU), `langchain-core` tools, Pydantic v2, `pytest`.

**Spec:** `docs/design.md` §7; `specs/acceptance-criteria.md` AC-11.

## Global Constraints

- Python `>=3.11`. Package `pa_copilot` under `src/`. No Docker, no DB service — Chroma is
  an in-process persistent client writing a local dir (`.pa_chroma*/`, gitignored).
- All data synthetic. The RAG corpus is the **committed**
  `data/synthetic/clinical_guidance/*.md` (6 files, one per policy) — `rag/corpus.py` is a
  loader/chunker, **not** a generator.
- Google Gemini is the only LLM provider. RAG uses **local** embeddings only — no API calls
  in the retrieval path.
- Test output must be pristine (`filterwarnings = ["error"]`). **chromadb 0.6.3 prints a
  telemetry error to stderr on client creation** — it must be silenced (see Task 3).
- Real-model tests (bge-small ~130 MB download) are `@pytest.mark.slow`. Fast tests inject a
  deterministic `FakeEmbedder`. CI runs only `-m "not slow"`.
- Committed evidence is **byte-stable** across reruns: `data/synthetic/clinical_guidance_chunks.jsonl`
  (deterministic from corpus.py), `traces/rag_index_summary.json` (no timestamp),
  `traces/agentic_rag_decision.md`.
- Every AC test/artifact carries `AC-11`.
- Work on branch `feat/agentic-rag`; the controller does the `--no-ff` merge after a
  whole-branch review — do NOT merge in a task.
- TDD: failing test first, minimal impl, passing test, commit.

## Interfaces from PR1/PR2 (on `main`)

- `pa_copilot.config.load_settings(env_file=None) -> Settings` / `get_settings()`
  (`@lru_cache`, autouse-cleared in tests). `Settings` has `.synthetic_dir`, `.chroma_dir`
  (`./.pa_chroma`), `.traces_dir`, `.repo_root`, `.model_agent`. **This PR adds fields** —
  see Task 3.
- `pa_copilot.schemas.CriteriaCitation(source: Literal["mcp_resource","rag_corpus"],
  clause_id: str, quote: str, relevance: str)`.
- `data/synthetic/generators.py::SERVICES` — `list[{service_code, name, policy_id}]`, 6
  entries. `policy_id` values: `PA-MRI-LUMBAR`(72148), `PA-KNEE-SCOPE`(29881),
  `PA-AFLIBERCEPT`(J0178), `PA-PSG`(95810), `PA-EGD`(43239), `PA-TFESI`(64483).
- `data/synthetic/clinical_guidance/pa-{mri-lumbar,knee-scope,aflibercept,psg,egd,tfesi}.md`
  — each headed `# <Title>` with `## Indications`, `## Step therapy`, `## Exclusions`
  sections (plus a pre-heading paragraph, chunked as section `"Overview"`).
- `pa_copilot.mcp_server.data_access.criteria_check(...) -> {"status": "not_found"|
  "excluded"|"indeterminate", ...}` — the `status` PR5's worker branches on.
- The MCP→worker status mapping table is in `docs/design.md` §4.1.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/pa_copilot/rag/__init__.py` | package marker; sets `ANONYMIZED_TELEMETRY` env before anything imports chromadb |
| `src/pa_copilot/rag/corpus.py` | load + clause-chunk `clinical_guidance/*.md` → chunk dicts; write `clinical_guidance_chunks.jsonl` |
| `src/pa_copilot/rag/embedder.py` | `Embedder` protocol + `BgeEmbedder` (sentence-transformers, CPU, lazy) |
| `src/pa_copilot/rag/index.py` | `build_index()`, `get_collection()`, `search()` over Chroma; `RagIndexSummary` |
| `src/pa_copilot/rag/tool.py` | `search_clinical_guidance` LangChain `@tool` + `should_search_guidance()` predicate |
| `config/models.yaml` | + `embedding_model: BAAI/bge-small-en-v1.5` |
| `config/rag.yaml` | `top_k`, `min_score`, `collection`, `query_prefix`, `rewrite_min_score` |
| `src/pa_copilot/config.py` | MODIFY — add RAG fields to `Settings` + `load_settings` |
| `scripts/ingest_rag.py` | standalone `build_index()` runner (PR7's `pac ingest` will call the same fn) |
| `data/synthetic/clinical_guidance_chunks.jsonl` | committed deterministic chunk dump |
| `traces/rag_index_summary.json` | committed evidence: doc/chunk counts, model, corpus sha |
| `traces/agentic_rag_decision.md` | committed evidence: indeterminate→RAG-called vs excluded→not-called |
| `tests/test_rag_corpus.py`, `tests/test_rag_index.py`, `tests/test_ac11_agentic_rag.py` | |

---

## Task 1: `rag/corpus.py` — loader + clause chunker

**Files:** Create `src/pa_copilot/rag/__init__.py`, `src/pa_copilot/rag/corpus.py`,
`tests/test_rag_corpus.py`. Create (generated, committed)
`data/synthetic/clinical_guidance_chunks.jsonl`.

**Interfaces:**
- `rag/__init__.py`: docstring + `import os; os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")`.
- `rag/corpus.py` produces:
  - `POLICY_BY_FILE: dict[str, tuple[str, str]]` — `{"pa-psg": ("PA-PSG", "95810"), ...}`,
    derived from `data.synthetic.generators.SERVICES` (map policy_id → a kebab filename
    stem: `policy_id.lower().replace("pa-", "pa-")`… actually the stems are fixed:
    `pa-mri-lumbar, pa-knee-scope, pa-aflibercept, pa-psg, pa-egd, pa-tfesi`. Hardcode the
    map and assert in a test that it covers every `SERVICES` policy_id.)
  - `Chunk` — frozen dataclass: `chunk_id: str`, `policy_id: str`, `service_code: str`,
    `doc_title: str`, `section: str` (e.g. `"Indications"`, `"Overview"`), `clause_index: int`,
    `text: str`.
  - `load_guidance(guidance_dir: str | Path | None = None) -> list[Chunk]` — for each
    `*.md`: parse `# Title`; split into sections by `## Heading`; the pre-heading paragraph
    is section `"Overview"`; within each section, split into clauses on blank lines AND on
    `"- "` list-item boundaries (each bullet = one clause); skip empty. `chunk_id` =
    `f"{policy_id}:{section_slug}:{clause_index}"`. Deterministic ordering (files sorted,
    sections in document order).
  - `chunks_to_rows(chunks) -> list[dict]` — `asdict` per chunk.
  - `write_chunks_jsonl(path: str | Path, chunks) -> None` — one JSON object per line,
    `ensure_ascii=False`, `encoding="utf-8"`, `newline="\n"`, trailing `\n`.

- [ ] **Step 1: Failing test** — `tests/test_rag_corpus.py`:

```python
from data.synthetic import generators as g
from pa_copilot.rag import corpus


def test_every_policy_has_a_guidance_file():
    mapped = {pid for pid, _ in corpus.POLICY_BY_FILE.values()}
    assert mapped == {s["policy_id"] for s in g.SERVICES}


def test_load_is_deterministic_and_tagged():
    a = corpus.load_guidance()
    b = corpus.load_guidance()
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len(a) >= 24  # 6 docs, >= 4 clauses each
    psg = [c for c in a if c.policy_id == "PA-PSG"]
    assert psg and all(c.service_code == "95810" for c in psg)
    assert {c.section for c in psg} >= {"Indications", "Step therapy", "Exclusions"}


def test_chunk_ids_unique():
    ids = [c.chunk_id for c in corpus.load_guidance()]
    assert len(ids) == len(set(ids))
```

- [ ] **Step 2: Run → fails** (`ModuleNotFoundError: pa_copilot.rag`)
- [ ] **Step 3: Implement** `__init__.py` + `corpus.py` per Interfaces.
- [ ] **Step 4: Run → passes**
- [ ] **Step 5: Generate the committed chunk dump**

`scripts/ingest_rag.py` is the in-repo producer of
`data/synthetic/clinical_guidance_chunks.jsonl` (and `traces/rag_index_summary.json`);
`make chunks` regenerates just the JSONL without touching the model. Run either twice;
`git status` shows the file identical after the second run.

- [ ] **Step 6: Commit**

```bash
git add src/pa_copilot/rag/__init__.py src/pa_copilot/rag/corpus.py tests/test_rag_corpus.py data/synthetic/clinical_guidance_chunks.jsonl
git commit -m "feat(rag): clinical-guidance loader + clause chunker"
```

---

## Task 2: `rag/embedder.py` — local embeddings behind a protocol

**Files:** Create `src/pa_copilot/rag/embedder.py`. Test lives in `tests/test_rag_index.py`
(Task 3) — this task just adds the fast fake + a `@slow` real-model smoke.

**Interfaces:**
- `Embedder` — `typing.Protocol` with `embed_documents(texts: list[str]) -> list[list[float]]`
  and `embed_query(text: str) -> list[float]`.
- `BgeEmbedder(model_name: str, query_prefix: str = "")` — lazy `SentenceTransformer(model_name,
  device="cpu")`; `encode(..., normalize_embeddings=True, show_progress_bar=False)`;
  `embed_query` prepends `query_prefix`. Returns plain `list[float]` (`.tolist()` guarded).
- `EMBED_DIM_BGE_SMALL = 384`.

- [ ] **Step 1: `@slow` smoke test** — add to `tests/test_rag_index.py`:

```python
import pytest
from pa_copilot.rag.embedder import BgeEmbedder, EMBED_DIM_BGE_SMALL


@pytest.mark.slow
def test_bge_embedder_shapes():
    e = BgeEmbedder("BAAI/bge-small-en-v1.5")
    v = e.embed_query("polysomnography medical necessity")
    assert len(v) == EMBED_DIM_BGE_SMALL
    d = e.embed_documents(["home sleep apnea test", "attended in-lab study"])
    assert len(d) == 2 and len(d[0]) == EMBED_DIM_BGE_SMALL
```

- [ ] **Step 2: Run → fails** (import error)
- [ ] **Step 3: Implement `embedder.py`.**
- [ ] **Step 4: Run the slow test locally** (`pytest -m slow -k embedder`) — downloads the
  model once, passes. If offline, note it; the fast path (Task 3) does not need the model.
- [ ] **Step 5: Commit**

```bash
git add src/pa_copilot/rag/embedder.py tests/test_rag_index.py
git commit -m "feat(rag): BgeEmbedder (local sentence-transformers) behind an Embedder protocol"
```

---

## Task 3: `rag/index.py` — Chroma build + filtered search; config fields

**Files:** Create `src/pa_copilot/rag/index.py`, `config/rag.yaml`, `scripts/ingest_rag.py`.
Modify `src/pa_copilot/config.py`, `config/models.yaml`, `tests/test_config.py`,
`tests/test_rag_index.py`. Create committed `traces/rag_index_summary.json`. Modify
`.gitignore`.

**Interfaces:**
- `config/models.yaml` gains `embedding_model: BAAI/bge-small-en-v1.5`.
- `config/rag.yaml`: `collection: pa_guidance`, `top_k: 4`, `min_score: 0.30`,
  `rewrite_min_score: 0.20`, `query_prefix: ""`.
- `config.Settings` gains: `embedding_model: str`, `rag_collection: str`, `rag_top_k: int`,
  `rag_min_score: float`, `rag_rewrite_min_score: float`, `rag_query_prefix: str`.
  `load_settings` reads them (env overrides `PA_EMBEDDING_MODEL`, `PA_RAG_TOP_K`, …).
- `.gitignore`: change `.pa_chroma/` → `.pa_chroma*/`.
- `rag/index.py` — set `os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")` at module
  top **before** `import chromadb`; also pass
  `chromadb.config.Settings(anonymized_telemetry=False, is_persistent=True)` to the client.
  If a telemetry line still reaches stderr, add
  `logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)` (and `"chromadb"`).
  **Acceptance: `python -c "import pa_copilot.rag.index"` and the Task-3 pytest run emit no
  "Failed to send telemetry" line.**
  Produces:
  - `RagIndexSummary` — frozen dataclass: `doc_count: int`, `chunk_count: int`,
    `embedding_model: str`, `corpus_sha: str`, `collection: str`. (**No timestamp** — keeps
    the committed JSON byte-stable.)
  - `corpus_sha(guidance_dir) -> str` — sha1 over sorted `*.md` bytes.
  - `get_client(chroma_dir: str | Path)` / `get_collection(client, name)`.
  - `build_index(settings=None, *, embedder: Embedder | None = None, rebuild: bool = True)
    -> RagIndexSummary` — loads chunks via `corpus.load_guidance(settings.synthetic_dir/…)`,
    embeds `chunk.text` with `embedder` (default `BgeEmbedder(settings.embedding_model)`),
    `collection.upsert(ids, embeddings, documents=[c.text], metadatas=[{policy_id,
    service_code, section, doc_title, clause_index}])`. `rebuild=True` → `delete_collection`
    first (never `rmtree`). Writes `<chroma_dir>/rag_manifest.json` (with timestamp, not
    committed) AND returns the summary.
  - `search(query: str, *, settings=None, embedder: Embedder | None = None, k: int | None =
    None, service_code: str | None = None) -> list[dict]` — returns
    `[{"policy_id","service_code","section","doc_title","clause_index","text","score"}]`
    ordered by score desc. `score = 1 - distance` (cosine). `where={"service_code":
    service_code}` when given. Opens the existing collection; raises
    `RagIndexUnavailable` (a defined Exception) with a "run scripts/ingest_rag.py" message
    if the collection is empty/missing.
- `scripts/ingest_rag.py`: `sys.path` shim, `from pa_copilot.rag.index import build_index`,
  `s = build_index()`, write `traces/rag_index_summary.json` (from
  `dataclasses.asdict(s)`, `indent=2` + `\n`, utf-8/LF), print the summary.

- [ ] **Step 1: Failing tests** — add to `tests/test_rag_index.py`:

```python
import json
from pathlib import Path

from pa_copilot.rag import index as rag_index
from pa_copilot.rag.corpus import load_guidance


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder — no model download."""
    DIM = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.DIM
        for tok in text.lower().split():
            v[hash(tok) % self.DIM] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def test_build_index_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("PA_CHROMA_DIR", str(tmp_path / "chroma"))
    from pa_copilot.config import get_settings
    get_settings.cache_clear()
    fake = FakeEmbedder()
    s1 = rag_index.build_index(embedder=fake, rebuild=True)
    s2 = rag_index.build_index(embedder=fake, rebuild=True)
    assert s1 == s2
    assert s1.chunk_count == len(load_guidance())
    assert s1.doc_count == 6


def test_search_filters_by_service_code(tmp_path, monkeypatch):
    monkeypatch.setenv("PA_CHROMA_DIR", str(tmp_path / "chroma"))
    from pa_copilot.config import get_settings
    get_settings.cache_clear()
    fake = FakeEmbedder()
    rag_index.build_index(embedder=fake, rebuild=True)
    hits = rag_index.search("home sleep study attended polysomnography",
                            embedder=fake, service_code="95810")
    assert hits and all(h["service_code"] == "95810" for h in hits)
    assert "score" in hits[0]


def test_search_without_index_raises_clearly(tmp_path, monkeypatch):
    monkeypatch.setenv("PA_CHROMA_DIR", str(tmp_path / "empty"))
    from pa_copilot.config import get_settings
    get_settings.cache_clear()
    import pytest
    with pytest.raises(rag_index.RagIndexUnavailable):
        rag_index.search("anything", embedder=FakeEmbedder())


def test_no_chroma_telemetry_noise(capfd):
    import importlib
    importlib.reload(rag_index)
    out, err = capfd.readouterr()
    assert "Failed to send telemetry" not in err
```

Also `tests/test_config.py`: assert the new `Settings` fields load from yaml + env override.

- [ ] **Step 2: Run → fails**
- [ ] **Step 3: Implement** `config` changes, `config/*.yaml`, `index.py`, `scripts/ingest_rag.py`.
- [ ] **Step 4: Run → fast tests pass, pristine.** Then `pytest -m slow -k rag` locally
  (real bge-small): `build_index()` + a `search("polysomnography …")` returns PSG chunks
  top-ranked.
- [ ] **Step 5: Generate committed evidence** — `python scripts/ingest_rag.py` (needs the
  real model; if offline, hand-write `traces/rag_index_summary.json` from the known
  `corpus_sha` + counts and note it — the `@slow` test regenerates it). Confirm
  `traces/rag_index_summary.json` is byte-stable on a second run.
- [ ] **Step 6: Commit**

```bash
git add src/pa_copilot/rag/index.py src/pa_copilot/config.py config/models.yaml config/rag.yaml scripts/ingest_rag.py .gitignore tests/test_rag_index.py tests/test_config.py traces/rag_index_summary.json
git commit -m "feat(rag): Chroma index build + filtered search + RAG config (AC-11 index)"
```

---

## Task 4: `rag/tool.py` — the agentic tool + decision predicate (AC-11)

**Files:** Create `src/pa_copilot/rag/tool.py`, `tests/test_ac11_agentic_rag.py`. Create
committed `traces/agentic_rag_decision.md`.

**Interfaces:**
- `should_search_guidance(criteria_status: str, *, unmet_requirements: list[str] | None =
  None) -> bool` — `True` when `criteria_status == "indeterminate"`, OR when it's
  `"not_found"` (no structured policy → narrative is all we have), OR when
  `unmet_requirements` is non-empty (checklist gaps needing interpretation). `False` for
  `"excluded"` (mechanical deny signal — no narrative needed) and any clear `"met"`.
  Docstring states this is the predicate the `medical_necessity` worker (PR5) uses so
  retrieval stays *inside* the agent loop, not a fixed pipeline step.
- `search_clinical_guidance` — a `langchain_core.tools.@tool`:
  `def search_clinical_guidance(query: str, service_code: str | None = None) ->
  list[dict]`. Body: `hits = index.search(query, service_code=service_code)`; if
  `not hits or hits[0]["score"] < settings.rag_min_score`: do **one** corrective rewrite —
  `rewritten = f"{query} medical necessity criteria indications"` (+ the service name from
  `SERVICES` if `service_code` known) — and re-search; keep hits with `score >=
  rewrite_min_score`, capped at `top_k`. Return
  `[CriteriaCitation(source="rag_corpus", clause_id=f"{h['policy_id']}:{h['section']}:{h['clause_index']}",
  quote=h["text"], relevance=f"{h['section']} · score {h['score']:.2f}").model_dump()
  for h in hits]` (list of dicts — MCP/LangChain tool return must be JSON-serializable).
  Empty list is a valid "nothing relevant" answer (PR5's worker then goes
  `indeterminate` → `human_review`).
  The tool uses a module-level default embedder; a `set_tool_embedder(e)` /
  `reset_tool_embedder()` pair lets tests inject the `FakeEmbedder` without a model.

- [ ] **Step 1: Failing test** — `tests/test_ac11_agentic_rag.py`:

```python
import json
from pathlib import Path

import pytest

from pa_copilot.config import get_settings
from pa_copilot.rag import index as rag_index
from pa_copilot.rag.tool import (
    reset_tool_embedder,
    search_clinical_guidance,
    set_tool_embedder,
    should_search_guidance,
)
from tests.test_rag_index import FakeEmbedder


@pytest.fixture
def fake_index(tmp_path, monkeypatch):
    monkeypatch.setenv("PA_CHROMA_DIR", str(tmp_path / "chroma"))
    get_settings.cache_clear()
    fake = FakeEmbedder()
    rag_index.build_index(embedder=fake, rebuild=True)
    set_tool_embedder(fake)
    yield
    reset_tool_embedder()


def test_predicate_gates_on_status():
    assert should_search_guidance("indeterminate") is True
    assert should_search_guidance("not_found") is True
    assert should_search_guidance("met", unmet_requirements=["needs 6wk PT"]) is True
    assert should_search_guidance("excluded") is False
    assert should_search_guidance("met") is False


def test_tool_returns_citations_for_psg(fake_index):
    out = search_clinical_guidance.invoke(
        {"query": "attended polysomnography home sleep test screening", "service_code": "95810"}
    )
    assert isinstance(out, list) and out
    assert all(c["source"] == "rag_corpus" for c in out)
    assert any("95810" in c["clause_id"] or "PA-PSG" in c["clause_id"] for c in out)


def test_tool_tolerates_no_hits(fake_index):
    out = search_clinical_guidance.invoke({"query": "zzzzz nonsense tokens qqqq"})
    assert isinstance(out, list)  # possibly empty, never raises


def test_ac11_decision_evidence_committed():
    md = Path(get_settings().traces_dir) / "agentic_rag_decision.md"
    assert md.exists()
    body = md.read_text(encoding="utf-8")
    assert "indeterminate" in body and "excluded" in body
    assert "search_clinical_guidance" in body
```

- [ ] **Step 2: Run → fails**
- [ ] **Step 3: Implement `tool.py`.**
- [ ] **Step 4: Run → passes, pristine.**
- [ ] **Step 5: Write `traces/agentic_rag_decision.md`** — a short doc + a runnable snippet
  showing: (a) `psg_indeterminate` sample → `criteria_check` returns `indeterminate` →
  `should_search_guidance("indeterminate")` is `True` → `search_clinical_guidance(...)`
  returns N citations (list them, real text from the corpus via the FakeEmbedder or the
  real model); (b) `mri_lumbar_excluded` sample → `criteria_check` returns `excluded` →
  `should_search_guidance("excluded")` is `False` → RAG **not** called. Generate it with a
  committed `scripts/` snippet or inline; keep it byte-stable (no timestamps).
- [ ] **Step 6: Commit**

```bash
git add src/pa_copilot/rag/tool.py tests/test_ac11_agentic_rag.py traces/agentic_rag_decision.md
git commit -m "feat(rag): search_clinical_guidance tool + should_search_guidance predicate (AC-11)"
```

---

## Task 5: docs + AC ledger + `pac ingest` note

**Files:** Modify `docs/design.md` §7 (if it drifts from what shipped), `specs/acceptance-criteria.md`,
`docs/rubric-coverage.md` (create a stub if absent), `Makefile`.

- [ ] **Step 1** — `specs/acceptance-criteria.md`: AC-11 Status →
  `partial (PR3: search_clinical_guidance tool + should_search_guidance predicate +
  Chroma index + traces/agentic_rag_decision.md; full in-graph "agent decides" run PR5)`.
  Confirm the Test column names `tests/test_ac11_agentic_rag.py`.
- [ ] **Step 2** — `docs/design.md` §7: reconcile any wording with what shipped
  (`rag/corpus.py` loader, `rag/embedder.py`, `rag/index.py`, `rag/tool.py`, the predicate
  name `should_search_guidance`, `search()` return shape). Note `pac ingest` (PR7) calls
  `rag.index.build_index`.
- [ ] **Step 3** — `Makefile`: add `ingest-rag:` → `python scripts/ingest_rag.py`. Add to
  `.PHONY`.
- [ ] **Step 4** — run `pytest -q -m "not slow"` (`test_ac_traceability.py` green) + `ruff
  check`.
- [ ] **Step 5: Commit**

```bash
git add docs/design.md specs/acceptance-criteria.md Makefile docs/rubric-coverage.md
git commit -m "docs(rag): AC-11 ledger + design §7 reconcile + ingest-rag target"
```

---

## Task 6: PR3 verification (controller)

- [ ] `pytest -q -m "not slow"` — all pass, pristine (PR2's ~63 + PR3's new).
- [ ] `pytest -q -m slow -k "rag or embedder or bge"` with the model available — real
  index build + search returns the right policy's chunks top-ranked; committed
  `traces/rag_index_summary.json` regenerates byte-identically.
- [ ] `ruff check src tests data scripts conftest.py` — clean.
- [ ] `python -c "import pa_copilot.rag.index"` — **no telemetry line**.
- [ ] Evidence committed: `data/synthetic/clinical_guidance_chunks.jsonl`,
  `traces/rag_index_summary.json`, `traces/agentic_rag_decision.md`. `git status` clean,
  byte-stable on rerun.
- [ ] Hand off to the controller for the whole-branch review. Do NOT merge.

---

## Self-Review

**Spec coverage (design §7; AC-11):**
- §7 corpus (6 committed docs, loader not generator) → Task 1. ✓
- §7 index (Chroma in-process, bge-small local, clause chunks, `pac ingest`) → Tasks 2–3
  (`pac ingest` wiring deferred to PR7, `scripts/ingest_rag.py` + `build_index` provided). ✓
- §7 tool (`search_clinical_guidance(query, service_code) -> list[CriteriaCitation]`, bound
  to `medical_necessity`) → Task 4 (binding happens in PR5). ✓
- §7 "agentic, not a fixed step" — `should_search_guidance` predicate (Task 4) is the hook;
  the full in-graph decision + `traces/agentic_rag_decision.md` narrative → Task 4 evidence
  now, live in PR5. AC-11 marked `partial`. ✓
- §7 corrective sub-loop (weak retrieval → one rewrite) → Task 4 tool body. ✓

**Placeholder scan:** Task 3's telemetry-silencing has a fallback chain (env var → Chroma
Settings → logging level) with a concrete acceptance test (`test_no_chroma_telemetry_noise`),
not a vague "handle warnings". Real-model steps are `@slow` with explicit offline fallbacks
for the committed artifacts. All code steps carry code.

**Type consistency:** `Chunk` fields (Task 1) → `metadatas` keys in `build_index` (Task 3)
→ `search()` return keys (Task 3) → `clause_id` composition in `tool.py` (Task 4) all use
the same names (`policy_id`, `service_code`, `section`, `clause_index`, `doc_title`).
`FakeEmbedder` (defined in `test_rag_index.py`) is imported by `test_ac11_agentic_rag.py` —
consistent. `RagIndexUnavailable` / `RagIndexSummary` names match between `index.py` and its
tests. `should_search_guidance` / `set_tool_embedder` / `reset_tool_embedder` names match
between `tool.py` and `test_ac11_agentic_rag.py`.
