# Rubric Coverage — By Category

The full 22-parameter self-audit (per Section 9 of `PROBLEM_STATEMENT.md`) is compiled in PR7.

| Category | PR(s) | Status |
|---|---|---|
| **Business & Requirements** | PR1 | Foundations: `docs/business-case.md`, acceptance criteria, single-vs-multi decision |
| **Agent Architecture & LangGraph** | PR5 | Pending: full graph wiring (supervisor, workers, conditional routing, checkpointer) |
| **Patterns & Multi-Agent** | PR5, PR6 | Pending: multi-agent orchestration transcript (PR5), reflection loop (PR6) |
| **Context Engineering** | PR5 | Pending: write/select/compress/isolate strategies, summarization middleware, quarantine |
| **Memory Systems** | PR4 | Done: tiered memory (`memory/working.py` + `PolicyStore`), cross-session persistence test + log (AC-07), TTL/importance/LRU eviction policy (AC-08) — see `docs/memory-policy.md`, `tests/test_ac06_tiered_memory.py`, `tests/test_memory_persistence.py`, `tests/test_ac08_eviction.py` |
| **MCP & Interoperability** | PR2 | Partial: custom MCP server (3 tools + 1 resource), adapter integration; full in-graph transcript PR7 |
| **Agentic RAG & Reproducibility** | PR3 | Partial: search_clinical_guidance tool + should_search_guidance predicate + Chroma index (PR3); full in-graph "agent decides" run PR5; reproducibility (README, single command) PR7 |
