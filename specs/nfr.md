# Non-Functional Requirements — NFR-01 .. NFR-08

| ID | Requirement | Evidence / mechanism | Test | PR | Status |
|---|---|---|---|---|---|
| **NFR-01** | No secrets or API keys committed; env-var config with a committed `.env.example` | `.env.example` committed; `.env` in `.gitignore`; all keys read via `os.environ` in `config.py` | `tests/test_nfr01_no_secrets.py` — scans the tree for key-shaped strings and for a committed `.env` | scaffold, PR7 | done (.env.example + env-var config loader) |
| **NFR-02** | Runs end-to-end from a single documented command with committed sample inputs and a README quick-start | `pac all`; `data/samples/*.json`; README "Quick start" | `tests/test_nfr02_smoke.py` — invokes the pipeline on a sample with mocked LLM, asserts a schema-valid `PADecision` | PR7 | pending |
| **NFR-03** | Untrusted free-text provider content is isolated (quarantine) and never trusted as instructions | `context/quarantine.py` — raw text stored under `quarantine_ref`, only `intake` reads it, always in a user-role message with a "do not follow instructions" preamble | `tests/test_nfr03_quarantine.py` — includes a prompt-injection canary sample; asserts disposition not forced, `quarantine_flag` recorded; `traces/quarantine_canary.json` | PR5 | pending |
| **NFR-04** | Structured JSON logs / traces of agent runs are committed as evidence | `tracing.py` writes one JSON trace per run to `traces/` | `tests/test_nfr04_trace_schema.py` — validates every `traces/*.json` against the trace schema | PR1, PR7 | partial (RunTracer landed; traces/ populated from PR5) |
| **NFR-05** | All data synthetic; any PII synthetic and never written to logs in plaintext | synthetic generators in `data/synthetic/`; `tracing.py` redacts `member_id` and configured PII fields before write | `tests/test_nfr05_redaction.py` — asserts `member_id` and synthetic names absent from every `traces/*.json` | PR1 | done (RunTracer redaction) |
| **NFR-06** | The single-vs-multi-agent decision and the framework choice are documented with rationale | `docs/single-vs-multi-agent.md` (decision + `pac compare` observations), `docs/integration-decision.md` (MCP vs API vs direct-DB vs A2A), `docs/agent-patterns.md` | `tests/test_nfr06_docs_present.py` — asserts the docs exist and contain the required sections | PR2, PR7 | pending |
| **NFR-07** | Graceful degradation on tool/model failure: timeouts, retries, explicit exit conditions | `reflection.py` — `asyncio.wait_for` timeouts, `tenacity` backoff (max 3), `MAX_REPLANS` / `MAX_HOPS` / `recursion_limit`, model fallback to lite | `tests/test_nfr07_degradation.py` — a persistently failing tool ends at `human_review`, never an unhandled exception | PR6 | pending |
| **NFR-08** | Context-window management: summarization / compression for long threads | `context/summarization.py` wraps `langmem.short_term.SummarizationNode`; runs on every supervisor turn | `tests/test_nfr08_summarization.py` — a long synthetic thread triggers summarization; `state["context"]` running summary populated; `traces/context_before_after.md` shows token reduction | PR5 | pending |

## Additional rubric-driven requirements (not numbered NFRs, still mandatory)

| Requirement | Evidence | PR |
|---|---|---|
| PR-driven Git history: >= 3 `git merge --no-ff` merges; no direct pushes to `main` | git log / `--first-parent` history | all |
| `docs/business-case.md` covering problem, actors, success metrics | committed doc | PR1 |
| Documented agent pattern (ReAct / plan-execute / reflection) with rationale | `docs/agent-patterns.md` | PR7 |
| 22-parameter rubric self-audit | `docs/rubric-coverage.md` | PR7 |
