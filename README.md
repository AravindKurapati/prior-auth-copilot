# Prior-Authorization Copilot — Multi-Agent Authorization

> Capstone AAIE_AGT_009_HLC · Healthcare — Prior Authorization · LangGraph multi-agent +
> custom MCP server + engineered context + tiered cross-session memory.

A decision-**support** copilot for payer prior-authorization intake. A provider submits a
request; the copilot captures it, checks benefits, evaluates medical-necessity criteria,
and drafts an **approve / deny / refer-to-clinical-review** recommendation with cited
criteria — carrying context across a multi-turn, potentially multi-session case. Final
determination always stays with a human reviewer.

Built on **LangGraph** (hand-rolled supervisor graph, typed state, conditional routing,
SQLite checkpointer), a **custom MCP server** (3 tools + 1 resource over stdio) consumed via
`langchain-mcp-adapters`, engineered context (write / select / compress / isolate +
summarization + quarantine of untrusted provider text), and **tiered memory**
(`SqliteSaver` working memory + `SqliteStore` semantic long-term memory with TTL /
importance / LRU eviction) whose cross-session persistence is proven by a committed test.
Google Gemini is the only LLM provider. Runs with `pip` + Python alone — no Docker, no
database service.

## Status

Under active development — see `docs/design.md` for the full spec and `docs/rubric-coverage.md`
for the acceptance-criteria ledger. Built PR-by-PR (`git merge --no-ff`); see git history.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows;  macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]" -c constraints.txt          # constraints.txt pins the tested resolution
cp .env.example .env                                # add GEMINI_API_KEY

make ingest-rag                                     # build the clinical-guidance RAG index (.pa_chroma/); needed before
                                                    # search_clinical_guidance / pac submit work on a fresh clone
pac all                                             # ingest -> sample battery -> persistence test -> compare
pac submit data/samples/mri_lumbar_clearcut.json    # run one request through the graph
pac demo                                            # Streamlit UI (routing + memory state)
```

## Documentation

| Doc | What |
|---|---|
| `docs/PROBLEM_STATEMENT.md` | the capstone brief, verbatim |
| `docs/business-case.md` | problem, actors, success metrics |
| `docs/design.md` | full architecture + component spec |
| `specs/acceptance-criteria.md` | AC-01..AC-12, each mapped to a test + committed artifact |
| `specs/nfr.md` | NFR-01..NFR-08 |
| `docs/single-vs-multi-agent.md` | orchestration decision + comparison observations |
| `docs/integration-decision.md` | MCP vs API vs direct-DB vs A2A |
| `docs/context-engineering.md` | write / select / compress / isolate map |
| `docs/memory-policy.md` | tiered memory + eviction / importance policy |
| `docs/agent-patterns.md` | plan-execute + reflection rationale |
| `docs/rubric-coverage.md` | 22-parameter self-audit |

## Data

All data is synthetic and generated in-repo (`data/synthetic/`). No real member, provider,
or PHI data. Synthetic PII is redacted before it reaches any committed trace.
