# Integration Decision: MCP Server for Payer Systems

## Context

The copilot requires three kinds of structured, deterministic access to payer systems:

1. **Benefit eligibility lookups** — does the member's plan cover this service, and does it require prior authorization?
2. **Provider registry lookups** — retrieve basic provider details (NPI, name, specialty, network status).
3. **Coverage-criteria lookups** — retrieve the structured checklist of clinical requirements, exclusions, and evidence thresholds for a given service/policy.

In addition, the system accesses unstructured clinical-guidance knowledge (e.g., step-therapy narratives, conservative-care duration guidance) via agentic retrieval. That is handled separately by the RAG system (see `docs/design.md` §7).

The integration decision addresses how the copilot's agents communicate with payer systems for the structured, deterministic lookups.

---

## Options Considered

### (a) Custom MCP Server (chosen)

A FastMCP server running over stdio, exposing three tools (`benefit_lookup`, `provider_lookup`, `criteria_check`) and two resources (`pa://criteria/{policy_id}` for templated policy documents, and `pa://criteria/index` for the static index of all policies), consumed by the copilot via `langchain-mcp-adapters`.

### (b) Plain in-process `@tool` functions

Define the three lookups as Python functions decorated with `@tool` and bound directly to the agent nodes.

### (c) REST / HTTP API service

A separate web service (e.g., FastAPI) running on a local port, with the copilot making HTTP calls to fetch lookups.

### (d) Direct database or file access

The agent workers read directly from `data/synthetic/benefits.json`, `data/synthetic/providers.json`, and `data/synthetic/criteria.json`, or open a direct database connection.

### (e) Agent-to-Agent (A2A)

A separate "payer systems" agent running its own graph, with the copilot's workers sending structured requests to it and awaiting structured responses.

---

## Decision: MCP Server

**We use a custom MCP server over stdio, providing three tools and two resources, consumed via `langchain-mcp-adapters`.**

### Rationale

1. **Boundary modeling** — The MCP server models the genuine inter-system boundary a real payer exposes. The benefit, provider, and criteria systems are separate systems of record with their own APIs/contracts; MCP makes that boundary explicit and protocol-mediated.

2. **Swappability** — The protocol-mediated interface means the backing store (synthetic JSON in this demo, a real payer API in production) can change without touching the agent code. The agent always sees the same tool signatures and resource URIs.

3. **Tools + resources in one interface** — MCP allows both **tools** (deterministic functions like `benefit_lookup(member_id, service_code)`) **and** **resources** (named addressable data like `pa://criteria/{policy_id}` and `pa://criteria/index`). This models a realistic system-of-record: transactional lookups via tools, full documents via resources.

4. **Brief mandate** — The brief specifically requires a custom MCP server consumed via `langchain-mcp-adapters`, and this demonstrates the interoperability requirement.

---

## Why Not the Others

### Plain `@tool`: No boundary, no resource concept

Placing the three lookups as in-process Python functions does not exercise the inter-system boundary. The agent has no concept of calling an external system; the tools are indistinguishable from local computations. No resource concept, and no protocol that could be swapped for a real payer API.

### REST / HTTP API: Breaks the no-service rule

The project mandates no external services and no Docker. A REST API requires a running web server (even if local), violating the "single documented command" rule (NFR-02). MCP over stdio runs in-process with no separate service.

### Direct DB / file access: Couples to storage layout

If workers open `data/synthetic/benefits.json` directly, every worker becomes aware of the storage layout, schema, and file format. Changes to how criteria are stored ripple through the agent code. MCP decouples storage from interface: the server reads the files; the agents see only tool signatures and resource URIs.

### A2A (agent-to-agent): Overhead without autonomy

A2A is justified when two autonomous agents negotiate or have conflicting goals (e.g., a seller and buyer reaching a price). The copilot's workers and a hypothetical "payer systems" agent **share one `LangGraph` state object** (`PACaseState`); there is no negotiation boundary, no autonomy, no reason for separate graphs. A2A here is ceremony over a plain data lookup.

---

## Boundary with RAG (AC-11)

The MCP server and the agentic-RAG system are complementary:

- **MCP = structured, deterministic system-of-record** — `benefit_lookup` returns a definite yes/no/unknown; `criteria_check` returns a structured checklist + a mechanical status (`not_found` / `excluded` / `indeterminate`). These are lookups into versioned, auditable data.
- **RAG = unstructured narrative interpretation** — `search_clinical_guidance()` (planned, PR3) retrieves *why* a criterion matters (step-therapy narrative, conservative-care duration, medical literature snippets). The `medical_necessity` agent **decides** whether to call RAG only when `criteria_check` returns `indeterminate`, or when the checklist needs narrative interpretation.

The split preserves auditability: structured lookups are reproducible deterministic calls; narrative guidance is retrieval + LLM interpretation, flagged differently in traces.

---

## Second MCP Server (Planned, PR8)

A read-only `samples` server exposing `data/samples/*.json` as MCP resources, wired into the same `MultiServerMCPClient` via `mcp_client.py`. This demonstrates multi-server support with minimal additional code, and also lets the agent optionally look up past sample cases.

---

## Implementation

See `docs/design.md` §4:

- **Server**: `src/pa_copilot/mcp_server/server.py` — FastMCP with the three tools and resource namespace.
- **Client**: `src/pa_copilot/mcp_client.py` — `MultiServerMCPClient` initialized at graph build time.
- **Evidence**: `traces/mcp_capabilities.json` (tools + resources), `traces/mcp_tool_calls.jsonl` (logs of every tool invocation), `traces/mcp_toolcall_transcript.md` (annotated stdio session).
- **Tests**: `tests/test_ac09_mcp_server.py` (server spin-up + capability listing), `tests/test_ac10_mcp_integration.py` (agent invokes `criteria_check` through the adapter).
