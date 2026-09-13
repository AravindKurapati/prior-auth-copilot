# Agent Patterns Used

This project combines three agent patterns, each addressing a different concern
(`docs/PROBLEM_STATEMENT.md` §9, NFR-06's "documented agent pattern" requirement).

## Plan-Execute

`supervisor.py` is the planner: on every turn it decides which worker runs next, either
deterministically (`hard_route` — e.g. `request is None -> intake`) or via an LLM router
(`route_with_llm`) when no deterministic rule applies. The four workers
(`agents/intake.py`, `benefit_check.py`, `medical_necessity.py`, `decision_draft.py`) are
the executors — each does one bounded piece of the overall plan and returns control to the
supervisor. This is a plan-execute loop, not a single flat agent: the supervisor never
executes a tool itself, and no worker ever chooses its own successor (`design.md` §3.1–§3.3).

## ReAct

Each worker is a `create_react_agent` loop (`agents/_react.py::run_worker_react`,
wrapped by PR6's `run_worker_react_resilient`): given a system prompt and its bound tools,
the worker reasons about which tool to call, observes the result, and repeats until it
produces a structured output (`PARequest`, `BenefitResult`, `NecessityAssessment`, or
`PADecision`). `intake` alternates provider/benefit lookups with reasoning about missing
fields; `medical_necessity` alternates `criteria_check`/`search_clinical_guidance` calls
with reasoning about whether the criteria are met. This is the classic ReAct
interleaving of reasoning and tool use, scoped to one worker's turn at a time
(`design.md` §3.3).

## Reflection / Self-Healing

PR6's dual-trigger loop (`design.md` §3.5–§3.6, AC-12, NFR-07): a tool failure or a
low-confidence/indeterminate `NecessityAssessment` sets `needs_replan`, which
`build_supervisor_node` counts against a shared `replan_count`/`max_replans` budget and
surfaces as an explicit hint to the LLM router (`route_with_llm`), which may loop back to
the same worker or escalate to `human_review`. Beneath that, `reflection.py` adds two
narrower resilience mechanisms inside a single worker turn: `tenacity`-backed retries on a
transient `WorkerToolError`, an `asyncio.wait_for` timeout around the whole turn
(`WorkerTimeoutError`, excluded from retry — a blown time budget isn't transient), and one
fallback attempt on a lite model when structured-output validation itself fails
(`WorkerOutputError`). Together these give the system both turn-level resilience (retry/
timeout/fallback within one worker call) and case-level self-healing (replan/reroute across
worker turns), evidenced end-to-end by `tests/test_ac12_reflection.py`'s four scripted
scenarios through the real compiled graph.
