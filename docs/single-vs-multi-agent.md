# Single-Agent vs. Multi-Agent: Decision & Comparison

## Context

The capstone rubric (`docs/PROBLEM_STATEMENT.md` §9, NFR-06) requires the single-vs-multi
orchestration decision to be documented with rationale, not just implemented. This project
ships the multi-agent design (`docs/design.md` §3) as the production path, and PR7 adds
`single_agent.py` + `pac compare` purely as the comparison baseline the decision is judged
against — a real, runnable alternative, not a strawman.

## Decision

**Multi-agent (supervisor + 4 specialized workers)** is the production architecture:
`supervisor.py` routes to `intake` / `benefit_check` / `medical_necessity` /
`decision_draft` / `human_review`, each a narrow `create_react_agent` ReAct loop bound to
only the tools and context it needs (`design.md` §3.2–§3.3). `single_agent.py` exists
solely as `pac compare`'s baseline: one ReAct loop bound to the union of every tool the
four workers collectively use, no supervisor, no per-worker context scoping.

## Rationale

- **Context selection (design.md §5.1, `context/assembly.py::select_for`)**: each
  multi-agent worker sees only the state fields relevant to its job (e.g. `decision_draft`
  never sees `raw_provider_text`). The single agent has no such boundary — every tool call
  and every prior turn stays in one shared message history, so its context grows
  monotonically across the whole adjudication instead of being scoped per phase.
- **Quarantine (design.md §5.1, NFR-03)**: only `intake` ever reads `raw_provider_text`,
  inside a user-role message with an explicit "extract fields only" preamble. The single
  agent's system prompt states the same warning in words, but there is no structural
  boundary — quarantine as *isolation* (a role/field boundary) collapses to quarantine as
  *instruction* (a sentence in the prompt) once there is only one agent.
- **Reflection scope (design.md §3.5, AC-12, PR6)**: `needs_replan` targets re-routing to a
  *specific* worker (e.g. back to `medical_necessity` alone on low confidence). The single
  agent has no worker boundary to re-target — a retry re-runs the entire adjudication loop,
  not just the step that was uncertain.
- **Specialization**: each worker's system prompt is narrow and single-purpose
  (`agents/medical_necessity.py`'s prompt only ever talks about criteria/necessity). The
  single agent's one system prompt must describe the entire task, which is more prone to
  the model conflating unrelated sub-tasks (e.g. drafting a decision before benefits are
  confirmed) since nothing structurally prevents it.
- **Tool-failure isolation (PR6, `reflection.py`)**: a tool failure inside one multi-agent
  worker is attributed to that worker's own turn and retried/replanned in isolation. In the
  single agent, every tool shares one ReAct loop, so a failing tool retry consumes the same
  attempt budget as every other tool call in the run.

## Observed Differences (`pac compare data/samples/mri_lumbar_clearcut.json`)

`pac compare` runs the same synthetic case through both paths and prints both `PADecision`s
side by side. In this build environment there is no `GEMINI_API_KEY` (standing constraint,
`docs/BUILD_LOG.md`), so a live head-to-head run is not available here — this is not a gap
specific to this comparison; every other LLM-needing artifact in this repo is representative
for the same reason. The qualitative differences above (context growth, quarantine strength,
reflection granularity, tool-failure isolation) are structural, not run-dependent, and hold
regardless of which model backs either path. **Regenerate this section's numbers** by running
`pac compare data/samples/mri_lumbar_clearcut.json` with a real `GEMINI_API_KEY` configured,
and record: final disposition agreement/disagreement, confidence delta, and total tool-call
count per path (the single agent is expected to make more redundant tool calls, since it has
no per-worker memory of what a different "worker" already checked).
