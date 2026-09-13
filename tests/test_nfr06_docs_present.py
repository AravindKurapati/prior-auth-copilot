"""NFR-06: Integration decision and framework choice documented with rationale.

Asserts that docs/integration-decision.md exists and contains required sections.
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def test_integration_decision_doc_exists():
    """integration-decision.md must be committed."""
    doc = _ROOT / "docs" / "integration-decision.md"
    assert doc.exists(), "docs/integration-decision.md missing"


def test_integration_decision_sections():
    """integration-decision.md must contain required section headers."""
    doc = _ROOT / "docs" / "integration-decision.md"
    text = doc.read_text(encoding="utf-8")

    # anchor on the Markdown heading form so a bare word in prose (e.g. "the
    # integration decision addresses...") can't stand in for a real section.
    required_headings = [
        "## Context",
        "## Options Considered",
        "## Decision",  # matches "## Decision: MCP Server"
        "## Why Not the Others",
        "## Boundary with RAG",  # matches "## Boundary with RAG (AC-11)"
    ]

    for heading in required_headings:
        assert heading in text, (
            f"docs/integration-decision.md missing section heading: {heading}"
        )


def test_single_vs_multi_agent_doc_exists_and_has_sections():
    """PR7: docs/single-vs-multi-agent.md must be committed with the sections
    the single-vs-multi decision + comparison requires."""
    doc = _ROOT / "docs" / "single-vs-multi-agent.md"
    assert doc.exists(), "docs/single-vs-multi-agent.md missing"
    text = doc.read_text(encoding="utf-8")

    for heading in ["## Context", "## Decision", "## Rationale", "## Observed Differences"]:
        assert heading in text, (
            f"docs/single-vs-multi-agent.md missing section heading: {heading}"
        )


def test_agent_patterns_doc_exists_and_has_sections():
    """PR7: docs/agent-patterns.md must document the ReAct / plan-execute /
    reflection patterns actually implemented."""
    doc = _ROOT / "docs" / "agent-patterns.md"
    assert doc.exists(), "docs/agent-patterns.md missing"
    text = doc.read_text(encoding="utf-8")

    for heading in ["## Plan-Execute", "## ReAct", "## Reflection"]:
        assert heading in text, f"docs/agent-patterns.md missing section heading: {heading}"
