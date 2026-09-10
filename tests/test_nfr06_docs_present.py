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
