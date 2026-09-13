"""``pa-guidance`` — a second, independent FastMCP server (PR8 good-to-have).

Exposes the agentic-RAG clinical-guidance search (`rag/tool.py`'s corrective-
rewrite retrieval) as a stdio MCP tool, separate from the `pa` server's
structured lookups (`benefit_lookup`, `provider_lookup`, `criteria_check`).
Demonstrates a genuinely multi-server MCP topology — a different process,
different capability (semantic search over unstructured guidance text vs.
structured corpus lookups) — via `mcp_client.py`'s `MultiServerMCPClient`.

Run standalone with ``python -m pa_copilot.mcp_server.guidance_server``.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from pa_copilot.rag.tool import search_clinical_guidance_impl

mcp = FastMCP("pa-guidance", log_level="WARNING")


@mcp.tool()
def search_clinical_guidance(query: str, service_code: str | None = None) -> list[dict]:
    """Search the clinical-guidance corpus for medical-necessity criteria.

    Call this when a mechanical criteria check is ``indeterminate`` / ``not_found``
    or leaves unmet requirements. ``query`` is a natural-language description of
    what needs supporting; pass ``service_code`` to scope the search to one
    policy. Returns citation dicts (``source, clause_id, quote, relevance``); an
    empty list is a valid "nothing relevant found" answer.
    """
    return search_clinical_guidance_impl(query, service_code)


if __name__ == "__main__":
    mcp.run(transport="stdio")
