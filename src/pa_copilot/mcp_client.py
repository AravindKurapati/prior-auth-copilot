"""LangChain adapter wiring for the custom ``pa`` MCP server (AC-10).

Thin layer over ``langchain-mcp-adapters``: it turns the stdio MCP server from
:mod:`pa_copilot.mcp_server` into LangChain ``BaseTool`` objects and reads its
criteria resources as plain text. The agent nodes in PR5 consume ``load_pa_tools``
(to bind tools onto the model) and ``load_policy`` (medical-necessity worker).

``MultiServerMCPClient`` spawns a fresh stdio subprocess per ``get_tools()`` call
and per ``session()`` context, and tears it down when that call / context exits —
so there is no long-lived resource to close here.
"""

from __future__ import annotations

import sys

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.resources import load_mcp_resources

PA_SERVER_SPEC: dict = {
    "pa": {
        "command": sys.executable,
        "args": ["-m", "pa_copilot.mcp_server"],
        "transport": "stdio",
    }
}


def build_client() -> MultiServerMCPClient:
    """A :class:`MultiServerMCPClient` wired to the single ``pa`` stdio server."""
    return MultiServerMCPClient(PA_SERVER_SPEC)


async def load_pa_tools(client: MultiServerMCPClient | None = None) -> list[BaseTool]:
    """Load the ``pa`` MCP tools as LangChain tools (``benefit_lookup``,
    ``provider_lookup``, ``criteria_check``)."""
    return await (client or build_client()).get_tools()


async def load_policy(policy_id: str, client: MultiServerMCPClient | None = None) -> str:
    """Read ``pa://criteria/{policy_id}`` and return its body as text.

    FastMCP serialises the policy dict to a JSON string, so the returned value is
    the JSON document as text (parse with ``json.loads`` if you need the dict).
    """
    client = client or build_client()
    async with client.session("pa") as session:
        blobs = await load_mcp_resources(session, uris=[f"pa://criteria/{policy_id}"])
    if not blobs:
        raise LookupError(f"no resource content for pa://criteria/{policy_id}")
    blob = blobs[0]
    try:
        return blob.as_string()
    except (UnicodeDecodeError, ValueError):
        data = blob.data
        if isinstance(data, (bytes, bytearray)):
            return bytes(data).decode("utf-8", errors="replace")
        return str(data)


async def capability_report(client: MultiServerMCPClient | None = None) -> dict:
    """Small dict of what the ``pa`` server exposes, for startup logging."""
    tools = await (client or build_client()).get_tools()
    return {
        "server": "pa",
        "transport": "stdio",
        "tools": [t.name for t in tools],
    }
