"""AC-09: the custom FastMCP server exposes the three domain tools and the
criteria resource over stdio, and round-trips real lookups against the synthetic
corpora. Also writes the capability evidence to ``traces/mcp_capabilities.json``.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from pa_copilot.config import load_settings

pytestmark = pytest.mark.asyncio

SERVER = StdioServerParameters(command=sys.executable, args=["-m", "pa_copilot.mcp_server"])


async def _session(cm_stack: contextlib.AsyncExitStack) -> ClientSession:
    read, write = await cm_stack.enter_async_context(stdio_client(SERVER))
    session = await cm_stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session


async def test_ac09_server_exposes_three_tools_and_the_resource() -> None:
    async with contextlib.AsyncExitStack() as stack:
        session = await _session(stack)

        tools = (await session.list_tools()).tools
        names = {t.name for t in tools}
        assert {"benefit_lookup", "provider_lookup", "criteria_check"} <= names
        # docstrings reach the agent as tool descriptions
        assert all((t.description or "").strip() for t in tools)

        templates = (await session.list_resource_templates()).resourceTemplates
        assert any("pa://criteria/{policy_id}" in t.uriTemplate for t in templates)
        resources = (await session.list_resources()).resources
        assert any(str(r.uri) == "pa://criteria/index" for r in resources)

        bl = await session.call_tool(
            "benefit_lookup", {"member_id": "M100001", "service_code": "72148"}
        )
        payload = json.loads(bl.content[0].text)
        assert payload["requires_pa"] is True
        assert payload["covered"] is True

        cc = await session.call_tool(
            "criteria_check", {"service_code": "72148", "diagnosis_codes": ["M54.16"]}
        )
        assert json.loads(cc.content[0].text)["status"] == "indeterminate"

        pl = await session.call_tool("provider_lookup", {"npi": "1093817465"})
        assert json.loads(pl.content[0].text)["found"] is True

        res = await session.read_resource("pa://criteria/PA-PSG")
        assert json.loads(res.contents[0].text)["service_code"] == "95810"

        idx = await session.read_resource("pa://criteria/index")
        assert json.loads(idx.contents[0].text)["policies"]

        caps = {
            "server": "pa-copilot",
            "tools": sorted(names & {"benefit_lookup", "provider_lookup", "criteria_check"}),
            "resource_templates": [t.uriTemplate for t in templates],
            "static_resources": [str(r.uri) for r in resources],
        }
        out = Path(load_settings(env_file=None).traces_dir) / "mcp_capabilities.json"
        out.write_text(json.dumps(caps, indent=2) + "\n", encoding="utf-8", newline="\n")
