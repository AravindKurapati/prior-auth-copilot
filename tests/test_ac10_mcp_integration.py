"""AC-10: the LangChain MCP adapter loads the custom ``pa`` server's tools and
resources over a real stdio subprocess (no mocking the adapter), and a loaded
tool round-trips a real lookup.

Machine-readable evidence: ``traces/mcp_tool_calls.jsonl`` (from the non-slow
round-trip test below).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.tools import BaseTool

from pa_copilot.config import load_settings
from pa_copilot.mcp_client import load_pa_tools, load_policy

pytestmark = pytest.mark.asyncio


async def test_ac10_adapter_loads_mcp_tools_as_langchain_tools():
    tools = await load_pa_tools()
    by_name = {t.name: t for t in tools}
    assert isinstance(by_name["criteria_check"], BaseTool)
    assert {"benefit_lookup", "provider_lookup", "criteria_check"} <= set(by_name)


async def test_ac10_loaded_tool_round_trips_through_adapter():
    tools = {t.name: t for t in await load_pa_tools()}
    result = await tools["criteria_check"].ainvoke(
        {"service_code": "72148", "diagnosis_codes": ["M54.16"]}
    )
    payload = json.loads(result) if isinstance(result, str) else result
    assert payload["policy_id"] == "PA-MRI-LUMBAR"

    # record a structured tool-call line as evidence
    log = Path(load_settings(env_file=None).traces_dir) / "mcp_tool_calls.jsonl"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "via": "langchain-mcp-adapters",
            "tool": "criteria_check",
            "args": {"service_code": "72148", "diagnosis_codes": ["M54.16"]},
            "result_status": payload["status"],
        }) + "\n")


async def test_ac10_resource_read_through_adapter():
    text = await load_policy("PA-EGD")
    assert "43239" in text
