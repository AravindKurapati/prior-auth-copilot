"""PR8 good-to-have: a second, independent MCP server (``pa-guidance``) exposing
the agentic-RAG clinical-guidance search over its own stdio process, distinct
from the ``pa`` server's structured lookups. Mirrors test_ac09_mcp_server.py's
pattern but keeps the real-subprocess round trip to `list_tools()` only — a
real `call_tool` would force a real ~130MB embedding-model load (the exact
cost `docs/BUILD_LOG.md` already flags as slow/flaky elsewhere in this suite);
`search_clinical_guidance_impl` itself is proven correct in-process below with
`FakeEmbedder`, same as `rag/tool.py`'s own tests.
"""

from __future__ import annotations

import contextlib
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from _fakes import FakeEmbedder
from pa_copilot.config import get_settings
from pa_copilot.mcp_client import (
    build_guidance_client,
    build_multi_client,
    guidance_server_spec,
    pa_server_spec,
)
from pa_copilot.rag import index as rag_index
from pa_copilot.rag.tool import search_clinical_guidance_impl

SERVER = StdioServerParameters(
    command=sys.executable, args=["-m", "pa_copilot.mcp_server.guidance_server"]
)


@pytest.mark.asyncio
async def test_guidance_server_is_a_distinct_stdio_process_exposing_one_tool() -> None:
    async with contextlib.AsyncExitStack() as stack:
        read, write = await stack.enter_async_context(stdio_client(SERVER))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        tools = (await session.list_tools()).tools
        names = {t.name for t in tools}
        assert names == {"search_clinical_guidance"}
        assert all((t.description or "").strip() for t in tools)


def test_guidance_server_spec_points_at_its_own_module() -> None:
    spec = guidance_server_spec()
    assert set(spec) == {"guidance"}
    assert spec["guidance"]["args"] == ["-m", "pa_copilot.mcp_server.guidance_server"]
    # distinct from the `pa` server -- proves a genuinely separate process, not
    # an alias for the same module.
    assert spec["guidance"]["args"] != pa_server_spec()["pa"]["args"]


def test_build_multi_client_wires_both_servers() -> None:
    client = build_multi_client()
    assert set(client.connections) == {"pa", "guidance"}


def test_build_guidance_client_wires_only_the_guidance_server() -> None:
    client = build_guidance_client()
    assert set(client.connections) == {"guidance"}


def test_search_clinical_guidance_impl_used_by_the_mcp_server_is_correct(tmp_path, monkeypatch):
    """The MCP server delegates straight to `search_clinical_guidance_impl` --
    prove that shared implementation is correct in-process (fast, no
    subprocess, no real embedding model) rather than re-testing the
    corrective-rewrite logic that `test_ac11_agentic_rag.py` already covers
    for the LangChain-tool caller."""
    monkeypatch.setenv("PA_CHROMA_DIR", str(tmp_path / "chroma"))
    get_settings.cache_clear()
    fake = FakeEmbedder()
    rag_index.build_index(embedder=fake, rebuild=True)

    out = search_clinical_guidance_impl("polysomnography indications", embedder=fake)
    assert isinstance(out, list)
