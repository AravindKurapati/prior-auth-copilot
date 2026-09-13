"""LangChain adapter wiring for the custom ``pa`` MCP server (AC-10).

Thin layer over ``langchain-mcp-adapters``: it turns the stdio MCP server from
:mod:`pa_copilot.mcp_server` into LangChain ``BaseTool`` objects and reads its
criteria resources. The agent nodes in PR5 consume ``load_pa_tools`` (to bind
tools onto the model) and ``load_policy`` / ``load_policy_dict`` (medical-necessity
worker).

Process model
-------------
In the **sessionless** path (``load_pa_tools()`` with no ``session=``), the adapter
spawns a fresh ``python -m pa_copilot.mcp_server`` subprocess for *every tool
invocation* and tears it down when the call returns — not once per ``get_tools()``.
At ~1.2 s of process startup per call that is a real cost once PR5/PR6 make
10-25 calls per case (it eats into NFR-07's ``asyncio.wait_for`` budgets).

The **session** path binds the loaded tools to a single long-lived process: open
one :func:`pa_session` (or pass an existing ``session=``) and every tool call for
the graph's lifetime routes through that one stdio connection.
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.resources import load_mcp_resources
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.stdio import get_default_environment


def pa_server_spec(
    env: dict[str, str] | None = None, cwd: str | None = None
) -> dict:
    """The stdio connection spec for the single ``pa`` server.

    ``mcp.client.stdio`` scrubs the child environment down to a fixed whitelist
    (:func:`mcp.client.stdio.get_default_environment`), so ``PA_SYNTHETIC_DIR``
    and the other ``PA_*`` overrides do **not** reach the server process unless
    passed here explicitly. When ``env`` is given it is merged on top of that
    whitelist; ``cwd`` sets the server's working directory.
    """
    spec: dict[str, Any] = {
        "command": sys.executable,
        "args": ["-m", "pa_copilot.mcp_server"],
        "transport": "stdio",
    }
    if env:
        spec["env"] = {**get_default_environment(), **env}
    if cwd:
        spec["cwd"] = cwd
    return {"pa": spec}


#: Back-compat module constant — the default spec with no env/cwd overrides.
PA_SERVER_SPEC: dict = pa_server_spec()


def guidance_server_spec(
    env: dict[str, str] | None = None, cwd: str | None = None
) -> dict:
    """The stdio connection spec for the second, independent ``guidance`` server
    (PR8 good-to-have — see ``mcp_server/guidance_server.py``)."""
    spec: dict[str, Any] = {
        "command": sys.executable,
        "args": ["-m", "pa_copilot.mcp_server.guidance_server"],
        "transport": "stdio",
    }
    if env:
        spec["env"] = {**get_default_environment(), **env}
    if cwd:
        spec["cwd"] = cwd
    return {"guidance": spec}


def build_client(
    env: dict[str, str] | None = None, cwd: str | None = None
) -> MultiServerMCPClient:
    """A :class:`MultiServerMCPClient` wired to the single ``pa`` stdio server.

    ``env`` / ``cwd`` are forwarded into the stdio spec (see :func:`pa_server_spec`)
    so a caller — e.g. a PR5 test — can point the server at an alternate
    synthetic-corpus directory instead of the committed one.
    """
    return MultiServerMCPClient(pa_server_spec(env, cwd))


def build_guidance_client(
    env: dict[str, str] | None = None, cwd: str | None = None
) -> MultiServerMCPClient:
    """A :class:`MultiServerMCPClient` wired to the single ``guidance`` stdio
    server (PR8)."""
    return MultiServerMCPClient(guidance_server_spec(env, cwd))


def build_multi_client(
    env: dict[str, str] | None = None, cwd: str | None = None
) -> MultiServerMCPClient:
    """A :class:`MultiServerMCPClient` wired to **both** stdio servers (PR8) —
    ``pa`` (structured lookups) and ``guidance`` (semantic search). Demonstrates
    a genuine multi-server MCP topology in one client."""
    return MultiServerMCPClient({**pa_server_spec(env, cwd), **guidance_server_spec(env, cwd)})


@asynccontextmanager
async def pa_session(
    client: MultiServerMCPClient | None = None,
) -> AsyncIterator[ClientSession]:
    """One long-lived stdio session to the ``pa`` server.

    Hold this for a graph's lifetime and pass the yielded session to
    :func:`load_pa_tools` so every tool call reuses the one subprocess.
    """
    async with (client or build_client()).session("pa") as session:
        yield session


@asynccontextmanager
async def guidance_session(
    client: MultiServerMCPClient | None = None,
) -> AsyncIterator[ClientSession]:
    """One long-lived stdio session to the ``guidance`` server (PR8)."""
    async with (client or build_guidance_client()).session("guidance") as session:
        yield session


async def load_pa_tools(
    client: MultiServerMCPClient | None = None,
    session: ClientSession | None = None,
) -> list[BaseTool]:
    """Load the ``pa`` MCP tools as LangChain tools (``benefit_lookup``,
    ``provider_lookup``, ``criteria_check``).

    With ``session`` the tools are bound to that single live process; without it
    each tool invocation spawns and tears down its own subprocess.
    """
    if session is not None:
        return await load_mcp_tools(session)
    return await (client or build_client()).get_tools()


async def load_guidance_tools(
    client: MultiServerMCPClient | None = None,
    session: ClientSession | None = None,
) -> list[BaseTool]:
    """Load the ``guidance`` MCP tools as LangChain tools (``search_clinical_guidance``,
    PR8). Same session/sessionless split as :func:`load_pa_tools`."""
    if session is not None:
        return await load_mcp_tools(session)
    return await (client or build_guidance_client()).get_tools()


async def load_policy(policy_id: str, client: MultiServerMCPClient | None = None) -> str:
    """Read ``pa://criteria/{policy_id}`` and return its body as text.

    FastMCP serialises the policy dict to a JSON string, so the returned value is
    the JSON document as text (use :func:`load_policy_dict` if you need the dict).
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


async def load_policy_dict(
    policy_id: str, client: MultiServerMCPClient | None = None
) -> dict:
    """:func:`load_policy` parsed into a dict — the shape PR5 callers want."""
    return json.loads(await load_policy(policy_id, client))


async def capability_report(client: MultiServerMCPClient | None = None) -> dict:
    """Small dict of what the ``pa`` server exposes, for startup logging."""
    client = client or build_client()
    tools = await client.get_tools()
    report: dict[str, Any] = {
        "server": "pa",
        "transport": "stdio",
        "tools": [t.name for t in tools],
    }
    async with client.session("pa") as session:
        resources = (await session.list_resources()).resources
        templates = (await session.list_resource_templates()).resourceTemplates
    report["resources"] = [str(r.uri) for r in resources]
    report["resource_templates"] = [t.uriTemplate for t in templates]
    return report
