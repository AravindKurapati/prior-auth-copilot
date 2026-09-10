"""Custom FastMCP server — a thin MCP surface over the :mod:`data_access` layer.

Every tool/resource body is a one-line delegation to a ``data_access`` function.
The tool docstrings are the descriptions the agent sees, so they say *when* to
reach for each tool, not just what it does.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from pa_copilot.mcp_server import data_access

mcp = FastMCP("pa-copilot", log_level="WARNING")


@mcp.tool()
def benefit_lookup(member_id: str, service_code: str) -> dict:
    """Check whether a member's plan covers a service and whether it needs prior auth.

    Use this first for any coverage question — it returns ``covered``,
    ``requires_pa``, ``network_status`` and the plan id for the given
    ``member_id`` / ``service_code`` pair.
    """
    return data_access.benefit_lookup(member_id, service_code)


@mcp.tool()
def provider_lookup(npi: str) -> dict:
    """Resolve an ordering provider by NPI to name, specialty, and network status.

    Use this to confirm the requesting provider exists and to check whether they
    are in-network before finalizing a determination.
    """
    return data_access.provider_lookup(npi)


@mcp.tool()
def criteria_check(
    service_code: str, diagnosis_codes: list[str] | None = None
) -> dict:
    """Run the mechanical medical-necessity check for a service against its policy.

    Use this to fetch the prior-auth policy for ``service_code`` and screen the
    supplied ``diagnosis_codes`` against documented exclusions. ``diagnosis_codes``
    is optional — omit it for a pure policy lookup (a request with no coded
    diagnosis), and ``status`` is then ``indeterminate``. ``status`` is one of
    ``not_found`` / ``excluded`` / ``indeterminate`` — ``indeterminate`` means a
    policy exists but a human must verify the clinical conditions.
    """
    return data_access.criteria_check(service_code, diagnosis_codes)


@mcp.resource("pa://criteria/index")
def criteria_index() -> dict:
    """Index of every prior-auth policy as ``{policy_id, service_code, title}``."""
    return data_access.list_policies()


@mcp.resource("pa://criteria/{policy_id}")
def criteria_policy(policy_id: str) -> dict:
    """Full prior-auth policy document for ``policy_id``."""
    return data_access.get_policy(policy_id)
