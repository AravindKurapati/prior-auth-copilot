"""``python -m pa_copilot.mcp_server`` — run the FastMCP server over stdio.

Keep this quiet: FastMCP owns the stdio framing, so nothing may be written to
stdout before the handshake.
"""

from __future__ import annotations

from pa_copilot.mcp_server.server import mcp

if __name__ == "__main__":
    mcp.run()
