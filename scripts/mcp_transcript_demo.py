"""AC-10 full (PR7): regenerates the REAL tool-call/result halves of
traces/mcp_toolcall_transcript.md and traces/mcp_tool_calls.jsonl from a genuine
stdio MCP session against pa_copilot.mcp_server -- no GEMINI_API_KEY needed,
since only the two tool calls run (no model call). The narrative
(user/assistant reasoning) text in the .md stays hand-written and
representative, per its own committed note; only the two ```json``` blocks are
regenerated from a live tool invocation, matching the existing file's exact
member/service/diagnosis so nothing else in the surrounding narrative goes
stale.

    python scripts/mcp_transcript_demo.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

_TRANSCRIPT_PATH = _REPO_ROOT / "traces" / "mcp_toolcall_transcript.md"
_JSONL_PATH = _REPO_ROOT / "traces" / "mcp_tool_calls.jsonl"

_BENEFIT_ARGS = {"member_id": "M100001", "service_code": "72148"}
_CRITERIA_ARGS = {"service_code": "72148", "diagnosis_codes": ["M54.16"]}


_JSON_BLOCK = re.compile(r"```json\n.*?\n```", re.DOTALL)


def _replace_json_blocks_in_order(text: str, new_blocks: list[str]) -> str:
    """Replace each ```json ... ``` fenced block in `text`, in document order,
    with the corresponding entry in `new_blocks` -- used to swap in the real
    tool-call args/result blocks without touching the surrounding hand-written
    narrative. The transcript's fixed structure (call, result, call, result) is
    verified by the caller matching `len(new_blocks)` to the actual block count."""
    matches = list(_JSON_BLOCK.finditer(text))
    if len(matches) != len(new_blocks):
        raise RuntimeError(
            f"expected {len(new_blocks)} ```json blocks in the transcript, found {len(matches)}"
        )
    out = []
    cursor = 0
    for match, new_json in zip(matches, new_blocks):
        out.append(text[cursor:match.start()])
        out.append(f"```json\n{new_json}\n```")
        cursor = match.end()
    out.append(text[cursor:])
    return "".join(out)


async def main() -> None:
    from pa_copilot.mcp_client import build_client, load_pa_tools, pa_session

    client = build_client()
    async with pa_session(client) as session:
        tools = await load_pa_tools(session=session)
        by_name = {t.name: t for t in tools}

        benefit_raw = await by_name["benefit_lookup"].ainvoke(_BENEFIT_ARGS)
        benefit_result = json.loads(benefit_raw) if isinstance(benefit_raw, str) else benefit_raw

        criteria_raw = await by_name["criteria_check"].ainvoke(_CRITERIA_ARGS)
        criteria_result = json.loads(criteria_raw) if isinstance(criteria_raw, str) else criteria_raw

    text = _TRANSCRIPT_PATH.read_text(encoding="utf-8")
    text = _replace_json_blocks_in_order(text, [
        json.dumps(_BENEFIT_ARGS, indent=2),
        json.dumps(benefit_result, indent=2),
        json.dumps(_CRITERIA_ARGS, indent=2),
        json.dumps(criteria_result, indent=2),
    ])
    _TRANSCRIPT_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {_TRANSCRIPT_PATH.relative_to(_REPO_ROOT).as_posix()}")

    jsonl_lines = [json.dumps(c, sort_keys=False) for c in [
        {"via": "langchain-mcp-adapters", "tool": "benefit_lookup", "args": _BENEFIT_ARGS, "result_status": "covered" if benefit_result.get("covered") else "not_covered"},
        {"via": "langchain-mcp-adapters", "tool": "criteria_check", "args": _CRITERIA_ARGS, "result_status": criteria_result.get("status")},
    ]]
    _JSONL_PATH.write_text("\n".join(jsonl_lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {_JSONL_PATH.relative_to(_REPO_ROOT).as_posix()}")


if __name__ == "__main__":
    asyncio.run(main())
