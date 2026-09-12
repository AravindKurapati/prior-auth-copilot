"""Shared test doubles.

`tests/` is not a package (no `__init__.py`), so cross-file sharing goes through
this module, which pytest's ``prepend`` import mode puts on ``sys.path`` alongside
the test files. Keep it dependency-light — plain classes, no pytest fixtures.
"""

from __future__ import annotations

import itertools
import zlib
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from pydantic import Field


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder — no model download.

    Uses ``zlib.crc32`` for token bucketing because builtin ``hash()`` on strings
    is per-process randomized (PYTHONHASHSEED), which would make the index
    non-reproducible across runs.
    """

    DIM = 64

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.DIM
        for tok in text.lower().split():
            v[zlib.crc32(tok.encode()) % self.DIM] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


_next_call_id = itertools.count(1)


def ai_tool_call(name: str, args: dict, *, call_id: str | None = None) -> AIMessage:
    """Build an AIMessage with a single scripted tool call. `call_id` defaults
    to a fresh, process-unique id per invocation (rather than a fixed literal
    like "call1") so a test scripting two-or-more tool calls in one message
    thread doesn't silently produce duplicate tool-call ids."""
    if call_id is None:
        call_id = f"call{next(_next_call_id)}"
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


class FakeToolCallingModel(BaseChatModel):
    """Deterministic BaseChatModel double for testing create_react_agent-based
    worker loops with no network calls. `script` drives the tool-calling phase
    (one scripted AIMessage per model turn); `structured_responses` drives the
    separate structured-output call create_react_agent makes when response_format
    is set. Both are consumed in order, once each, across the agent's lifetime —
    build a fresh instance per test."""

    script: list[BaseMessage] = Field(default_factory=list)
    structured_responses: list[Any] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs) -> "FakeToolCallingModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        assert self.script, "FakeToolCallingModel script exhausted"
        msg = self.script.pop(0)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def with_structured_output(self, schema, **kwargs):
        def _pop(*_args, **_kwargs):
            assert self.structured_responses, "FakeToolCallingModel structured_responses exhausted"
            return self.structured_responses.pop(0)

        return RunnableLambda(_pop)

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"
