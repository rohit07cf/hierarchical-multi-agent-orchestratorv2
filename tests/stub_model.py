"""Deterministic, network-free SDK ``Model`` for tests.

Identifies which agent is calling by the tool names it was given, and returns
scripted tool-calls / final messages per agent. The real ``@function_tool``
bodies still execute, so the shared-``RunContext`` hand-off (RAG→Summarizer,
Coder→Reviewer) is exercised for real.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import AsyncIterator
from typing import Any

from agents import Model, ModelResponse
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)

_ids = itertools.count(1)


def _usage() -> Usage:
    return Usage(requests=1, input_tokens=10, output_tokens=10, total_tokens=20)


def func_call(name: str, **args: Any) -> ModelResponse:
    """A model turn that calls a single tool."""
    cid = f"call_{next(_ids)}"
    item = ResponseFunctionToolCall(
        type="function_call",
        call_id=cid,
        name=name,
        arguments=json.dumps(args),
        id=f"fc_{cid}",
        status="completed",
    )
    return ModelResponse(output=[item], usage=_usage(), response_id=None)


def message(text: str) -> ModelResponse:
    """A model turn that emits a final assistant message."""
    item = ResponseOutputMessage(
        id=f"msg_{next(_ids)}",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
    )
    return ModelResponse(output=[item], usage=_usage(), response_id=None)


def agent_key(tool_names: set[str]) -> str:
    """Identify the calling agent from the tools it was handed."""
    if "ResearchManagerAgent" in tool_names:
        return "supervisor"
    if "call_rag_agent" in tool_names:
        return "research"
    if "call_coding_agent" in tool_names:
        return "build"
    if "simple_retriever" in tool_names:
        return "rag"
    if "code_generation_tool" in tool_names:
        return "coding"
    if "code_review_tool" in tool_names:
        return "review"
    return "summarizer"  # no tools


class StubModel(Model):
    """Replays scripted ``ModelResponse``s, routed by calling agent."""

    def __init__(self, script: dict[str, list[ModelResponse]]) -> None:
        # one independent iterator per agent key
        self._iters = {k: iter(v) for k, v in script.items()}

    async def get_response(
        self,
        system_instructions: str | None,
        input: Any,
        model_settings: Any,
        tools: list[Any],
        output_schema: Any,
        handoffs: Any,
        tracing: Any,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any = None,
    ) -> ModelResponse:
        key = agent_key({getattr(t, "name", "") for t in tools})
        it = self._iters.get(key)
        if it is not None:
            nxt = next(it, None)
            if nxt is not None:
                return nxt
        return message(f"[stub:{key}] done")  # safety fallback

    async def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        raise NotImplementedError("StubModel does not stream")
        yield  # pragma: no cover

    async def get_retry_advice(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def close(self) -> None:
        return None
