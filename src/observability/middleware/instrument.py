"""High-level instrumentation helpers — the only API call sites touch.

Each ``observe_*`` context manager bundles the three signals for one
execution boundary so the agent/tool/LLM code stays clean and the
correlation between span + metric + log is guaranteed identical
everywhere:

    async with observe_agent("RAGAgent") as obs:
        ...
        obs.set_tools(selected, skipped)
        obs.set_success(True)

The helper opens the span (with correlation attributes), starts the
timer, increments the right counters, and — crucially — records
duration and failure in a ``finally`` so an exception is still measured
before it propagates. Call sites never duplicate that bookkeeping.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from opentelemetry.trace import SpanKind

from src.observability.context import agent_scope
from src.observability.metrics import registry as M
from src.observability.metrics.cost import estimate_cost_usd, estimate_tokens
from src.observability.tracing import attributes as A
from src.observability.tracing import span


# ------------------------------- Agent -----------------------------------
class AgentObservation:
    def __init__(self, sp) -> None:
        self._sp = sp
        self.failed = False

    def set_tools(self, agent: str, selected: list[str], skipped: list[str]) -> None:
        if self._sp.is_recording():
            self._sp.set_attribute(A.AGENT_SELECTED_TOOLS, selected)
            self._sp.set_attribute(A.AGENT_SKIPPED_TOOLS, skipped)
        for tool in selected:
            M.AGENT_TOOL_SELECTION.labels(agent, tool, "selected").inc()
        for tool in skipped:
            M.AGENT_TOOL_SELECTION.labels(agent, tool, "skipped").inc()

    def set_success(self, success: bool) -> None:
        self.failed = not success
        if self._sp.is_recording():
            self._sp.set_attribute(A.AGENT_SUCCESS, success)


@asynccontextmanager
async def observe_agent(agent_name: str) -> AsyncIterator[AgentObservation]:
    """Wrap one ``agent.handle()``: span + invocation/duration/failure metrics."""
    layer = A.layer_for(agent_name)
    M.AGENT_INVOCATIONS.labels(agent_name, layer).inc()
    start = time.perf_counter()
    with agent_scope(agent_name):
        async with span(
            A.SPAN_AGENT,
            kind=SpanKind.INTERNAL,
            attributes={A.AGENT_NAME: agent_name, A.AGENT_LAYER: layer},
        ) as sp:
            obs = AgentObservation(sp)
            try:
                yield obs
            except Exception:
                obs.failed = True
                raise
            finally:
                M.AGENT_DURATION.labels(agent_name, layer).observe(
                    time.perf_counter() - start
                )
                if obs.failed:
                    M.AGENT_FAILURES.labels(agent_name, layer).inc()


# -------------------------------- Tool ------------------------------------
class ToolObservation:
    def __init__(self, sp) -> None:
        self._sp = sp
        self.failed = False

    def set_result(self, success: bool, preview: str = "") -> None:
        self.failed = not success
        if self._sp.is_recording():
            self._sp.set_attribute(A.TOOL_SUCCESS, success)
            if preview:
                self._sp.set_attribute(A.TOOL_RESULT_PREVIEW, preview[:200])


@asynccontextmanager
async def observe_tool(
    tool_name: str, agent_name: str, rationale: str = ""
) -> AsyncIterator[ToolObservation]:
    """Wrap one deterministic tool call: span + duration/usage/failure metrics."""
    M.TOOL_USAGE.labels(tool_name).inc()
    start = time.perf_counter()
    async with span(
        A.SPAN_TOOL,
        kind=SpanKind.INTERNAL,
        attributes={
            A.TOOL_NAME: tool_name,
            A.AGENT_NAME: agent_name,
            A.TOOL_RATIONALE: rationale[:200] if rationale else None,
        },
    ) as sp:
        obs = ToolObservation(sp)
        try:
            yield obs
        except Exception:
            obs.failed = True
            raise
        finally:
            M.TOOL_DURATION.labels(tool_name, agent_name).observe(
                time.perf_counter() - start
            )
            if obs.failed:
                M.TOOL_FAILURES.labels(tool_name, agent_name).inc()


# --------------------------------- LLM ------------------------------------
class LLMObservation:
    def __init__(self, sp, model: str, operation: str) -> None:
        self._sp = sp
        self._model = model
        self._operation = operation
        self.failed = False

    def record_usage(
        self,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        prompt_text: str | None = None,
        completion_text: str | None = None,
    ) -> None:
        """Record token usage + cost. Falls back to char-heuristic if needed."""
        in_tok = input_tokens if input_tokens is not None else estimate_tokens(prompt_text)
        out_tok = (
            output_tokens if output_tokens is not None else estimate_tokens(completion_text)
        )
        cost = estimate_cost_usd(self._model, in_tok, out_tok)
        M.LLM_TOKENS_INPUT.labels(self._model, self._operation).inc(in_tok)
        M.LLM_TOKENS_OUTPUT.labels(self._model, self._operation).inc(out_tok)
        M.LLM_COST_ESTIMATE.labels(self._model, self._operation).inc(cost)
        if self._sp.is_recording():
            self._sp.set_attribute(A.LLM_TOKENS_IN, in_tok)
            self._sp.set_attribute(A.LLM_TOKENS_OUT, out_tok)
            self._sp.set_attribute(A.LLM_COST_USD, cost)

    def mark_failure(self) -> None:
        self.failed = True
        M.LLM_FAILURES.labels(self._model, self._operation).inc()


@asynccontextmanager
async def observe_llm(
    model: str, operation: str, mode: str
) -> AsyncIterator[LLMObservation]:
    """Wrap one LLM round-trip: span + latency + (caller-recorded) tokens/cost."""
    start = time.perf_counter()
    async with span(
        A.SPAN_LLM,
        kind=SpanKind.CLIENT,
        attributes={
            A.LLM_MODEL: model,
            A.LLM_OPERATION: operation,
            A.LLM_MODE: mode,
        },
    ) as sp:
        obs = LLMObservation(sp, model, operation)
        try:
            yield obs
        except Exception:
            obs.mark_failure()
            raise
        finally:
            M.LLM_DURATION.labels(model, operation, mode).observe(
                time.perf_counter() - start
            )


# --------------------------------- RAG ------------------------------------
class RetrievalObservation:
    def __init__(self, sp) -> None:
        self._sp = sp

    def record(self, *, docs_returned: int, top_score: float, context_chars: int) -> None:
        M.RETRIEVAL_DOCS_RETURNED.observe(docs_returned)
        M.RETRIEVAL_TOP_SCORE.observe(top_score)
        M.RETRIEVAL_CONTEXT_CHARS.observe(context_chars)
        # "Empty" = nothing actually relevant, even if top-k padded the list.
        if docs_returned == 0 or top_score <= 0.0:
            M.RETRIEVAL_EMPTY.inc()
        if self._sp.is_recording():
            self._sp.set_attribute(A.RAG_DOCS_RETURNED, docs_returned)
            self._sp.set_attribute(A.RAG_TOP_SCORE, top_score)
            self._sp.set_attribute(A.RAG_CONTEXT_CHARS, context_chars)
            self._sp.set_attribute(A.RAG_EMPTY, docs_returned == 0 or top_score <= 0.0)


@asynccontextmanager
async def observe_retrieval(
    query: str, top_k: int
) -> AsyncIterator[RetrievalObservation]:
    """Wrap a retrieval: span + latency + quality signals (docs, score, size)."""
    start = time.perf_counter()
    async with span(
        A.SPAN_RETRIEVAL,
        kind=SpanKind.INTERNAL,
        attributes={A.RAG_QUERY: query[:120], A.RAG_TOP_K: top_k},
    ) as sp:
        try:
            yield RetrievalObservation(sp)
        finally:
            M.RETRIEVAL_DURATION.observe(time.perf_counter() - start)
