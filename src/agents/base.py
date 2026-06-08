"""Reasoning-agent base class.

Every agent in the hierarchy follows the same three-step contract:

1. **Reason** over the request + prior context to decide which tools
   to call (`LLMClient.reason()`).
2. **Execute** the chosen tools sequentially (`_execute_tools()`).
3. **Synthesize** a final user-visible response from the tool results
   (`LLMClient.synthesize()`).

Subclasses customize behavior by:

- declaring `tools` (their available `ToolSpec`s),
- providing `system_prompt` and `user_prompt(request)`,
- supplying an optional `_mock_reason_policy()` for offline mode,
- optionally overriding `_mock_synthesize()`,
- optionally overriding `_extra_data()` to forward structured payloads
  to downstream agents (e.g. retrieved docs, generated code).

The base class produces an `AgentTrace`, attaches it to every
`AgentResponse`, and emits one log line per phase so the orchestration
trace is consistent across agents.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from src.llm.client import LLMClient, MockReasonPolicy, get_llm_client
from src.models.requests import AgentRequest
from src.models.responses import AgentResponse
from src.models.trace import (
    AgentReasoning,
    AgentTrace,
    ToolDecision,
    ToolInvocation,
    ToolSpec,
)
from src.observability.middleware import observe_agent, observe_tool

logger = logging.getLogger(__name__)

_PREVIEW_CHARS = 400


class ReasoningAgent(ABC):
    """LLM-powered agent that reasons, selects tools, and synthesizes."""

    name: str = "ReasoningAgent"
    system_prompt: str = "You are a helpful agent."
    tools: list[ToolSpec] = []

    def __init__(self, model: str = "claude-opus-4-8", llm: LLMClient | None = None) -> None:
        self.model = model
        self.llm = llm or get_llm_client(model)

    # ----------------- Public API -----------------

    @property
    def available_tools(self) -> list[str]:
        """Names of all tools this agent can choose to invoke."""
        return [t.name for t in self.tools]

    async def handle(self, request: AgentRequest) -> AgentResponse:
        """Run the reason → execute → synthesize loop and return the response.

        The whole invocation is wrapped in an ``agent.handle`` span (which
        also pushes this agent onto the correlation ``agent_path``), so a
        manager delegating to a worker produces a clean parent-child span
        tree that mirrors the hierarchy.
        """
        async with observe_agent(self.name) as obs:
            self._reset_run_state()
            self._log(f"Reasoning over request: {request.query[:80]!r}")
            # Tool-less agents (e.g. the summarizer) have nothing to select,
            # so skip the reason() LLM round-trip — it would just confirm
            # "no tools". The trace reasoning comes from the deterministic
            # policy instead, with no API call.
            reasoning = (
                await self._reason(request)
                if self.tools
                else self._toolless_reasoning(request)
            )
            self._log(f"Reasoning: {reasoning.reasoning[:120]}")

            skipped = reasoning.skipped_tools or self._infer_skipped(reasoning)
            selected = [d.tool_name for d in reasoning.selected_tools]
            obs.set_tools(self.name, selected, skipped)

            invocations = await self._execute_tools(reasoning.selected_tools)
            for inv in invocations:
                status = "ok" if inv.success else "ERROR"
                self._log(f"Tool {inv.tool_name} [{status}]: {inv.result_preview[:80]}")

            final = await self._synthesize(request, reasoning, invocations)
            self._log(f"Final response generated ({len(final)} chars)")

            # An agent "fails" if every tool it chose to run errored.
            obs.set_success(
                not invocations or any(inv.success for inv in invocations)
            )

            trace = AgentTrace(
                agent_name=self.name,
                llm_mode=self.llm.mode,
                reasoning=reasoning.reasoning,
                available_tools=self.available_tools,
                selected_tools=selected,
                skipped_tools=skipped,
                tool_invocations=invocations,
                final_response=final,
            )
            return AgentResponse(
                agent_name=self.name,
                content=final,
                data=self._extra_data(invocations),
                trace=trace.model_dump(mode="json"),
            )

    # ----------------- Subclass hooks -----------------

    @abstractmethod
    def user_prompt(self, request: AgentRequest) -> str:
        """Build the LLM user-prompt for this request."""

    def _mock_reason_policy(self, request: AgentRequest) -> MockReasonPolicy | None:
        """Return a mock reasoning policy for offline mode (or None)."""
        return None

    def _reset_run_state(self) -> None:
        """Override to clear any per-run instance state before handling.

        Managers that cache worker outputs across tool calls (e.g. the
        retrieved document set forwarded to the summarizer) should clear
        them here so reuse across queries doesn't leak state.
        """

    def _mock_synthesize(
        self,
        request: AgentRequest,
        reasoning: AgentReasoning,
        invocations: list[ToolInvocation],
    ) -> str | None:
        """Return a custom mock synthesis (or None to use the default)."""
        return None

    def _extra_data(self, invocations: list[ToolInvocation]) -> dict[str, Any]:
        """Structured payload to forward to downstream agents."""
        return {}

    # ----------------- Internals -----------------

    async def _reason(self, request: AgentRequest) -> AgentReasoning:
        return await self.llm.reason(
            agent_name=self.name,
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt(request),
            available_tools=self.tools,
            mock_policy=self._mock_reason_policy(request),
        )

    def _toolless_reasoning(self, request: AgentRequest) -> AgentReasoning:
        """Deterministic reasoning for an agent with no tools (no LLM call).

        Reuses the agent's mock-reason policy for descriptive trace text
        when available — it never selects tools for a tool-less agent —
        otherwise falls back to a generic message.
        """
        policy = self._mock_reason_policy(request)
        if policy is not None:
            reasoning = policy()
            reasoning.selected_tools = []
            return reasoning
        return AgentReasoning(
            reasoning="No tools available; responding directly.",
            selected_tools=[],
            skipped_tools=[],
        )

    async def _execute_tools(
        self, decisions: list[ToolDecision]
    ) -> list[ToolInvocation]:
        invocations: list[ToolInvocation] = []
        by_name = {t.name: t for t in self.tools}
        for decision in decisions:
            spec = by_name.get(decision.tool_name)
            if spec is None:
                invocations.append(
                    ToolInvocation(
                        tool_name=decision.tool_name,
                        arguments=decision.arguments,
                        rationale=decision.rationale,
                        result_preview="",
                        success=False,
                        error=f"Unknown tool: {decision.tool_name}",
                    )
                )
                continue
            invocations.append(await self._call_tool(spec, decision))
        return invocations

    async def _call_tool(
        self, spec: ToolSpec, decision: ToolDecision
    ) -> ToolInvocation:
        # A "tool" may be a deterministic function *or* a delegation to a
        # child agent's handle(); either way we time it and tag success.
        # Tool exceptions are caught and returned as failed invocations, so
        # ``observe_tool`` records the failure without aborting the agent.
        async with observe_tool(spec.name, self.name, decision.rationale) as tool_obs:
            try:
                result = spec.handler(**decision.arguments)
                if hasattr(result, "__await__"):
                    result = await result
                preview = self._preview(result)
                tool_obs.set_result(True, preview)
                return ToolInvocation(
                    tool_name=spec.name,
                    arguments=decision.arguments,
                    rationale=decision.rationale,
                    result=result,
                    result_preview=preview,
                    success=True,
                )
            except Exception as e:
                logger.exception("Tool %s raised", spec.name)
                tool_obs.set_result(False)
                return ToolInvocation(
                    tool_name=spec.name,
                    arguments=decision.arguments,
                    rationale=decision.rationale,
                    result=None,
                    result_preview="",
                    success=False,
                    error=str(e),
                )

    async def _synthesize(
        self,
        request: AgentRequest,
        reasoning: AgentReasoning,
        invocations: list[ToolInvocation],
    ) -> str:
        mock_synth = self._mock_synthesize(request, reasoning, invocations)
        if self.llm.mode.value == "mock" and mock_synth is not None:
            return mock_synth
        return await self.llm.synthesize(
            agent_name=self.name,
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt(request),
            tool_results=invocations,
            mock_policy=(lambda _: mock_synth) if mock_synth is not None else None,
        )

    def _infer_skipped(self, reasoning: AgentReasoning) -> list[str]:
        """Default: everything the agent didn't select is considered skipped."""
        selected = {d.tool_name for d in reasoning.selected_tools}
        return [name for name in self.available_tools if name not in selected]

    def _log(self, message: str) -> None:
        logger.info("[%s] %s", self.name, message)

    @staticmethod
    def _preview(value: Any) -> str:
        text = str(value)
        return text if len(text) <= _PREVIEW_CHARS else text[:_PREVIEW_CHARS] + "…"


# Backward-compat alias: some external callers used `BaseAgent`.
BaseAgent = ReasoningAgent
