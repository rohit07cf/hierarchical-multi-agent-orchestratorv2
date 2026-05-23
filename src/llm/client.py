"""LLM client used by every reasoning agent.

The client exposes two operations:

- `reason()` — given a system prompt, user prompt, and the agent's
  available tools, return an `AgentReasoning` describing which tools
  the agent has decided to call (with rationale) and which were
  skipped.
- `synthesize()` — given the tool results, produce the final natural-
  language response.

When `OPENAI_API_KEY` is unset the client switches to **mock mode**:
`reason()` falls back to a per-agent mock policy (so the orchestration
shape stays visible offline) and `synthesize()` returns a clearly
labelled placeholder. The UI surfaces the mock-mode banner.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Awaitable, Callable

from src.models.trace import (
    AgentReasoning,
    LLMMode,
    ToolDecision,
    ToolInvocation,
    ToolSpec,
)
from src.observability.middleware import observe_llm

logger = logging.getLogger(__name__)


MockReasonPolicy = Callable[[], AgentReasoning]
MockSynthPolicy = Callable[[list[ToolInvocation]], str]
AsyncMockReasonPolicy = Callable[[], Awaitable[AgentReasoning]]


MOCK_PREFIX = "[mock-llm]"


@dataclass
class LLMClient:
    """Thin wrapper around OpenAI Chat Completions with mock fallback."""

    model: str = "gpt-4.1-nano"
    mode: LLMMode = LLMMode.MOCK

    @property
    def enabled(self) -> bool:
        """Whether real-LLM mode is active."""
        return self.mode == LLMMode.REAL

    async def reason(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        available_tools: list[ToolSpec],
        mock_policy: MockReasonPolicy | None = None,
    ) -> AgentReasoning:
        """Ask the LLM to reason about the request and pick tools.

        In mock mode, defers to `mock_policy` if supplied, otherwise
        returns a labelled "skip all tools" reasoning.
        """
        async with observe_llm(self.model, "reason", self.mode.value) as obs:
            if self.mode == LLMMode.MOCK:
                if mock_policy is not None:
                    reasoning = mock_policy()
                    # Always tag mock reasoning so the UI shows the mode.
                    if not reasoning.reasoning.startswith(MOCK_PREFIX):
                        reasoning.reasoning = f"{MOCK_PREFIX} {reasoning.reasoning}"
                else:
                    reasoning = AgentReasoning(
                        reasoning=(
                            f"{MOCK_PREFIX} {agent_name} running in mock mode. "
                            "Set OPENAI_API_KEY for real reasoning."
                        ),
                        selected_tools=[],
                        skipped_tools=[t.name for t in available_tools],
                    )
                obs.record_usage(
                    prompt_text=user_prompt, completion_text=reasoning.reasoning
                )
                return reasoning

            try:
                return await self._openai_reason(
                    agent_name, system_prompt, user_prompt, available_tools, obs
                )
            except Exception as e:
                logger.warning("LLM reason() failed, falling back to mock: %s", e)
                obs.mark_failure()
                return AgentReasoning(
                    reasoning=f"{MOCK_PREFIX} LLM call failed ({e}); using empty plan.",
                    selected_tools=[],
                    skipped_tools=[t.name for t in available_tools],
                )

    async def synthesize(
        self,
        *,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        tool_results: list[ToolInvocation],
        mock_policy: MockSynthPolicy | None = None,
    ) -> str:
        """Synthesize the agent's final user-facing response."""
        async with observe_llm(self.model, "synthesize", self.mode.value) as obs:
            if self.mode == LLMMode.MOCK:
                if mock_policy is not None:
                    out = mock_policy(tool_results)
                else:
                    out = self._default_mock_synthesize(agent_name, tool_results)
                obs.record_usage(prompt_text=user_prompt, completion_text=out)
                return out

            try:
                return await self._openai_synthesize(
                    system_prompt, user_prompt, tool_results, obs
                )
            except Exception as e:
                logger.warning("LLM synthesize() failed, falling back to mock: %s", e)
                obs.mark_failure()
                return self._default_mock_synthesize(agent_name, tool_results)

    # ----------------- OpenAI implementations -----------------

    async def _openai_reason(
        self,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        available_tools: list[ToolSpec],
        obs,
    ) -> AgentReasoning:
        """Real OpenAI reasoning call — asks for JSON-formatted decision."""
        from openai import AsyncOpenAI

        if not available_tools:
            tools_block = "(no tools available — respond with empty selected_tools)"
        else:
            tools_block = "\n".join(
                f"- {t.name}: {t.description}" for t in available_tools
            )

        full_prompt = (
            f"{user_prompt}\n\n"
            f"Available tools for {agent_name}:\n{tools_block}\n\n"
            "Decide which tools (if any) you need to call to satisfy the "
            "request. Respond with JSON in this exact shape:\n"
            '{"reasoning": "<short chain-of-thought>", '
            '"selected_tools": [{"tool_name": "<name>", "rationale": "<why>", '
            '"arguments": {"<arg>": <value>}}], '
            '"skipped_tools": ["<name>", ...]}'
        )

        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        # Prefer real provider usage; fall back to the char-heuristic.
        usage = getattr(response, "usage", None)
        obs.record_usage(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            prompt_text=user_prompt,
            completion_text=response.choices[0].message.content,
        )
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        return AgentReasoning(
            reasoning=parsed.get("reasoning", ""),
            selected_tools=[
                ToolDecision(
                    tool_name=t.get("tool_name", ""),
                    rationale=t.get("rationale", ""),
                    arguments=t.get("arguments", {}) or {},
                )
                for t in parsed.get("selected_tools", [])
            ],
            skipped_tools=list(parsed.get("skipped_tools", [])),
        )

    async def _openai_synthesize(
        self,
        system_prompt: str,
        user_prompt: str,
        tool_results: list[ToolInvocation],
        obs,
    ) -> str:
        """Real OpenAI synthesis call — free-form completion."""
        from openai import AsyncOpenAI

        if not tool_results:
            tool_block = "(no tools were called)"
        else:
            tool_block = "\n\n".join(
                f"### {ti.tool_name}\nrationale: {ti.rationale}\n"
                f"result:\n{ti.result_preview}"
                for ti in tool_results
            )

        prompt = (
            f"{user_prompt}\n\n"
            f"Tool results:\n{tool_block}\n\n"
            "Write a clean, natural response that uses the tool results "
            "to satisfy the request. Do not mention internal tool names."
        )

        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        usage = getattr(response, "usage", None)
        content = response.choices[0].message.content or ""
        obs.record_usage(
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            prompt_text=prompt,
            completion_text=content,
        )
        return content

    # ----------------- Default mock synthesizer -----------------

    @staticmethod
    def _default_mock_synthesize(
        agent_name: str, tool_results: list[ToolInvocation]
    ) -> str:
        """Generic mock synthesis: list tool outputs verbatim with a banner."""
        if not tool_results:
            return (
                f"{MOCK_PREFIX} {agent_name} produced no tool output. "
                "Set OPENAI_API_KEY for real synthesis."
            )
        lines = [
            f"{MOCK_PREFIX} {agent_name} ran the following tool(s); set "
            "OPENAI_API_KEY for natural-language synthesis.",
        ]
        for ti in tool_results:
            lines.append(f"- {ti.tool_name}: {ti.result_preview}")
        return "\n".join(lines)


def get_llm_client(model: str = "gpt-4.1-nano") -> LLMClient:
    """Return an LLMClient configured from the environment."""
    mode = LLMMode.REAL if os.environ.get("OPENAI_API_KEY") else LLMMode.MOCK
    return LLMClient(model=model, mode=mode)
