"""StreamingCallbackHandler for real-time UI updates during agent execution."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any

from agents import Agent, AgentHooks, RunContextWrapper, Tool

from models.streaming_models import StreamingModelResponseStep, StreamingStatus

logger = logging.getLogger(__name__)


class StreamingCallbackHandler(AgentHooks):
    """Callback handler that captures agent lifecycle events for streaming to the UI.

    Implements the AgentHooks interface to intercept tool calls, LLM tokens,
    and agent actions, buffering them for efficient broadcast to the
    Streamlit frontend via an async queue.
    """

    def __init__(self, buffer_size: int = 100) -> None:
        self._queue: asyncio.Queue[StreamingModelResponseStep] = asyncio.Queue()
        self._buffer: deque[StreamingModelResponseStep] = deque(maxlen=buffer_size)
        self._is_active = True

    @property
    def queue(self) -> asyncio.Queue[StreamingModelResponseStep]:
        """Access the async queue for consuming streaming updates."""
        return self._queue

    @property
    def history(self) -> list[StreamingModelResponseStep]:
        """Get the buffered history of streaming steps."""
        return list(self._buffer)

    async def _emit(self, step: StreamingModelResponseStep) -> None:
        """Emit a streaming step to both the queue and buffer."""
        if not self._is_active:
            return
        self._buffer.append(step)
        try:
            self._queue.put_nowait(step)
        except asyncio.QueueFull:
            logger.warning("Streaming queue full, dropping event")

    def stop(self) -> None:
        """Stop the handler from emitting further events."""
        self._is_active = False

    def resume(self) -> None:
        """Resume event emission."""
        self._is_active = True

    # --- AgentHooks interface ---

    async def on_start(
        self, context: RunContextWrapper[Any], agent: Agent
    ) -> None:
        """Called when the agent starts processing."""
        await self._emit(
            StreamingModelResponseStep(
                status=StreamingStatus.STARTED,
                name=agent.name,
                message_fragment=f"Agent {agent.name} started",
            )
        )

    async def on_end(
        self, context: RunContextWrapper[Any], agent: Agent, output: Any
    ) -> None:
        """Called when the agent finishes processing."""
        await self._emit(
            StreamingModelResponseStep(
                status=StreamingStatus.COMPLETED,
                name=agent.name,
                message_fragment=f"Agent {agent.name} completed",
                metadata={"output": str(output)[:500]},
            )
        )

    async def on_tool_start(
        self, context: RunContextWrapper[Any], agent: Agent, tool: Tool
    ) -> None:
        """Called when a tool is about to be executed."""
        await self._emit(
            StreamingModelResponseStep.tool_call(
                name=agent.name,
                tool_name=tool.name,
                params={},
            )
        )

    async def on_tool_end(
        self, context: RunContextWrapper[Any], agent: Agent, tool: Tool, result: str
    ) -> None:
        """Called when a tool finishes execution."""
        await self._emit(
            StreamingModelResponseStep.tool_result(
                name=agent.name,
                tool_name=tool.name,
                result=result[:500] if result else "",
            )
        )

    async def on_handoff(
        self, context: RunContextWrapper[Any], agent: Agent, source: Agent
    ) -> None:
        """Called when control is handed off to this agent."""
        await self._emit(
            StreamingModelResponseStep(
                status=StreamingStatus.IN_PROGRESS,
                name=agent.name,
                message_fragment=f"Handoff from {source.name} to {agent.name}",
                metadata={"source_agent": source.name, "target_agent": agent.name},
            )
        )

    # --- Token streaming helpers ---

    async def on_llm_new_token(self, agent_name: str, token: str) -> None:
        """Called for each new LLM token during streaming.

        This method is called manually when processing raw streaming events
        from Runner.run_streamed(), not as part of the AgentHooks interface.

        Args:
            agent_name: Name of the agent generating the token.
            token: The new token fragment.
        """
        await self._emit(
            StreamingModelResponseStep.token(name=agent_name, fragment=token)
        )

    async def on_agent_action(self, agent_name: str, tool_name: str, params: dict[str, Any]) -> None:
        """Called when an agent selects a tool to execute.

        Args:
            agent_name: Name of the agent taking the action.
            tool_name: Name of the selected tool.
            params: Parameters for the tool call.
        """
        await self._emit(
            StreamingModelResponseStep.tool_call(
                name=agent_name,
                tool_name=tool_name,
                params=params,
            )
        )

    async def on_hitl_pause(self, agent_name: str, reason: str) -> None:
        """Called when execution pauses for HITL review.

        Args:
            agent_name: Name of the agent being paused.
            reason: Reason for the pause.
        """
        await self._emit(
            StreamingModelResponseStep.hitl_pause(name=agent_name, reason=reason)
        )

    async def on_error(self, agent_name: str, error: str) -> None:
        """Called when an error occurs during execution.

        Args:
            agent_name: Name of the agent that encountered the error.
            error: Error message.
        """
        await self._emit(
            StreamingModelResponseStep.error(name=agent_name, error_msg=error)
        )
