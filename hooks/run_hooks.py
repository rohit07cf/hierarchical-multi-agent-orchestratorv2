"""Agent and Tool hook implementations for lifecycle event handling."""

from __future__ import annotations

import logging
from typing import Any

from agents import Agent, AgentHooks as BaseAgentHooks, RunContextWrapper, Tool

from models.streaming_models import StreamingModelResponseStep, StreamingStatus

logger = logging.getLogger(__name__)


class AgentHooks(BaseAgentHooks):
    """Custom agent hooks for logging and streaming lifecycle events.

    Provides detailed logging at each agent lifecycle point and
    optionally pushes events to a streaming callback handler.
    """

    def __init__(self, agent_name: str, streaming_handler: Any | None = None) -> None:
        self.agent_name = agent_name
        self._streaming_handler = streaming_handler

    async def on_start(
        self, context: RunContextWrapper[Any], agent: Agent
    ) -> None:
        """Called when the agent starts processing."""
        logger.info("[%s] Agent started", agent.name)
        if self._streaming_handler:
            await self._streaming_handler.on_start(context, agent)

    async def on_end(
        self, context: RunContextWrapper[Any], agent: Agent, output: Any
    ) -> None:
        """Called when the agent finishes processing."""
        logger.info("[%s] Agent completed with output length: %d", agent.name, len(str(output)))
        if self._streaming_handler:
            await self._streaming_handler.on_end(context, agent, output)

    async def on_handoff(
        self, context: RunContextWrapper[Any], agent: Agent, source: Agent
    ) -> None:
        """Called when control is handed off to this agent from another."""
        logger.info("[%s] Handoff received from %s", agent.name, source.name)
        if self._streaming_handler:
            await self._streaming_handler.on_handoff(context, agent, source)

    async def on_tool_start(
        self, context: RunContextWrapper[Any], agent: Agent, tool: Tool
    ) -> None:
        """Called immediately before a tool is invoked."""
        logger.info("[%s] Tool starting: %s", agent.name, tool.name)
        if self._streaming_handler:
            await self._streaming_handler.on_tool_start(context, agent, tool)

    async def on_tool_end(
        self, context: RunContextWrapper[Any], agent: Agent, tool: Tool, result: str
    ) -> None:
        """Called immediately after a tool finishes."""
        logger.info(
            "[%s] Tool completed: %s (result length: %d)",
            agent.name,
            tool.name,
            len(str(result)),
        )
        if self._streaming_handler:
            await self._streaming_handler.on_tool_end(context, agent, tool, result)


class ToolHooksImpl:
    """Hooks for tool-level lifecycle events.

    Provides pre/post execution hooks for individual tools,
    enabling logging, validation, and metrics collection.
    """

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    async def on_before_execute(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Called before a tool is executed. Can modify parameters.

        Args:
            tool_name: Name of the tool being called.
            params: Parameters being passed to the tool.

        Returns:
            Potentially modified parameters.
        """
        logger.debug(
            "[%s] Pre-tool hook for %s with params: %s",
            self.agent_name,
            tool_name,
            params,
        )
        return params

    async def on_after_execute(self, tool_name: str, result: Any) -> Any:
        """Called after a tool is executed. Can modify the result.

        Args:
            tool_name: Name of the tool that was called.
            result: The tool's output.

        Returns:
            Potentially modified result.
        """
        logger.debug(
            "[%s] Post-tool hook for %s, result type: %s",
            self.agent_name,
            tool_name,
            type(result).__name__,
        )
        return result

    async def on_error(self, tool_name: str, error: Exception) -> None:
        """Called when a tool raises an exception.

        Args:
            tool_name: Name of the tool that failed.
            error: The exception raised.
        """
        logger.error(
            "[%s] Tool %s failed: %s",
            self.agent_name,
            tool_name,
            error,
            exc_info=True,
        )
