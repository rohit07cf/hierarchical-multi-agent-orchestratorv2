"""Base agent class providing the Template Method pattern for agent execution."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from agents import Agent, Runner, function_tool

from models.agent_state import AgentState
from models.tool_models import ToolResult

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class implementing the Template Method pattern for agents.

    Subclasses define their tools, prompts, and specific behavior while
    inheriting the standard execution pipeline including state tracking,
    HITL checkpoints, and streaming support.
    """

    def __init__(self, name: str, model: str = "gpt-4.1-nano") -> None:
        self.name = name
        self.model = model
        self._agent: Agent | None = None
        self._tools: list[Any] = []
        self._state = AgentState(tool_path=name)

    @property
    def agent(self) -> Agent:
        """Lazily build and return the underlying OpenAI Agent instance."""
        if self._agent is None:
            self._agent = self._build_agent()
        return self._agent

    @property
    def state(self) -> AgentState:
        """Current agent execution state."""
        return self._state

    @property
    def tool_names(self) -> list[str]:
        """List of tool names available to this agent."""
        return [t.__name__ if hasattr(t, "__name__") else str(t) for t in self._tools]

    def reset_state(self) -> None:
        """Reset agent state for a fresh execution."""
        self._state = AgentState(tool_path=self.name)

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return the system prompt for this agent. Subclasses must implement."""

    @abstractmethod
    def _register_tools(self) -> list[Any]:
        """Return the list of tools for this agent. Subclasses must implement."""

    def _build_agent(self) -> Agent:
        """Build the OpenAI Agent with configured tools and prompts (Template Method)."""
        self._tools = self._register_tools()
        return Agent(
            name=self.name,
            instructions=self._get_system_prompt(),
            tools=self._tools,
            model=self.model,
        )

    async def run(self, user_input: str, context: Any = None) -> str:
        """Execute the agent with the given input using Runner.run.

        Args:
            user_input: The user's message or task.
            context: Optional context for dependency injection.

        Returns:
            The agent's final output string.
        """
        self._state.current_inputs = {"user_input": user_input}
        logger.info("Agent %s starting execution", self.name)

        try:
            result = await Runner.run(
                self.agent,
                input=user_input,
                context=context,
            )
            output = result.final_output
            self._state.add_step(
                agent_name=self.name,
                action="final_output",
                observation=str(output),
            )
            logger.info("Agent %s completed execution", self.name)
            return str(output)

        except Exception as e:
            error_msg = f"Agent {self.name} execution failed: {e}"
            logger.error(error_msg)
            self._state.add_step(
                agent_name=self.name,
                action="error",
                observation=error_msg,
            )
            return ToolResult.fail(error=error_msg, tool_name=self.name).model_dump_json()

    def run_sync(self, user_input: str, context: Any = None) -> str:
        """Synchronous wrapper for agent execution.

        Args:
            user_input: The user's message or task.
            context: Optional context for dependency injection.

        Returns:
            The agent's final output string.
        """
        self._state.current_inputs = {"user_input": user_input}
        logger.info("Agent %s starting sync execution", self.name)

        try:
            result = Runner.run_sync(
                self.agent,
                input=user_input,
                context=context,
            )
            output = result.final_output
            self._state.add_step(
                agent_name=self.name,
                action="final_output",
                observation=str(output),
            )
            return str(output)

        except Exception as e:
            error_msg = f"Agent {self.name} execution failed: {e}"
            logger.error(error_msg)
            return ToolResult.fail(error=error_msg, tool_name=self.name).model_dump_json()
