"""SDK ``RunHooks`` that feed the Streamlit "Streaming Log".

Top-level run hooks fire for the supervisor and its manager-tool calls (they do
NOT propagate into ``as_tool`` sub-runs), which is exactly the manager-level
granularity the Streaming Log shows. Per-agent traces and token/cost are
captured elsewhere (``traced_as_tool`` + the bridge reading ``result`` usage).
"""

from __future__ import annotations

from typing import Any

from agents import RunHooks

from src.agents.factory import MANAGER_TOOLS

_SUPERVISOR = "RootSupervisorAgent"


class StreamingRunHooks(RunHooks):
    """Push orchestration timeline events into a ``StreamingCallbackHandler``."""

    def __init__(self, handler: Any) -> None:
        self._h = handler

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        if agent.name == _SUPERVISOR:
            self._h.push_orchestration_step(
                agent.name, "task_decomposition", "Routing the request"
            )

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        if getattr(tool, "name", None) in MANAGER_TOOLS:
            self._h.push_orchestration_step(
                tool.name, "subtask_started", f"{tool.name} starting"
            )

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: str) -> None:
        if getattr(tool, "name", None) in MANAGER_TOOLS:
            self._h.push_orchestration_step(
                tool.name, "subtask_complete", f"{tool.name} completed"
            )

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        if agent.name == _SUPERVISOR:
            self._h.push_orchestration_step(
                agent.name, "orchestration_complete", "Orchestration complete"
            )
