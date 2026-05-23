"""Execution engine: runs `AgentTask`s against the registered agents.

Kept as a thin helper so the supervisor's responsibilities stay focused
on planning and aggregation. The engine owns the per-task bookkeeping —
recording timeline steps, updating task status, and surfacing errors —
and is called once per turn from the supervisor's dynamic routing loop.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from src.agents.base import BaseAgent
from src.models.requests import AgentRequest
from src.models.responses import AgentResponse
from src.models.state_models import (
    AgentTask,
    ExecutionStepKind,
    OrchestratorState,
    TaskStatus,
)
from src.observability.tracing import attributes as A
from src.observability.tracing import span

logger = logging.getLogger(__name__)

PreSubtaskHook = Callable[[int, AgentTask, OrchestratorState], Awaitable[bool]]


class ExecutionEngine:
    """Run individual tasks against the registered agents."""

    def __init__(self, agents: dict[str, BaseAgent]) -> None:
        self._agents = agents

    async def run_one_task(
        self,
        task: AgentTask,
        state: OrchestratorState,
        accumulated_context: dict,
    ) -> AgentResponse | None:
        """Execute a single `AgentTask`, updating `state` in place.

        Returns the produced `AgentResponse`, or `None` if the task failed
        (in which case the failure is recorded on the timeline).

        Wrapped in an ``engine.run_task`` span — the supervisor's view of
        dispatching one manager — so the manager's own ``agent.handle``
        span (and everything below it) nests cleanly under the dispatch.
        """
        async with span(
            A.SPAN_SUBTASK,
            attributes={
                A.AGENT_NAME: task.agent_name,
                A.AGENT_LAYER: A.layer_for(task.agent_name),
            },
        ):
            return await self._dispatch(task, state, accumulated_context)

    async def _dispatch(
        self,
        task: AgentTask,
        state: OrchestratorState,
        accumulated_context: dict,
    ) -> AgentResponse | None:
        agent = self._agents.get(task.agent_name)
        if agent is None:
            task.status = TaskStatus.FAILED
            task.error = f"Unknown agent: {task.agent_name}"
            state.add_step(
                agent_name=task.agent_name,
                kind=ExecutionStepKind.ERROR,
                message=task.error,
            )
            return None

        state.add_step(
            agent_name=task.agent_name,
            kind=ExecutionStepKind.SUBTASK_STARTED,
            message=f"{task.agent_name} starting: {task.description[:80]}",
        )
        state.current_tool = task.agent_name
        state.tool_path = f"RootSupervisorAgent.{task.agent_name}"
        task.status = TaskStatus.RUNNING

        try:
            response = await agent.handle(
                AgentRequest(
                    query=task.description,
                    context=dict(accumulated_context),
                    parent_agent="RootSupervisorAgent",
                )
            )
        except Exception as e:
            logger.exception("Subtask failed")
            task.status = TaskStatus.FAILED
            task.error = str(e)
            state.add_step(
                agent_name=task.agent_name,
                kind=ExecutionStepKind.ERROR,
                message=f"{task.agent_name} raised: {e}",
            )
            return None

        task.status = (
            TaskStatus.COMPLETED if response.success else TaskStatus.FAILED
        )
        task.result = response.model_dump(mode="json")
        task.error = response.error
        state.add_step(
            agent_name=task.agent_name,
            kind=ExecutionStepKind.SUBTASK_COMPLETE,
            message=f"{task.agent_name} completed",
            payload={
                "output_preview": response.content[:300],
                "trace": response.trace,
            },
        )
        return response
