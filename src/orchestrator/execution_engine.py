"""Execution engine: runs an `ExecutionPlan` against a set of agents.

Kept as a separate module so the supervisor's responsibilities stay
focused on planning and aggregation. The engine owns the task loop,
records every transition on the `OrchestratorState`, and applies an
optional async pre-hook used by HITL to pause before each subtask.
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable

from src.agents.base import BaseAgent
from src.models.requests import AgentRequest
from src.models.responses import AgentResponse
from src.models.state_models import (
    AgentTask,
    ExecutionPlan,
    ExecutionStepKind,
    OrchestratorState,
    TaskStatus,
)

logger = logging.getLogger(__name__)

PreSubtaskHook = Callable[[int, AgentTask, OrchestratorState], Awaitable[bool]]


class ExecutionEngine:
    """Run tasks sequentially against the registered agents."""

    def __init__(self, agents: dict[str, BaseAgent]) -> None:
        self._agents = agents

    async def run(
        self,
        plan: ExecutionPlan,
        state: OrchestratorState,
        *,
        pre_subtask: PreSubtaskHook | None = None,
        start_index: int = 0,
    ) -> list[AgentResponse]:
        """Execute `plan.tasks[start_index:]` and return the agent responses.

        Args:
            plan: The execution plan to run.
            state: The orchestrator state, mutated in place with timeline
                entries.
            pre_subtask: Optional async hook called *before* each subtask.
                Returning False short-circuits execution — used by HITL to
                pause cleanly.
            start_index: Index to begin execution from (for HITL resume).

        Returns:
            The list of `AgentResponse` objects produced so far.
        """
        responses: list[AgentResponse] = []
        accumulated_context: dict = {}

        for idx in range(start_index, len(plan.tasks)):
            task = plan.tasks[idx]

            if pre_subtask is not None:
                proceed = await pre_subtask(idx, task, state)
                if not proceed:
                    return responses

            agent = self._agents.get(task.agent_name)
            if agent is None:
                task.status = TaskStatus.FAILED
                task.error = f"Unknown agent: {task.agent_name}"
                state.add_step(
                    agent_name=task.agent_name,
                    kind=ExecutionStepKind.ERROR,
                    message=task.error,
                )
                continue

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
                task.status = (
                    TaskStatus.COMPLETED if response.success else TaskStatus.FAILED
                )
                task.result = response.model_dump(mode="json")
                task.error = response.error
                responses.append(response)
                accumulated_context.update(response.data)
                state.add_step(
                    agent_name=task.agent_name,
                    kind=ExecutionStepKind.SUBTASK_COMPLETE,
                    message=f"{task.agent_name} completed",
                    payload={
                        "output_preview": response.content[:300],
                        # Surface the agent's reasoning trace on the timeline
                        # so the Streamlit state inspector can replay it.
                        "trace": response.trace,
                    },
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

        return responses
