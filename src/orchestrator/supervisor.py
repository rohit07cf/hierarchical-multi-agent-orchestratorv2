"""RootSupervisorAgent — top of the 3-layer hierarchy."""

from __future__ import annotations

import logging

from src.agents.base import BaseAgent
from src.agents.build_manager import BuildManagerAgent
from src.agents.research_manager import ResearchManagerAgent
from src.llm.client import get_llm_client
from src.models.requests import AgentRequest
from src.models.responses import AgentResponse
from src.models.state_models import (
    AgentTask,
    ExecutionPlan,
    ExecutionStepKind,
    OrchestratorState,
)
from src.orchestrator.execution_engine import ExecutionEngine, PreSubtaskHook
from src.orchestrator.router import Router

logger = logging.getLogger(__name__)


class RootSupervisorAgent:
    """Top-level supervisor.

    Owns three responsibilities:
    1. Plan — decompose the query into tasks via the `Router`.
    2. Route — invoke the chosen manager agents through the
       `ExecutionEngine`.
    3. Aggregate — synthesize a single user-facing response.
    """

    name = "RootSupervisorAgent"
    tools = ["router", "execution_engine", "llm_aggregator"]

    def __init__(self, model: str = "gpt-4.1-nano") -> None:
        self.model = model
        self.llm = get_llm_client(model)
        self.router = Router()
        self.managers: dict[str, BaseAgent] = {
            "ResearchManagerAgent": ResearchManagerAgent(model=model),
            "BuildManagerAgent": BuildManagerAgent(model=model),
        }
        self.engine = ExecutionEngine(self.managers)
        self.state: OrchestratorState | None = None

    def plan(self, user_query: str) -> ExecutionPlan:
        """Decompose `user_query` into an execution plan.

        The plan is deterministic and built from the router decision so it
        is reproducible across runs. Managers always come before workers
        in the same chain since the build phase may consume context from
        the research phase.
        """
        decision = self.router.decide(user_query)
        tasks: list[AgentTask] = []
        if decision.use_research:
            tasks.append(
                AgentTask(
                    agent_name="ResearchManagerAgent",
                    description=user_query,
                    tools_needed=["RAGAgent", "SummarizerAgent"],
                )
            )
        if decision.use_build:
            tasks.append(
                AgentTask(
                    agent_name="BuildManagerAgent",
                    description=user_query,
                    tools_needed=["CodingAgent", "ReviewAgent"],
                    depends_on=[0] if decision.use_research else [],
                )
            )
        return ExecutionPlan(
            original_request=user_query,
            reasoning=f"Routing decision: {decision.reasoning}.",
            tasks=tasks,
        )

    async def orchestrate(
        self,
        user_query: str,
        *,
        pre_subtask: PreSubtaskHook | None = None,
    ) -> tuple[OrchestratorState, AgentResponse]:
        """Run the full plan-execute-aggregate pipeline.

        Args:
            user_query: The user's request.
            pre_subtask: Optional async hook (used by HITL) invoked
                before each subtask. Returning False pauses execution.

        Returns:
            A tuple of `(state, response)`. When execution pauses for HITL
            the response will reflect partial progress and the state's
            timeline will end at the pause point.
        """
        state = OrchestratorState(user_query=user_query, status="planning")
        self.state = state

        plan = self.plan(user_query)
        state.plan = plan
        state.add_step(
            agent_name=self.name,
            kind=ExecutionStepKind.TASK_DECOMPOSITION,
            message=f"Plan created with {len(plan.tasks)} task(s): "
            + ", ".join(t.agent_name for t in plan.tasks),
            payload={"reasoning": plan.reasoning},
        )

        state.status = "running"
        responses = await self.engine.run(plan, state, pre_subtask=pre_subtask)

        # If the engine paused before completion, surface that and return.
        if len(responses) < len(plan.tasks):
            state.status = "paused"
            return state, AgentResponse(
                agent_name=self.name,
                content="Execution paused for review.",
                data={"partial_responses": [r.model_dump(mode="json") for r in responses]},
            )

        final_text = await self._aggregate(user_query, responses)
        state.final_answer = final_text
        state.status = "completed"
        state.add_step(
            agent_name=self.name,
            kind=ExecutionStepKind.ORCHESTRATION_COMPLETE,
            message="Orchestration complete",
            payload={"final_answer_preview": final_text[:200]},
        )

        merged_data: dict = {}
        for r in responses:
            merged_data.update(r.data)
        return state, AgentResponse(
            agent_name=self.name,
            content=final_text,
            data=merged_data,
        )

    async def resume(
        self,
        state: OrchestratorState,
        start_index: int,
        *,
        pre_subtask: PreSubtaskHook | None = None,
    ) -> tuple[OrchestratorState, AgentResponse]:
        """Resume a paused orchestration from `start_index`.

        Used by the HITL flow after the user approves a paused subtask.
        """
        assert state.plan is not None, "Cannot resume without a plan"
        self.state = state
        state.status = "running"

        responses = await self.engine.run(
            state.plan, state, pre_subtask=pre_subtask, start_index=start_index
        )

        completed_count = sum(1 for t in state.plan.tasks if t.is_success)
        if completed_count < len(state.plan.tasks) and not all(
            t.status.value in {"completed", "failed"} for t in state.plan.tasks
        ):
            state.status = "paused"
            return state, AgentResponse(
                agent_name=self.name,
                content="Execution paused again for review.",
                data={},
            )

        # Aggregate over ALL completed task results, not just this resume batch.
        all_responses: list[AgentResponse] = []
        for task in state.plan.tasks:
            if task.is_success and isinstance(task.result, dict):
                all_responses.append(AgentResponse.model_validate(task.result))

        final_text = await self._aggregate(state.user_query, all_responses)
        state.final_answer = final_text
        state.status = "completed"
        state.add_step(
            agent_name=self.name,
            kind=ExecutionStepKind.ORCHESTRATION_COMPLETE,
            message="Orchestration complete",
            payload={"final_answer_preview": final_text[:200]},
        )

        merged_data: dict = {}
        for r in all_responses:
            merged_data.update(r.data)
        return state, AgentResponse(
            agent_name=self.name,
            content=final_text,
            data=merged_data,
        )

    async def _aggregate(self, user_query: str, responses: list[AgentResponse]) -> str:
        """Combine manager responses into a single user-facing answer."""
        if not responses:
            return "No subtasks were executed."

        if self.llm.enabled:
            joined = "\n\n".join(
                f"### {r.agent_name}\n{r.content}" for r in responses if r.content
            )
            prompt = (
                f"User query: {user_query}\n\n"
                f"Manager outputs:\n{joined}\n\n"
                "Write a concise, natural answer for the user, combining "
                "the outputs without mentioning the agent names."
            )
            return await self.llm.complete(
                prompt,
                system="You synthesize multi-agent results into a clean answer.",
            )

        parts: list[str] = []
        for r in responses:
            if r.agent_name == "ResearchManagerAgent":
                parts.append(f"Research: {r.content}")
            elif r.agent_name == "BuildManagerAgent":
                parts.append(f"Implementation:\n{r.content}")
            else:
                parts.append(r.content)
        return "\n\n".join(parts)
