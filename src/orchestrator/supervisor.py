"""RootSupervisorAgent — LLM-driven planner at the top of the hierarchy.

The supervisor uses the LLM to:

1. Decide which managers (ResearchManagerAgent, BuildManagerAgent, or
   both) to invoke for the given query.
2. Aggregate manager outputs into a single user-facing answer.

In mock mode the planner falls back to the deterministic `Router` (the
same intent-based rules tested in `tests/test_routing.py`), so the
orchestration shape stays correct offline. Real LLM mode replaces the
router decision with a structured JSON plan from the model.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agents.base import ReasoningAgent
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


_PLANNER_SYSTEM_PROMPT = (
    "You are the RootSupervisorAgent at the top of a 3-layer hierarchical "
    "multi-agent system. Two managers are available:\n"
    "- ResearchManagerAgent: coordinates retrieval + summarization.\n"
    "- BuildManagerAgent: coordinates code generation + review.\n\n"
    "Pick which managers to invoke based on user intent:\n"
    "- Reflective, philosophical, summarization, or rewriting requests → "
    "ResearchManagerAgent only.\n"
    "- Explicit code/implementation requests (build, implement, fastapi, "
    "redis, write code, etc.) → BuildManagerAgent.\n"
    "- Both, in Research→Build order, only when the user explicitly asks "
    "to combine the two."
)

_AGGREGATOR_SYSTEM_PROMPT = (
    "You synthesize multi-agent results into a clean, natural answer for "
    "the user. Combine outputs verbatim where useful; never mention "
    "internal agent names or routing decisions."
)


class RootSupervisorAgent:
    """Top-level supervisor — LLM-driven plan / route / aggregate."""

    name = "RootSupervisorAgent"
    tools = ["llm_planner", "router_fallback", "execution_engine", "llm_aggregator"]

    def __init__(self, model: str = "gpt-4.1-nano") -> None:
        self.model = model
        self.llm = get_llm_client(model)
        self.router = Router()
        self.managers: dict[str, ReasoningAgent] = {
            "ResearchManagerAgent": ResearchManagerAgent(model=model),
            "BuildManagerAgent": BuildManagerAgent(model=model),
        }
        self.engine = ExecutionEngine(self.managers)
        self.state: OrchestratorState | None = None

    # ----------------- Planning -----------------

    async def plan(self, user_query: str) -> ExecutionPlan:
        """Build the execution plan. LLM-driven, deterministic mock fallback."""
        if self.llm.enabled:
            try:
                return await self._llm_plan(user_query)
            except Exception as e:
                logger.warning("LLM planner failed, falling back to router: %s", e)
        return self._router_plan(user_query)

    async def _llm_plan(self, user_query: str) -> ExecutionPlan:
        """Ask the LLM directly to produce a manager list + reasoning."""
        from openai import AsyncOpenAI

        prompt = (
            f"User query: {user_query}\n\n"
            "Respond with JSON in this exact shape:\n"
            '{"reasoning": "<one-paragraph rationale>", '
            '"managers": ["ResearchManagerAgent" | "BuildManagerAgent", ...]}'
        )
        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        managers = [
            m for m in parsed.get("managers", []) if m in self.managers
        ]
        reasoning = parsed.get("reasoning", "")
        return self._build_plan(user_query, managers, reasoning)

    def _router_plan(self, user_query: str) -> ExecutionPlan:
        """Deterministic plan via the intent-based router (mock-mode fallback)."""
        decision = self.router.decide(user_query)
        managers: list[str] = []
        if decision.use_research:
            managers.append("ResearchManagerAgent")
        if decision.use_build:
            managers.append("BuildManagerAgent")
        reasoning = (
            f"[mock-llm planner via deterministic router] {decision.reasoning}"
        )
        return self._build_plan(user_query, managers, reasoning)

    def _build_plan(
        self, user_query: str, managers: list[str], reasoning: str
    ) -> ExecutionPlan:
        """Wrap a manager list into an `ExecutionPlan` with the right tool hints."""
        tools_by_manager = {
            "ResearchManagerAgent": ["RAGAgent", "SummarizerAgent"],
            "BuildManagerAgent": ["CodingAgent", "ReviewAgent"],
        }
        tasks: list[AgentTask] = []
        for i, manager_name in enumerate(managers):
            tasks.append(
                AgentTask(
                    agent_name=manager_name,
                    description=user_query,
                    tools_needed=tools_by_manager.get(manager_name, []),
                    depends_on=[0] if i > 0 else [],
                )
            )
        return ExecutionPlan(
            original_request=user_query,
            reasoning=reasoning,
            tasks=tasks,
        )

    # ----------------- Orchestration -----------------

    async def orchestrate(
        self,
        user_query: str,
        *,
        pre_subtask: PreSubtaskHook | None = None,
    ) -> tuple[OrchestratorState, AgentResponse]:
        """Run plan → execute → aggregate."""
        state = OrchestratorState(user_query=user_query, status="planning")
        self.state = state

        plan = await self.plan(user_query)
        state.plan = plan
        state.add_step(
            agent_name=self.name,
            kind=ExecutionStepKind.TASK_DECOMPOSITION,
            message=(
                f"Plan created with {len(plan.tasks)} task(s): "
                + ", ".join(t.agent_name for t in plan.tasks)
            ),
            payload={"reasoning": plan.reasoning},
        )

        state.status = "running"
        responses = await self.engine.run(plan, state, pre_subtask=pre_subtask)

        if len(responses) < len(plan.tasks):
            state.status = "paused"
            return state, AgentResponse(
                agent_name=self.name,
                content="Execution paused for review.",
                data={
                    "partial_responses": [r.model_dump(mode="json") for r in responses]
                },
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

        merged_data: dict[str, Any] = {}
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
        """Resume a paused orchestration from `start_index` (HITL path)."""
        assert state.plan is not None, "Cannot resume without a plan"
        self.state = state
        state.status = "running"

        responses = await self.engine.run(
            state.plan, state, pre_subtask=pre_subtask, start_index=start_index
        )

        unfinished = [
            t for t in state.plan.tasks
            if t.status.value not in {"completed", "failed"}
        ]
        if unfinished:
            state.status = "paused"
            return state, AgentResponse(
                agent_name=self.name,
                content="Execution paused again for review.",
                data={},
            )

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

        merged_data: dict[str, Any] = {}
        for r in all_responses:
            merged_data.update(r.data)
        return state, AgentResponse(
            agent_name=self.name,
            content=final_text,
            data=merged_data,
        )

    # ----------------- Aggregation -----------------

    async def _aggregate(
        self, user_query: str, responses: list[AgentResponse]
    ) -> str:
        """Combine manager responses into a single user-facing answer."""
        if not responses:
            return "No subtasks were executed."

        if self.llm.enabled:
            try:
                return await self._llm_aggregate(user_query, responses)
            except Exception as e:
                logger.warning("LLM aggregator failed, falling back: %s", e)

        parts: list[str] = []
        for r in responses:
            if r.agent_name == "ResearchManagerAgent":
                parts.append(f"Research: {r.content}")
            elif r.agent_name == "BuildManagerAgent":
                parts.append(f"Implementation:\n{r.content}")
            else:
                parts.append(r.content)
        return "\n\n".join(parts)

    async def _llm_aggregate(
        self, user_query: str, responses: list[AgentResponse]
    ) -> str:
        from openai import AsyncOpenAI

        joined = "\n\n".join(
            f"### {r.agent_name}\n{r.content}" for r in responses if r.content
        )
        prompt = (
            f"User query: {user_query}\n\n"
            f"Manager outputs:\n{joined}\n\n"
            "Write a concise, natural answer for the user. Do not mention "
            "agent names or routing decisions."
        )
        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _AGGREGATOR_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or ""
