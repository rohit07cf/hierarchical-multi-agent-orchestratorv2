"""RootSupervisorAgent — LLM-driven dynamic router at the top of the hierarchy.

The supervisor uses the LLM **turn-by-turn** to decide which manager to
invoke next. Each turn:

1. Pick the next manager (or stop) based on the user query plus the
   responses produced so far.
2. Execute that one manager.
3. Loop until the LLM decides no further work is needed (or a safety
   cap is hit).

A final LLM call aggregates the responses into a single user-facing
answer.

When `OPENAI_API_KEY` is unset, both the per-turn decision and the
aggregation fall back to deterministic rules (the intent-based `Router`
and a simple concatenation), so the orchestration shape stays correct
offline.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.agents.base import ReasoningAgent
from src.agents.build_manager import BuildManagerAgent
from src.agents.research_manager import ResearchManagerAgent
from src.llm.client import get_llm_client
from src.models.responses import AgentResponse
from src.models.state_models import (
    AgentTask,
    ExecutionPlan,
    ExecutionStepKind,
    OrchestratorState,
)
from src.observability.context import correlation_scope
from src.observability.metrics import registry as M
from src.observability.tracing import SpanKind
from src.observability.tracing import attributes as A
from src.observability.tracing import span
from src.orchestrator.execution_engine import ExecutionEngine, PreSubtaskHook
from src.orchestrator.router import Router

logger = logging.getLogger(__name__)


_PLANNER_SYSTEM_PROMPT = (
    "You are the RootSupervisorAgent at the top of a 3-layer hierarchical "
    "multi-agent system. On each turn you pick at most ONE manager to run "
    "next, or finish if the user's request has been satisfied. Two "
    "managers are available:\n"
    "- ResearchManagerAgent: coordinates retrieval + summarization.\n"
    "- BuildManagerAgent: coordinates code generation + review.\n\n"
    "Routing rules:\n"
    "- Reflective, philosophical, summarization, or rewriting requests → "
    "ResearchManagerAgent only.\n"
    "- Explicit code/implementation requests (build, implement, fastapi, "
    "redis, write code, etc.) → BuildManagerAgent.\n"
    "- Combined requests run Research first, then Build.\n"
    "- Never invoke the same manager twice. Once both have produced "
    "useful output, finish."
)

_AGGREGATOR_SYSTEM_PROMPT = (
    "You synthesize multi-agent results into a clean, natural answer for "
    "the user. Combine outputs verbatim where useful; never mention "
    "internal agent names or routing decisions."
)


# Default tool hints carried on each `AgentTask`, surfaced in the UI.
_TOOLS_BY_MANAGER: dict[str, list[str]] = {
    "ResearchManagerAgent": ["RAGAgent", "SummarizerAgent"],
    "BuildManagerAgent": ["CodingAgent", "ReviewAgent"],
}


class RootSupervisorAgent:
    """Top-level supervisor — LLM-driven dynamic plan / route / aggregate."""

    name = "RootSupervisorAgent"
    tools = ["llm_next_step", "router_fallback", "execution_engine", "llm_aggregator"]

    # Safety cap on the dynamic routing loop. Each turn calls one
    # manager, so this is also the maximum number of managers that can
    # be invoked for a single user request.
    MAX_TURNS = 6

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
        """Initial plan — the LLM's first manager pick (or empty if none).

        Used directly by the HITL bridge to show an upfront decomposition
        preview. Subsequent managers are added dynamically during
        `orchestrate()`.
        """
        manager_name, reasoning = await self._decide_next_manager(
            user_query, prior=[]
        )
        managers = [manager_name] if manager_name else []
        return self._build_plan(user_query, managers, reasoning)

    def _build_plan(
        self, user_query: str, managers: list[str], reasoning: str
    ) -> ExecutionPlan:
        """Wrap a manager list into an `ExecutionPlan` with the right tool hints."""
        tasks: list[AgentTask] = []
        for i, manager_name in enumerate(managers):
            tasks.append(
                AgentTask(
                    agent_name=manager_name,
                    description=user_query,
                    tools_needed=_TOOLS_BY_MANAGER.get(manager_name, []),
                    depends_on=[i - 1] if i > 0 else [],
                )
            )
        return ExecutionPlan(
            original_request=user_query,
            reasoning=reasoning,
            tasks=tasks,
        )

    # ----------------- Per-turn decision -----------------

    async def _decide_next_manager(
        self,
        user_query: str,
        prior: list[tuple[str, str]],
    ) -> tuple[str | None, str]:
        """Pick the next manager to invoke given the responses so far.

        Returns `(manager_name, reasoning)` where `manager_name` is None
        when the supervisor decides no further work is needed.

        This is the hottest debugging surface in the system: routing loops,
        the wrong manager firing, or an agent never being reached all start
        here. We span it, time it, and record which manager was picked and
        whether the decision came from the LLM or the deterministic router.
        """
        turn = len(prior)
        async with span(
            A.SPAN_ROUTING, attributes={A.ROUTE_TURN: turn}
        ) as sp:
            start = time.perf_counter()
            source = "llm"
            try:
                if self.llm.enabled:
                    try:
                        manager, reasoning = await self._llm_decide_next(
                            user_query, prior
                        )
                    except Exception as e:
                        logger.warning(
                            "LLM next-step decision failed, using router fallback: %s",
                            e,
                        )
                        source = "router_fallback"
                        manager, reasoning = self._router_decide_next(user_query, prior)
                else:
                    source = "router_fallback"
                    manager, reasoning = self._router_decide_next(user_query, prior)
            finally:
                M.ROUTING_LATENCY.labels(source).observe(time.perf_counter() - start)

            decision_label = manager or "finish"
            M.ROUTING_DECISIONS_TOTAL.labels(decision_label, source).inc()
            if manager:
                M.MANAGER_SELECTION.labels(manager).inc()
            sp.set_attribute(A.ROUTE_NEXT_MANAGER, decision_label)
            sp.set_attribute(A.ROUTE_DECISION_SOURCE, source)
            sp.set_attribute(A.ROUTE_REASONING, reasoning[:200])
            return manager, reasoning

    def _router_decide_next(
        self, user_query: str, prior: list[tuple[str, str]]
    ) -> tuple[str | None, str]:
        """Deterministic next-step decision via the intent-based router."""
        decision = self.router.decide(user_query)
        already_ran = {name for name, _ in prior}
        if decision.use_research and "ResearchManagerAgent" not in already_ran:
            return (
                "ResearchManagerAgent",
                f"[mock-llm dynamic router] {decision.reasoning}",
            )
        if decision.use_build and "BuildManagerAgent" not in already_ran:
            return (
                "BuildManagerAgent",
                f"[mock-llm dynamic router] {decision.reasoning}",
            )
        return None, "[mock-llm] All needed managers have run; finishing."

    async def _llm_decide_next(
        self, user_query: str, prior: list[tuple[str, str]]
    ) -> tuple[str | None, str]:
        """Ask the LLM which manager to call next (or null to finish)."""
        from openai import AsyncOpenAI

        if prior:
            history_block = "Managers that have already run:\n\n" + "\n\n".join(
                f"### {name}\n{content[:600]}" for name, content in prior
            )
        else:
            history_block = "No managers have run yet."

        available_block = "\n".join(f"- {name}" for name in self.managers.keys())

        prompt = (
            f"User query:\n{user_query}\n\n"
            f"{history_block}\n\n"
            f"Available managers:\n{available_block}\n\n"
            "Decide which manager to invoke next, or finish if no further "
            "work is needed. Respond with JSON in this exact shape:\n"
            '{"reasoning": "<one-line rationale>", '
            '"next_manager": "<manager name>" | null}'
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
        next_manager = parsed.get("next_manager")
        if next_manager not in self.managers:
            next_manager = None
        return next_manager, parsed.get("reasoning", "")

    # ----------------- Orchestration -----------------

    async def orchestrate(
        self,
        user_query: str,
        *,
        pre_subtask: PreSubtaskHook | None = None,
    ) -> tuple[OrchestratorState, AgentResponse]:
        """Instrumented entry point — the root span for an entire run.

        Owns the request-scoped correlation context (``request_id`` ==
        ``state_id``), the in-flight gauge, the terminal-status counters,
        and the end-to-end duration histogram. The actual routing loop
        lives in ``_run_orchestration`` so every agent/tool/LLM span nests
        under this one and shares the same trace.
        """
        state = OrchestratorState(user_query=user_query, status="planning")
        self.state = state

        M.ACTIVE_ORCHESTRATIONS.inc()
        start = time.perf_counter()
        with correlation_scope(request_id=state.state_id):
            async with span(
                A.SPAN_ORCHESTRATION,
                kind=SpanKind.SERVER,
                attributes={
                    A.ORCH_QUERY_PREVIEW: user_query[:120],
                    A.ORCH_QUERY_LEN: len(user_query),
                },
            ) as sp:
                try:
                    result_state, response = await self._run_orchestration(
                        state, user_query, pre_subtask=pre_subtask
                    )
                except Exception as exc:
                    M.ORCHESTRATION_FAILURES.labels(type(exc).__name__).inc()
                    raise
                finally:
                    M.ACTIVE_ORCHESTRATIONS.dec()
                    status = state.status
                    M.ORCHESTRATION_TOTAL.labels(status).inc()
                    M.ORCHESTRATION_DURATION.labels(status).observe(
                        time.perf_counter() - start
                    )
                turns = sum(
                    1
                    for s in state.steps
                    if s.kind == ExecutionStepKind.SUBTASK_COMPLETE
                )
                M.ORCHESTRATION_TURNS.observe(turns)
                sp.set_attribute(A.ORCH_STATUS, state.status)
                sp.set_attribute(A.ORCH_TURNS, turns)
                if state.plan:
                    sp.set_attribute(
                        A.ORCH_MANAGERS,
                        [t.agent_name for t in state.plan.tasks],
                    )
                return result_state, response

    async def _run_orchestration(
        self,
        state: OrchestratorState,
        user_query: str,
        *,
        pre_subtask: PreSubtaskHook | None = None,
    ) -> tuple[OrchestratorState, AgentResponse]:
        """Run the dynamic routing loop and aggregate the final answer."""
        plan = await self.plan(user_query)
        state.plan = plan
        state.add_step(
            agent_name=self.name,
            kind=ExecutionStepKind.TASK_DECOMPOSITION,
            message=(
                "Initial plan: "
                + (
                    ", ".join(t.agent_name for t in plan.tasks)
                    if plan.tasks
                    else "(empty — no managers selected yet)"
                )
            ),
            payload={"reasoning": plan.reasoning},
        )

        state.status = "running"
        responses: list[AgentResponse] = []
        accumulated_context: dict[str, Any] = {}

        for idx in range(self.MAX_TURNS):
            if idx >= len(plan.tasks):
                manager_name, reasoning = await self._decide_next_manager(
                    user_query,
                    prior=[(r.agent_name, r.content) for r in responses],
                )
                if manager_name is None:
                    break
                plan.tasks.append(
                    AgentTask(
                        agent_name=manager_name,
                        description=user_query,
                        tools_needed=_TOOLS_BY_MANAGER.get(manager_name, []),
                        depends_on=[idx - 1] if idx > 0 else [],
                    )
                )
                state.add_step(
                    agent_name=self.name,
                    kind=ExecutionStepKind.TASK_DECOMPOSITION,
                    message=f"Dynamic routing: next agent is {manager_name}",
                    payload={"reasoning": reasoning},
                )

            task = plan.tasks[idx]

            if pre_subtask is not None:
                proceed = await pre_subtask(idx, task, state)
                if not proceed:
                    state.status = "paused"
                    M.HITL_PAUSED.labels("before_subtask").inc()
                    return state, AgentResponse(
                        agent_name=self.name,
                        content="Execution paused for review.",
                        data={
                            "partial_responses": [
                                r.model_dump(mode="json") for r in responses
                            ]
                        },
                    )

            response = await self.engine.run_one_task(
                task, state, accumulated_context
            )
            if response is None:
                # Task errored — stop here; the engine has already
                # recorded the error on the timeline.
                break
            responses.append(response)
            accumulated_context.update(response.data)

        if not responses:
            state.final_answer = "No managers were invoked."
            state.status = "completed"
            return state, AgentResponse(
                agent_name=self.name, content=state.final_answer
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
        # Re-bind the original run's correlation id so the resumed spans/logs
        # join the same logical request as the pre-pause portion.
        M.HITL_RESUMES.labels("approve").inc()
        with correlation_scope(request_id=state.state_id):
            return await self._resume_impl(
                state, start_index, pre_subtask=pre_subtask
            )

    async def _resume_impl(
        self,
        state: OrchestratorState,
        start_index: int,
        *,
        pre_subtask: PreSubtaskHook | None = None,
    ) -> tuple[OrchestratorState, AgentResponse]:
        assert state.plan is not None
        state.status = "running"

        plan = state.plan
        responses: list[AgentResponse] = []
        accumulated_context: dict[str, Any] = {}

        # Replay previously-completed tasks into the response list so the
        # dynamic loop has the full history when deciding what to do next.
        for prior_task in plan.tasks[:start_index]:
            if prior_task.is_success and isinstance(prior_task.result, dict):
                prior_response = AgentResponse.model_validate(prior_task.result)
                responses.append(prior_response)
                accumulated_context.update(prior_response.data)

        for idx in range(start_index, start_index + self.MAX_TURNS):
            if idx >= len(plan.tasks):
                manager_name, reasoning = await self._decide_next_manager(
                    state.user_query,
                    prior=[(r.agent_name, r.content) for r in responses],
                )
                if manager_name is None:
                    break
                plan.tasks.append(
                    AgentTask(
                        agent_name=manager_name,
                        description=state.user_query,
                        tools_needed=_TOOLS_BY_MANAGER.get(manager_name, []),
                        depends_on=[idx - 1] if idx > 0 else [],
                    )
                )
                state.add_step(
                    agent_name=self.name,
                    kind=ExecutionStepKind.TASK_DECOMPOSITION,
                    message=f"Dynamic routing: next agent is {manager_name}",
                    payload={"reasoning": reasoning},
                )

            task = plan.tasks[idx]

            if pre_subtask is not None:
                proceed = await pre_subtask(idx, task, state)
                if not proceed:
                    state.status = "paused"
                    M.HITL_PAUSED.labels("before_subtask").inc()
                    return state, AgentResponse(
                        agent_name=self.name,
                        content="Execution paused again for review.",
                        data={},
                    )

            response = await self.engine.run_one_task(
                task, state, accumulated_context
            )
            if response is None:
                break
            responses.append(response)
            accumulated_context.update(response.data)

        final_text = await self._aggregate(state.user_query, responses)
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

    # ----------------- Aggregation -----------------

    async def _aggregate(
        self, user_query: str, responses: list[AgentResponse]
    ) -> str:
        """Combine manager responses into a single user-facing answer."""
        if not responses:
            return "No subtasks were executed."

        async with span(
            A.SPAN_AGGREGATION,
            attributes={"hmao.aggregate.inputs": len(responses)},
        ):
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
