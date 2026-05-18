"""Backward-compatible SupervisorAgent — bridge to the new RootSupervisorAgent.

The Streamlit UI is built around the legacy `SupervisorAgent` interface:

    SupervisorAgent.orchestrate(...)
    SupervisorAgent.orchestrate_manual(...)
    SupervisorAgent.resume_orchestration(...)
    SupervisorAgent.state    -> AgentState
    SupervisorAgent._child_agents

This module keeps that surface intact while delegating the real work to
the refactored 3-layer hierarchy in `src/`. The bridge translates between
the legacy `SupervisorOutput` / `AgentState` models and the new
`OrchestratorState` / `AgentResponse` models so the UI does not need to
change.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from models.agent_state import AgentState, HITLCheckpointType
from models.supervisor_output import (
    PlannedSubtask,
    SubtaskResult,
    SubtaskStatus,
    SupervisorOutput,
    TaskDecomposition,
)
from orchestration.hitl_manager import HITLManager
from src.models.state_models import OrchestratorState, TaskStatus
from src.models.responses import AgentResponse as NewAgentResponse
from src.orchestrator.supervisor import RootSupervisorAgent

logger = logging.getLogger(__name__)


def _plan_to_decomposition(state: OrchestratorState) -> TaskDecomposition:
    """Convert the new `ExecutionPlan` into the legacy `TaskDecomposition`."""
    plan = state.plan
    assert plan is not None
    return TaskDecomposition(
        original_request=plan.original_request,
        reasoning=plan.reasoning,
        subtasks=[
            PlannedSubtask(
                agent_name=t.agent_name,
                description=t.description,
                tools_needed=t.tools_needed,
                depends_on=t.depends_on,
            )
            for t in plan.tasks
        ],
    )


def _task_result_to_subtask(task) -> SubtaskResult:
    """Convert a new `AgentTask` (post-execution) to a legacy `SubtaskResult`.

    Carries the agent's reasoning trace (and any worker traces) through
    `tool_calls` so the Streamlit subtask panel can render them.
    """
    status_map = {
        TaskStatus.COMPLETED: SubtaskStatus.COMPLETED,
        TaskStatus.FAILED: SubtaskStatus.FAILED,
        TaskStatus.RUNNING: SubtaskStatus.RUNNING,
        TaskStatus.PENDING: SubtaskStatus.PENDING,
    }
    result_payload: Any = None
    tool_calls: list[dict[str, Any]] = []
    if task.result and isinstance(task.result, dict):
        result_payload = task.result.get("content") or task.result
        trace = task.result.get("trace")
        if isinstance(trace, dict):
            tool_calls.append({"agent_trace": trace})
        worker_traces = (task.result.get("data") or {}).get("worker_traces") or []
        for wt in worker_traces:
            tool_calls.append({"agent_trace": wt})
    return SubtaskResult(
        agent_name=task.agent_name,
        subtask=task.description,
        result=result_payload,
        status=status_map[task.status],
        error=task.error,
        tool_calls=tool_calls,
    )


def _sync_legacy_state(
    legacy: AgentState,
    new: OrchestratorState,
    streaming: Any | None = None,
) -> None:
    """Mirror the new state's timeline onto the legacy `AgentState`.

    The legacy state is what the Streamlit "State Inspector" reads from,
    so every execution step recorded on `OrchestratorState` is replayed
    onto `AgentState` (idempotent — only new steps are appended). When a
    streaming handler is supplied, the same steps are mirrored into its
    buffer so the "Streaming Log" panel updates in lock-step.
    """
    legacy.tool = new.current_tool
    legacy.tool_path = new.tool_path
    legacy.current_inputs = {"user_input": new.user_query}

    existing = len(legacy.intermediate_steps)
    for step in new.steps[existing:]:
        legacy.add_step(
            agent_name=step.agent_name,
            action=step.kind.value,
            action_input=step.payload,
            observation=step.message,
        )
        if streaming is not None:
            streaming.push_orchestration_step(
                agent_name=step.agent_name,
                kind=step.kind.value,
                message=step.message,
            )


class SupervisorAgent:
    """Legacy-shaped supervisor that delegates to `RootSupervisorAgent`.

    The class keeps the original public API (orchestrate / orchestrate_manual
    / resume_orchestration / state / _child_agents) so callers — most
    notably the Streamlit app — do not need to change. Internally every
    call routes through the new hierarchical orchestrator in `src/`.
    """

    # Backward-compatible map of the *old* worker agent names. The new
    # system does not use these; the field is preserved only so any
    # caller that introspected `CHILD_AGENT_MAP` continues to resolve.
    CHILD_AGENT_MAP: dict[str, str] = {
        "SimpleAgent": "agent_defs.simple_agent.SimpleAgentDef",
        "MathAgent": "agent_defs.math_agent.MathAgentDef",
        "EchoAgent": "agent_defs.echo_agent.EchoAgentDef",
        "ClassifierAgent": "agent_defs.classifier_agent.ClassifierAgentDef",
    }

    def __init__(self, model: str = "gpt-4.1-nano") -> None:
        self.name = "RootSupervisorAgent"
        self.model = model
        self._state = AgentState(tool_path=self.name)
        self._root = RootSupervisorAgent(model=model)
        self.streaming_handler: Any | None = None
        # Expose the live hierarchy (managers + workers) plus the legacy
        # worker agents under their original names for backward compat.
        self._child_agents: dict[str, Any] = {
            "ResearchManagerAgent": self._root.managers["ResearchManagerAgent"],
            "BuildManagerAgent": self._root.managers["BuildManagerAgent"],
            "RAGAgent": self._root.managers["ResearchManagerAgent"].rag,
            "SummarizerAgent": self._root.managers["ResearchManagerAgent"].summarizer,
            "CodingAgent": self._root.managers["BuildManagerAgent"].coder,
            "ReviewAgent": self._root.managers["BuildManagerAgent"].reviewer,
        }

    @property
    def child_agents(self) -> dict[str, Any]:
        """Live registry of manager and worker agents (read-only)."""
        return self._child_agents

    # Legacy state surface — what Streamlit reads.
    @property
    def state(self) -> AgentState:
        """Current agent execution state (for state inspector)."""
        return self._state

    def reset_state(self) -> None:
        """Reset agent state for a fresh execution."""
        self._state = AgentState(tool_path=self.name)

    async def orchestrate(
        self,
        user_input: str,
        hitl_manager: HITLManager | None = None,
        enable_hitl: bool = False,
    ) -> SupervisorOutput | None:
        """Run the full pipeline, optionally pausing at HITL checkpoints.

        Behaviour matches the original implementation:
        - When `enable_hitl=False`, runs end-to-end and returns a
          `SupervisorOutput`.
        - When `enable_hitl=True`, pauses after planning and again before
          each subtask; returns `None` while paused so the caller can read
          paused state from `hitl_manager`.
        """
        self.reset_state()
        self._state.current_inputs = {"user_input": user_input}
        logger.info("Supervisor starting orchestration: %s", user_input[:100])

        try:
            # In HITL mode we materialize the plan up-front (without
            # running it) so we can pause for the user to review the
            # decomposition. Outside HITL the supervisor will await its
            # own LLM-driven plan() during orchestrate().
            plan = await self._root.plan(user_input)

            if enable_hitl and hitl_manager:
                # HITL: we have to materialize the decomposition step on the
                # legacy state up-front because we're about to pause before
                # the new supervisor ever runs.
                self._state.add_step(
                    agent_name=self.name,
                    action="task_decomposition",
                    action_input={"user_input": user_input},
                    observation=json.dumps(
                        {
                            "reasoning": plan.reasoning,
                            "tasks": [t.agent_name for t in plan.tasks],
                        }
                    ),
                )
                if self.streaming_handler is not None:
                    self.streaming_handler.push_orchestration_step(
                        agent_name=self.name,
                        kind="task_decomposition",
                        message=f"Plan created with {len(plan.tasks)} task(s)",
                    )
                # Stash everything needed to resume on the legacy AgentState.
                self._state.pause(
                    checkpoint_type=HITLCheckpointType.DECOMPOSITION,
                    pending_data={
                        "decomposition": _plan_to_decomposition(
                            OrchestratorState(user_query=user_input, plan=plan)
                        ).model_dump(mode="json"),
                        "subtask_index": 0,
                    },
                )
                hitl_manager.capture_state(self._state)
                return None

            state, response = await self._root.orchestrate(user_input)
            _sync_legacy_state(self._state, state, self.streaming_handler)
            return self._build_output(state, response)

        except Exception as e:
            logger.error("Orchestration failed", exc_info=True)
            return SupervisorOutput(
                final_answer=f"Orchestration failed: {e}",
                subtasks=[
                    SubtaskResult(
                        agent_name=self.name,
                        subtask=user_input,
                        result=None,
                        status=SubtaskStatus.FAILED,
                        error=str(e),
                    )
                ],
            )

    async def orchestrate_manual(self, user_input: str) -> SupervisorOutput:
        """Run orchestration to completion without HITL pauses."""
        self.reset_state()
        result = await self.orchestrate(user_input)
        assert result is not None, "manual orchestration should always return a result"
        return result

    async def resume_orchestration(
        self,
        hitl_manager: HITLManager,
        state_id: str,
    ) -> SupervisorOutput | None:
        """Resume a paused HITL orchestration after the user has acted.

        Reconstructs the execution plan from the paused state, executes the
        next pending subtask, and re-pauses for the following one. When
        the final subtask completes, aggregates and returns the
        SupervisorOutput.
        """
        state = hitl_manager.get_state(state_id)
        if state is None:
            return SupervisorOutput(
                final_answer=f"Error: State {state_id} not found",
                subtasks=[],
            )
        self._state = state

        last_action = state.hitl_actions[-1] if state.hitl_actions else None
        if last_action and last_action.action.value == "CANCEL":
            return SupervisorOutput(
                final_answer="Orchestration cancelled by user.",
                subtasks=[],
            )

        user_input = state.current_inputs.get("user_input", "")
        if last_action and last_action.action.value == "REVISE" and last_action.input:
            user_input = last_action.input
            state.current_inputs["user_input"] = user_input

        pending = state.pending_data
        decomposition_data = pending.get("decomposition", {})
        decomposition = TaskDecomposition.model_validate(decomposition_data)

        # Rebuild a fresh plan from the (possibly revised) input. The new
        # supervisor's plan() is async (LLM-driven, with deterministic
        # router fallback) so we await it here.
        plan = await self._root.plan(user_input)

        # Pause-before-each-subtask semantics:
        # - DECOMPOSITION approval → re-pause at TOOL_EXECUTION for task 0
        #   without executing anything (the user has only approved the plan).
        # - TOOL_EXECUTION approval → execute task at `subtask_index`, then
        #   either re-pause for the next task or aggregate if done.
        if state.checkpoint_type == HITLCheckpointType.DECOMPOSITION:
            return self._repause_before_task(
                hitl_manager,
                plan=plan,
                decomposition=decomposition,
                subtask_index=0,
                prior_subtasks=[],
            )

        subtask_index = pending.get("subtask_index", 0)
        prior_subtasks = [
            SubtaskResult.model_validate(r)
            for r in pending.get("completed_results", [])
        ]

        try:
            subtask_results = list(prior_subtasks)
            current = plan.tasks[subtask_index]
            response = await self._root.managers[current.agent_name].handle(
                _make_request(current, _accumulated(subtask_results))
            )
            subtask_results.append(
                SubtaskResult(
                    agent_name=current.agent_name,
                    subtask=current.description,
                    result=response.content,
                    status=SubtaskStatus.COMPLETED,
                )
            )
            self._state.add_step(
                agent_name=current.agent_name,
                action=f"subtask_complete:{current.description[:60]}",
                observation=response.content[:300],
            )
            if self.streaming_handler is not None:
                self.streaming_handler.push_orchestration_step(
                    agent_name=current.agent_name,
                    kind="subtask_complete",
                    message=f"{current.agent_name} completed",
                )
        except Exception as e:
            logger.error("Resume failed", exc_info=True)
            return SupervisorOutput(
                final_answer=f"Resume failed: {e}",
                subtasks=[
                    SubtaskResult(
                        agent_name=plan.tasks[subtask_index].agent_name,
                        subtask=user_input,
                        result=None,
                        status=SubtaskStatus.FAILED,
                        error=str(e),
                    )
                ],
            )

        next_index = subtask_index + 1
        if next_index >= len(plan.tasks):
            # All done — synthesize the final answer.
            responses = [
                NewAgentResponse(
                    agent_name=s.agent_name,
                    content=str(s.result) if s.result is not None else "",
                    data={},
                )
                for s in subtask_results
            ]
            final_answer = await self._root._aggregate(user_input, responses)
            self._state.add_step(
                agent_name=self.name,
                action="orchestration_complete",
                observation=final_answer,
            )
            return SupervisorOutput(
                final_answer=final_answer,
                subtasks=subtask_results,
                decomposition=decomposition,
            )

        return self._repause_before_task(
            hitl_manager,
            plan=plan,
            decomposition=decomposition,
            subtask_index=next_index,
            prior_subtasks=subtask_results,
        )

    def _repause_before_task(
        self,
        hitl_manager: HITLManager,
        *,
        plan,
        decomposition: TaskDecomposition,
        subtask_index: int,
        prior_subtasks: list[SubtaskResult],
    ) -> None:
        """Pause before the task at `subtask_index` and persist HITL state."""
        next_task = plan.tasks[subtask_index]
        self._state.pause(
            checkpoint_type=HITLCheckpointType.TOOL_EXECUTION,
            pending_data={
                "decomposition": decomposition.model_dump(mode="json"),
                "subtask_index": subtask_index,
                "completed_results": [r.model_dump(mode="json") for r in prior_subtasks],
                "pending_tool": {
                    "agent_name": next_task.agent_name,
                    "description": next_task.description,
                    "tools_needed": next_task.tools_needed,
                },
            },
        )
        self._state.tool = next_task.agent_name
        self._state.tool_path = f"RootSupervisorAgent.{next_task.agent_name}"
        hitl_manager.capture_state(self._state)
        return None

    def _build_output(
        self,
        state: OrchestratorState,
        response: NewAgentResponse,
    ) -> SupervisorOutput:
        """Project a finished `OrchestratorState` into the legacy output type."""
        assert state.plan is not None
        decomposition = _plan_to_decomposition(state)
        subtasks = [_task_result_to_subtask(t) for t in state.plan.tasks]
        return SupervisorOutput(
            final_answer=state.final_answer or response.content,
            subtasks=subtasks,
            decomposition=decomposition,
        )


def _make_request(task, context: dict[str, Any]):
    """Build a new `AgentRequest` for a single task during HITL resume."""
    from src.models.requests import AgentRequest

    return AgentRequest(
        query=task.description,
        context=context,
        parent_agent="RootSupervisorAgent",
    )


def _accumulated(prior: list[SubtaskResult]) -> dict[str, Any]:
    """Build the context dict downstream tasks see during HITL resume."""
    return {
        "previous_results": [
            {"agent": r.agent_name, "result": r.result} for r in prior
        ],
    }
