"""SupervisorAgent — the bridge the Streamlit UI is built around.

Keeps the legacy public surface (`orchestrate` / `orchestrate_manual` /
`resume_orchestration` / `state` / `child_agents`) while delegating the real
work to the **OpenAI Agents SDK** graph (`src/agents/factory.build_supervisor`)
powered by Claude via LiteLLM. A finished `RunResult` is projected back into the
legacy `SupervisorOutput` (`src/agents/result_mapper`) so the UI is unchanged.

HITL: when `enable_hitl=True`, the manager tools require approval. A paused run
surfaces SDK `interruptions`; we persist the SDK `RunState` JSON inside a legacy
`AgentState.pending_data` blob so the existing HITL controls keep working, and
resume by re-running the serialized state with approve/reject applied.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agents import Runner, RunState

from models.agent_state import AgentState, HITLActionType, HITLCheckpointType
from models.supervisor_output import SubtaskResult, SubtaskStatus, SupervisorOutput
from orchestration.hitl_manager import HITLManager
from src.agents.factory import (
    DEFAULT_MODEL,
    MANAGER_WORKERS,
    build_supervisor,
    make_model,
)
from src.agents.result_mapper import map_run_result
from src.agents.sdk_tools import RunContext
from src.observability.context import correlation_scope
from src.observability.metrics import registry as M
from src.observability.metrics.cost import estimate_cost_usd
from src.observability.sdk_hooks import OrchestrationHooks
from src.observability.tracing import SpanKind, attributes as A, span

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """Legacy-shaped supervisor delegating to the SDK agent graph."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.name = "RootSupervisorAgent"
        self.model = model
        self._state = AgentState(tool_path=self.name)
        self.streaming_handler: Any | None = None
        # Per-request observability summary (cost/tokens/latency/agents), surfaced
        # in the UI alongside the cumulative Prometheus panel.
        self.last_request_metrics: dict[str, Any] | None = None
        self._last_latency_s: float = 0.0
        # Live registry of the hierarchy (names → SDK Agent) for the UI sidebar.
        self._supervisor = build_supervisor(make_model(model))
        self._child_agents: dict[str, Any] = {
            "ResearchManagerAgent": None,
            "BuildManagerAgent": None,
        }

    @property
    def child_agents(self) -> dict[str, Any]:
        return self._child_agents

    @property
    def state(self) -> AgentState:
        return self._state

    def reset_state(self) -> None:
        self._state = AgentState(tool_path=self.name)

    # ----------------- Orchestration -----------------

    async def orchestrate(
        self,
        user_input: str,
        hitl_manager: HITLManager | None = None,
        enable_hitl: bool = False,
    ) -> SupervisorOutput | None:
        """Run the SDK graph; return `SupervisorOutput`, or `None` if HITL-paused."""
        self.reset_state()
        self._state.current_inputs = {"user_input": user_input}
        logger.info("Supervisor starting orchestration: %s", user_input[:100])

        hooks = OrchestrationHooks(self.streaming_handler, self.model)
        supervisor = build_supervisor(
            make_model(self.model), hitl=enable_hitl, hooks=hooks
        )
        ctx = RunContext()

        try:
            result = await self._run(supervisor, user_input, ctx, hooks)
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

        if getattr(result, "interruptions", None) and hitl_manager is not None:
            self._pause_for_hitl(result, hitl_manager, user_input)
            return None

        return self._finish(result, user_input)

    async def orchestrate_manual(self, user_input: str) -> SupervisorOutput:
        """Run to completion without HITL pauses."""
        result = await self.orchestrate(user_input)
        assert result is not None, "manual orchestration should always return a result"
        return result

    async def resume_orchestration(
        self, hitl_manager: HITLManager, state_id: str
    ) -> SupervisorOutput | None:
        """Resume a HITL-paused run after the user approved/revised/cancelled."""
        state = hitl_manager.get_state(state_id)
        if state is None:
            return SupervisorOutput(
                final_answer=f"Error: state {state_id} not found", subtasks=[]
            )
        self._state = state
        user_input = state.current_inputs.get("user_input", "")

        action = state.hitl_actions[-1] if state.hitl_actions else None
        action_type = action.action if action else HITLActionType.APPROVE

        if action_type == HITLActionType.CANCEL:
            return SupervisorOutput(
                final_answer="Orchestration cancelled by user.", subtasks=[]
            )
        if action_type == HITLActionType.REVISE and action and action.input:
            return await self.orchestrate(action.input, hitl_manager, enable_hitl=True)

        # APPROVE (or SKIP): rebuild the graph, restore the SDK state, approve
        # the pending interruptions, and resume.
        hooks = OrchestrationHooks(self.streaming_handler, self.model)
        supervisor = build_supervisor(make_model(self.model), hitl=True, hooks=hooks)
        # Restore the RunContext type (it serializes as a plain dict) so the
        # resumed sub-runs re-collect traces/docs into a typed context.
        run_state = await RunState.from_json(
            supervisor,
            state.pending_data["_run_state"],
            context_override=RunContext(),
        )
        for item in run_state.get_interruptions():
            run_state.approve(item)

        try:
            result = await Runner.run(supervisor, run_state, hooks=hooks)
        except Exception as e:
            logger.error("HITL resume failed", exc_info=True)
            return SupervisorOutput(final_answer=f"Resume failed: {e}", subtasks=[])

        if getattr(result, "interruptions", None):
            self._pause_for_hitl(result, hitl_manager, user_input)
            return None

        ctx = result.context_wrapper.context
        if not isinstance(ctx, RunContext):
            ctx = RunContext()
        return self._finish(result, user_input, ctx=ctx)

    # ----------------- Internals -----------------

    async def _run(self, supervisor, user_input, ctx, hooks):
        """Execute the run inside a request-scoped span + orchestration metrics.

        Per-agent/tool/LLM spans and token/cost are recorded by `hooks`
        (`OrchestrationHooks`) across the whole sub-run tree; here we own the
        request root: a `request_id`-correlated `SPAN_ORCHESTRATION` and the
        orchestration-level counters/latency.
        """
        M.ACTIVE_ORCHESTRATIONS.inc()
        start = time.perf_counter()
        status = "completed"
        try:
            with correlation_scope(request_id=self._state.state_id):
                async with span(
                    A.SPAN_ORCHESTRATION,
                    kind=SpanKind.SERVER,
                    attributes={
                        A.ORCH_QUERY_PREVIEW: user_input[:120],
                        A.ORCH_QUERY_LEN: len(user_input),
                    },
                ) as sp:
                    result = await Runner.run(
                        supervisor, user_input, context=ctx, hooks=hooks
                    )
                    if sp.is_recording():
                        sp.set_attribute(A.ORCH_STATUS, "completed")
            self._last_ctx = ctx
            self._last_latency_s = time.perf_counter() - start
            return result
        except Exception as exc:
            status = "failed"
            M.ORCHESTRATION_FAILURES.labels(type(exc).__name__).inc()
            raise
        finally:
            M.ACTIVE_ORCHESTRATIONS.dec()
            M.ORCHESTRATION_TOTAL.labels(status).inc()
            M.ORCHESTRATION_DURATION.labels(status).observe(time.perf_counter() - start)

    def _finish(self, result, user_input, ctx=None):
        """Map the result, record request-level metrics, mirror the legacy state."""
        ctx = ctx if ctx is not None else getattr(self, "_last_ctx", RunContext())
        output = map_run_result(result, user_input, ctx)

        # Turns + per-agent tool-selection metrics (the hooks record invocation/
        # duration/LLM tokens; selected/skipped come from the captured traces).
        M.ORCHESTRATION_TURNS.observe(len(output.subtasks))
        for trace in ([*ctx.traces] if isinstance(ctx, RunContext) else []):
            agent = trace.get("agent_name", "")
            for tool in trace.get("selected_tools", []):
                M.AGENT_TOOL_SELECTION.labels(agent, tool, "selected").inc()
            for tool in trace.get("skipped_tools", []):
                M.AGENT_TOOL_SELECTION.labels(agent, tool, "skipped").inc()

        self.last_request_metrics = self._request_metrics(result, ctx, output)
        self._mirror_state(output)
        return output

    def _request_metrics(self, result, ctx, output: SupervisorOutput) -> dict[str, Any]:
        """Per-request observability summary for the UI (this run only)."""
        usage = getattr(result.context_wrapper, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        traces = ctx.traces if isinstance(ctx, RunContext) else []
        agents = [self.name] + [t.get("agent_name", "") for t in traces]
        return {
            "request_id": self._state.state_id,
            "latency_ms": round(self._last_latency_s * 1000, 1),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
            "cost_usd": estimate_cost_usd(self.model, in_tok, out_tok),
            "turns": len(output.subtasks),
            "agents": list(dict.fromkeys(a for a in agents if a)),
            "per_agent": [
                {
                    "agent": t.get("agent_name", ""),
                    "selected_tools": t.get("selected_tools", []),
                    "tool_calls": len(t.get("tool_invocations", [])),
                }
                for t in traces
            ],
        }

    def _mirror_state(self, output: SupervisorOutput) -> None:
        """Replay the orchestration onto `self._state` for the State Inspector."""
        self._state.add_step(
            agent_name=self.name,
            action="task_decomposition",
            observation="Routed: "
            + ", ".join(s.agent_name for s in output.subtasks),
        )
        for sub in output.subtasks:
            self._state.add_step(
                agent_name=sub.agent_name,
                action=f"subtask_complete:{sub.subtask[:60]}",
                observation=str(sub.result)[:300],
            )
        self._state.add_step(
            agent_name=self.name,
            action="orchestration_complete",
            observation=output.final_answer[:300],
        )

    def _pause_for_hitl(
        self, result, hitl_manager: HITLManager, user_input: str
    ) -> None:
        """Persist the SDK RunState inside a legacy AgentState for the HITL UI."""
        run_state = result.to_state()
        pending = list(result.interruptions)
        first = pending[0]
        manager = first.name
        self._state.current_inputs = {"user_input": user_input}
        self._state.tool = manager
        self._state.tool_path = f"{self.name}.{manager}"
        self._state.pause(
            checkpoint_type=HITLCheckpointType.TOOL_EXECUTION,
            pending_data={
                "_run_state": run_state.to_json(),
                "subtask_index": 0,
                "decomposition": {
                    "subtasks": [
                        {
                            "agent_name": p.name,
                            "description": p.arguments.get("input", user_input)
                            if isinstance(p.arguments, dict)
                            else user_input,
                            "tools_needed": MANAGER_WORKERS.get(p.name, []),
                        }
                        for p in pending
                    ]
                },
                "pending_tool": {
                    "agent_name": manager,
                    "description": first.arguments.get("input", user_input)
                    if isinstance(first.arguments, dict)
                    else user_input,
                    "tools_needed": MANAGER_WORKERS.get(manager, []),
                },
                "completed_results": [],
            },
        )
        hitl_manager.capture_state(self._state)
