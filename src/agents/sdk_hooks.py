"""SDK ``RunHooks`` that drive streaming + per-request observability.

Threaded into the top run AND every ``as_tool`` sub-run (the SDK does not
propagate hooks into sub-runs), so it observes the whole Supervisor → Manager
→ Worker tree:

- per-agent invocation + duration metrics,
- per-agent LLM token/cost (the ``operation`` label is the agent name, giving a
  per-agent cost breakdown in the panel),
- per-tool usage + duration + routing/manager-selection counters,
- streaming timeline events for the Streamlit "Streaming Log".

Metrics are recorded **directly** (counters/histograms) rather than via the
``observe_*`` async context managers: the SDK fires start/end hooks in
*different* asyncio tasks, so entering a context manager in ``*_start`` and
exiting it in ``*_end`` corrupts the OpenTelemetry/contextvar token stack
("created in a different Context"). The request-level OTel span is owned by the
bridge, where enter/exit happen in one frame.

Agents and LLM calls are globally sequential (``as_tool`` sub-runs are awaited;
a model call returns before the next agent starts); only tool calls may run in
parallel, so per-(name) start times are matched LIFO/FIFO.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from agents import RunHooks

from src.agents.factory import MANAGER_TOOLS
from src.observability.metrics import registry as M
from src.observability.metrics.cost import estimate_cost_usd
from src.observability.tracing import attributes as A

_SUPERVISOR = "RootSupervisorAgent"


class OrchestrationHooks(RunHooks):
    """Streaming + observability for one orchestration run (pass to every run)."""

    def __init__(self, handler: Any | None = None, model: str = "claude-opus-4-8") -> None:
        self._h = handler  # StreamingCallbackHandler | None
        self._model = model
        self._agent_starts: dict[str, list[float]] = defaultdict(list)
        self._tool_starts: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._llm_start: tuple[str, float] | None = None

    # ----------------- Agents (invocation + duration) -----------------

    async def on_agent_start(self, context: Any, agent: Any) -> None:
        M.AGENT_INVOCATIONS.labels(agent.name, A.layer_for(agent.name)).inc()
        self._agent_starts[agent.name].append(time.perf_counter())
        if self._h and agent.name == _SUPERVISOR:
            self._h.push_orchestration_step(
                agent.name, "task_decomposition", "Routing the request"
            )

    async def on_agent_end(self, context: Any, agent: Any, output: Any) -> None:
        starts = self._agent_starts.get(agent.name)
        if starts:
            M.AGENT_DURATION.labels(agent.name, A.layer_for(agent.name)).observe(
                time.perf_counter() - starts.pop()
            )
        if self._h and agent.name == _SUPERVISOR:
            self._h.push_orchestration_step(
                agent.name, "orchestration_complete", "Orchestration complete"
            )

    # ----------------- Tools (usage/duration + routing) -----------------

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        name = getattr(tool, "name", "tool")
        M.TOOL_USAGE.labels(name).inc()
        self._tool_starts[(name, agent.name)].append(time.perf_counter())
        if name in MANAGER_TOOLS:
            M.MANAGER_SELECTION.labels(name).inc()
            M.ROUTING_DECISIONS_TOTAL.labels(name, "llm").inc()
            if self._h:
                self._h.push_orchestration_step(
                    name, "subtask_started", f"{name} starting"
                )

    async def on_tool_end(self, context: Any, agent: Any, tool: Any, result: str) -> None:
        name = getattr(tool, "name", "tool")
        starts = self._tool_starts.get((name, agent.name))
        if starts:
            M.TOOL_DURATION.labels(name, agent.name).observe(
                time.perf_counter() - starts.pop(0)
            )
        if self._h and name in MANAGER_TOOLS:
            self._h.push_orchestration_step(
                name, "subtask_complete", f"{name} completed"
            )

    # ----------------- LLM (per-agent tokens/cost/duration) -----------------

    async def on_llm_start(
        self, context: Any, agent: Any, system_prompt: Any, input_items: Any
    ) -> None:
        self._llm_start = (agent.name, time.perf_counter())

    async def on_llm_end(self, context: Any, agent: Any, response: Any) -> None:
        if self._llm_start is None:
            return
        name, start = self._llm_start
        self._llm_start = None
        M.LLM_DURATION.labels(self._model, name, "real").observe(
            time.perf_counter() - start
        )
        usage = getattr(response, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        M.LLM_TOKENS_INPUT.labels(self._model, name).inc(in_tok)
        M.LLM_TOKENS_OUTPUT.labels(self._model, name).inc(out_tok)
        M.LLM_COST_ESTIMATE.labels(self._model, name).inc(
            estimate_cost_usd(self._model, in_tok, out_tok)
        )
