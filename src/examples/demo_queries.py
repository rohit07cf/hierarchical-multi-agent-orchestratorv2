"""CLI demo — runs a handful of representative queries end-to-end.

Run with::

    python -m src.examples.demo_queries

Prints, for each query:

- the user query
- the routing/plan reasoning
- the execution plan
- the orchestration trace (one line per ExecutionStep)
- each agent's reasoning trace (selected/skipped tools + tool calls)
- the final aggregated response
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Allow running directly via `python src/examples/demo_queries.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.state_models import OrchestratorState  # noqa: E402
from src.orchestrator.supervisor import RootSupervisorAgent  # noqa: E402

DEMO_QUERIES: list[str] = [
    "Summarize the architecture of this project.",
    "Build a FastAPI endpoint for uploading documents and review the solution.",
    "Search the knowledge base for agent orchestration patterns and generate implementation guidance.",
    "Generate a simple Redis memory tool and review it for production concerns.",
    (
        "The meaning of life is to live, to understand, and to create "
        "something meaningful to share with others."
    ),
]


def _print_trace(trace: dict | None, indent: str = "  ") -> None:
    """Pretty-print a single agent's reasoning trace."""
    if not trace:
        return
    print(
        f"{indent}[{trace.get('agent_name')}] mode={trace.get('llm_mode')} "
        f"selected={trace.get('selected_tools')} "
        f"skipped={trace.get('skipped_tools')}"
    )
    print(f"{indent}  reasoning: {trace.get('reasoning', '')[:200]}")
    for inv in trace.get("tool_invocations") or []:
        status = "ok" if inv.get("success") else "ERR"
        print(
            f"{indent}  tool {inv.get('tool_name')} [{status}]: "
            f"{inv.get('result_preview', '')[:120]}"
        )


def _print_state(state: OrchestratorState) -> None:
    """Pretty-print the full orchestration trace from `state`."""
    assert state.plan is not None
    print(f"Reasoning: {state.plan.reasoning}")
    print("Execution plan:")
    for i, task in enumerate(state.plan.tasks, 1):
        print(
            f"  {i}. {task.agent_name} — {task.description[:80]} "
            f"(tools: {', '.join(task.tools_needed) or 'auto'})"
        )

    print("Orchestration trace:")
    for step in state.steps:
        print(f"  [{step.agent_name}] {step.kind.value}: {step.message}")

    print("Agent reasoning traces:")
    for task in state.plan.tasks:
        if not isinstance(task.result, dict):
            continue
        trace = task.result.get("trace")
        _print_trace(trace, indent="  ")
        # Nested worker traces (managers expose these in data.worker_traces).
        for worker_trace in (task.result.get("data") or {}).get(
            "worker_traces", []
        ):
            _print_trace(worker_trace, indent="    ")


async def run_demo() -> None:
    """Iterate through `DEMO_QUERIES` and print results."""
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "WARNING: OPENAI_API_KEY is not set — agents run in mock mode. "
            "Reasoning traces and tool outputs are real; synthesis is a "
            "labelled placeholder.\n"
        )

    supervisor = RootSupervisorAgent()
    for i, query in enumerate(DEMO_QUERIES, 1):
        print("\n" + "=" * 72)
        print(f"Demo {i}: {query}")
        print("=" * 72)

        state, response = await supervisor.orchestrate(query)
        _print_state(state)

        print("\nFinal response:")
        print(response.content)


if __name__ == "__main__":
    asyncio.run(run_demo())
