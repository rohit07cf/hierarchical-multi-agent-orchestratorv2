"""CLI demo: run a handful of representative queries end-to-end.

Run with::

    python -m src.examples.demo_queries

Each query prints the routing decision, the execution plan, the
orchestration trace and the final aggregated response.
"""

from __future__ import annotations

import asyncio
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
]


def _print_state(state: OrchestratorState) -> None:
    """Pretty-print the orchestration trace from `state`."""
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


async def run_demo() -> None:
    """Iterate through `DEMO_QUERIES` and print results."""
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
