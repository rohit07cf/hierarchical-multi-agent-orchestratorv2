"""CLI demo — runs a handful of representative queries end-to-end.

Run with::

    ANTHROPIC_API_KEY=... python -m src.examples.demo_queries

Each query goes through the OpenAI Agents SDK graph (Supervisor → Managers →
Workers) on Claude via LiteLLM. Prints the final answer plus, per subtask, the
manager/worker reasoning traces.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Allow running directly via `python src/examples/demo_queries.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_defs.supervisor import SupervisorAgent  # noqa: E402
from models.supervisor_output import SupervisorOutput  # noqa: E402
from src.observability import init_observability  # noqa: E402

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


def _print_output(output: SupervisorOutput) -> None:
    """Pretty-print the supervisor output + per-agent traces."""
    assert output.decomposition is not None
    print(f"Reasoning: {output.decomposition.reasoning}")
    print("Subtasks:")
    for sub in output.subtasks:
        print(f"  [{sub.agent_name}] {sub.status.value}: {str(sub.result)[:80]}")
        for call in sub.tool_calls:
            trace = call.get("agent_trace", {})
            print(
                f"    - {trace.get('agent_name')} "
                f"selected={trace.get('selected_tools')} "
                f"skipped={trace.get('skipped_tools')}"
            )


async def run_demo() -> None:
    """Iterate through `DEMO_QUERIES` and print results."""
    init_observability()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is required (the app no longer has an "
              "offline mock mode).")
        return

    supervisor = SupervisorAgent()
    for i, query in enumerate(DEMO_QUERIES, 1):
        print("\n" + "=" * 72)
        print(f"Demo {i}: {query}")
        print("=" * 72)

        output = await supervisor.orchestrate_manual(query)
        _print_output(output)

        print("\nFinal answer:")
        print(output.final_answer)


if __name__ == "__main__":
    asyncio.run(run_demo())
