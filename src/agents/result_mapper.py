"""Map an SDK ``RunResult`` (+ shared ``RunContext``) into the legacy
``SupervisorOutput`` the Streamlit UI consumes.

Worker/manager traces are NOT in the top-level ``RunResult.new_items`` (each
``as_tool`` runs an isolated sub-run), so they are collected during the run
into ``RunContext.traces`` by ``traced_as_tool``. This mapper combines:

- the supervisor's own trace (from the top-level items),
- the per-manager + per-worker traces (from ``ctx.traces``),

into ``SubtaskResult``s, plus a ``TaskDecomposition`` reconstructed from the
manager calls the supervisor actually made (there is no upfront plan anymore).
"""

from __future__ import annotations

from agents import RunResult, ToolCallItem, ToolCallOutputItem

from models.supervisor_output import (
    PlannedSubtask,
    SubtaskResult,
    SubtaskStatus,
    SupervisorOutput,
    TaskDecomposition,
)
from src.agents.factory import MANAGER_TOOLS, MANAGER_WORKERS, TOOLS_BY_MANAGER
from src.agents.sdk_tools import RunContext, _args, trace_from_run

# Tokens that distinguish real source code from a directory-tree code fence
# (the summarized answer often fences the project tree, which has none of these).
_CODE_TOKENS = ("import ", "def ", "class ", "function ", "const ", "func ", "public ")


def _preserve_code_artifact(final_answer: str, traces_by_name: dict) -> str:
    """Guarantee generated code reaches the user.

    The supervisor and build manager *summarize* their workers, so the
    CodingAgent's full source — the actual deliverable — is otherwise dropped
    from the user-facing answer (it only survives in the trace). Rather than
    pay to relay a large codebase verbatim through two LLM layers, we attach it
    structurally: if code was generated but the summarized answer carries none,
    append the CodingAgent's complete output as a "Generated code" artifact.
    """
    coding = traces_by_name.get("CodingAgent")
    if not coding:
        return final_answer
    code = (coding.get("final_response") or "").strip()
    produced_code = "```" in code and any(t in code for t in _CODE_TOKENS)
    answer_has_code = any(t in final_answer for t in _CODE_TOKENS)
    if produced_code and not answer_has_code:
        return f"{final_answer}\n\n---\n\n## 📦 Generated code\n\n{code}"
    return final_answer


def map_run_result(
    result: RunResult, user_query: str, ctx: RunContext
) -> SupervisorOutput:
    """Project a finished run into the legacy ``SupervisorOutput``."""
    supervisor_trace = trace_from_run("RootSupervisorAgent", MANAGER_TOOLS, result)
    traces_by_name = {t["agent_name"]: t for t in ctx.traces}
    traces_by_name["RootSupervisorAgent"] = supervisor_trace

    outputs = {
        it.call_id: it
        for it in result.new_items
        if isinstance(it, ToolCallOutputItem)
    }
    manager_calls = [
        it
        for it in result.new_items
        if isinstance(it, ToolCallItem) and it.tool_name in MANAGER_WORKERS
    ]

    subtasks: list[SubtaskResult] = []
    for call in manager_calls:
        manager = call.tool_name
        out = outputs.get(call.call_id)
        tool_calls: list[dict] = []
        if manager in traces_by_name:
            tool_calls.append({"agent_trace": traces_by_name[manager]})
        for worker in MANAGER_WORKERS[manager]:
            if worker in traces_by_name:
                tool_calls.append({"agent_trace": traces_by_name[worker]})
        subtasks.append(
            SubtaskResult(
                agent_name=manager,
                subtask=_args(call).get("input", user_query),
                result=(out.output if out is not None else None),
                status=SubtaskStatus.COMPLETED if out is not None else SubtaskStatus.FAILED,
                tool_calls=tool_calls,
            )
        )

    decomposition = TaskDecomposition(
        original_request=user_query,
        reasoning=supervisor_trace["reasoning"]
        or "Reconstructed from the supervisor's executed routing decisions.",
        subtasks=[
            PlannedSubtask(
                agent_name=s.agent_name,
                description=s.subtask,
                tools_needed=TOOLS_BY_MANAGER.get(s.agent_name, []),
                depends_on=[i - 1] if i > 0 else [],
            )
            for i, s in enumerate(subtasks)
        ],
    )

    final_answer = _preserve_code_artifact(
        str(result.final_output or ""), traces_by_name
    )
    return SupervisorOutput(
        final_answer=final_answer,
        subtasks=subtasks,
        decomposition=decomposition,
    )
