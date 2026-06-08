"""Visualization helpers for the Streamlit UI."""

from __future__ import annotations

from typing import Any

import streamlit as st

from models.agent_state import AgentState
from models.supervisor_output import SubtaskResult, SubtaskStatus, SupervisorOutput
from orchestration.agent_tree import AgentTree


def render_agent_tree(tree: AgentTree) -> None:
    """Render the agent hierarchy tree as a Graphviz diagram in Streamlit.

    Args:
        tree: The AgentTree to visualize.
    """
    try:
        digraph = tree.to_digraph()
        st.graphviz_chart(digraph.source)
    except ImportError:
        st.warning("Install graphviz for tree visualization: pip install graphviz")
        _render_text_tree(tree)


def _render_text_tree(tree: AgentTree) -> None:
    """Render a text-based fallback of the agent tree."""
    root = tree.root
    st.code(_format_tree_node(root, indent=0), language="text")


def _format_tree_node(node: Any, indent: int) -> str:
    """Format a single tree node as indented text."""
    prefix = "  " * indent
    tools_str = ", ".join(node.tools) if node.tools else "none"
    lines = [f"{prefix}{'└── ' if indent > 0 else ''}{node.name} [{tools_str}]"]
    for child in node.children:
        lines.append(_format_tree_node(child, indent + 1))
    return "\n".join(lines)


def render_subtask_table(subtasks: list[SubtaskResult]) -> None:
    """Render a table of subtask results.

    Args:
        subtasks: List of SubtaskResult objects to display.
    """
    if not subtasks:
        st.info("No subtasks executed yet.")
        return

    rows = []
    for s in subtasks:
        status_icon = {
            SubtaskStatus.COMPLETED: "✅",
            SubtaskStatus.FAILED: "❌",
            SubtaskStatus.RUNNING: "🔄",
            SubtaskStatus.PENDING: "⏳",
            SubtaskStatus.CANCELLED: "🚫",
        }.get(s.status, "❓")

        rows.append({
            "Status": f"{status_icon} {s.status.value}",
            "Agent": s.agent_name,
            "Subtask": s.subtask[:80] + ("..." if len(s.subtask) > 80 else ""),
            "Result": str(s.result)[:100] if s.result else "-",
            "Error": s.error or "-",
        })

    st.table(rows)


def render_agent_state(state: AgentState) -> None:
    """Render the full agent state for debugging inspection.

    Args:
        state: The AgentState to display.
    """
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Iteration Count", state.iteration_count)
    with col2:
        st.metric("Steps", len(state.intermediate_steps))
    with col3:
        paused_text = "PAUSED" if state.is_paused else "Active"
        st.metric("Status", paused_text)

    st.text(f"State ID: {state.state_id}")
    st.text(f"Tool Path: {state.tool_path or 'N/A'}")
    st.text(f"Current Tool: {state.tool or 'N/A'}")

    if state.intermediate_steps:
        st.subheader("Execution Steps")
        for step in state.intermediate_steps:
            with st.expander(f"Step {step.step_number}: {step.action}"):
                st.text(f"Agent: {step.agent_name}")
                st.text(f"Timestamp: {step.timestamp}")
                if step.action_input:
                    st.json(step.action_input)
                if step.observation:
                    st.text(f"Observation: {step.observation[:500]}")

    if state.hitl_actions:
        st.subheader("HITL History")
        for action in state.hitl_actions:
            st.text(
                f"  [{action.timestamp}] {action.action.value}: "
                f"{action.reason or 'no reason'}"
            )


def render_reasoning_panel(output: SupervisorOutput) -> None:
    """Render the supervisor's reasoning and decomposition panel.

    Args:
        output: The SupervisorOutput containing decomposition info.
    """
    if output.decomposition:
        st.subheader("Task Decomposition")
        st.text(f"Original Request: {output.decomposition.original_request}")
        st.markdown(f"**Reasoning:** {output.decomposition.reasoning}")

        if output.decomposition.subtasks:
            st.markdown("**Planned Subtasks:**")
            for i, sub in enumerate(output.decomposition.subtasks, 1):
                tools = ", ".join(sub.tools_needed) if sub.tools_needed else "auto"
                st.text(f"  {i}. [{sub.agent_name}] {sub.description} (tools: {tools})")


def render_llm_mode_banner() -> None:
    """Show a banner indicating whether agents run in real or mock LLM mode."""
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        st.success("LLM mode: **real** (ANTHROPIC_API_KEY set)")
    else:
        st.warning(
            "LLM mode: **mock** — agents emit reasoning traces and run "
            "deterministic tools, but synthesis is a labelled placeholder. "
            "Set `ANTHROPIC_API_KEY` for real natural-language reasoning."
        )


def render_agent_trace(trace: dict[str, Any]) -> None:
    """Render a single agent's reasoning trace.

    `trace` is the serialized `AgentTrace` dict produced by a
    `ReasoningAgent`. Shows the reasoning text, which tools were selected
    or skipped, each tool invocation's rationale + preview, and the
    final response.
    """
    if not trace:
        return

    mode = trace.get("llm_mode", "?")
    if mode == "mock":
        st.caption(f"LLM mode: `{mode}` (offline placeholder)")
    else:
        st.caption(f"LLM mode: `{mode}`")

    st.markdown(f"**Reasoning:** {trace.get('reasoning', '—')}")

    selected = trace.get("selected_tools") or []
    skipped = trace.get("skipped_tools") or []
    available = trace.get("available_tools") or []

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Available tools**")
        if available:
            st.markdown("\n".join(f"- `{t}`" for t in available))
        else:
            st.text("none")
    with col2:
        st.markdown("**Selected**")
        if selected:
            st.markdown("\n".join(f"- ✅ `{t}`" for t in selected))
        else:
            st.text("none")
    with col3:
        st.markdown("**Skipped**")
        if skipped:
            st.markdown("\n".join(f"- ⏭ `{t}`" for t in skipped))
        else:
            st.text("none")

    invocations = trace.get("tool_invocations") or []
    if invocations:
        st.markdown("**Tool invocations**")
        for inv in invocations:
            status = "✅" if inv.get("success") else "❌"
            with st.expander(f"{status} `{inv.get('tool_name', '?')}`"):
                if inv.get("rationale"):
                    st.markdown(f"_Rationale:_ {inv['rationale']}")
                if inv.get("arguments"):
                    st.markdown("_Arguments:_")
                    st.json(inv["arguments"])
                if inv.get("error"):
                    st.error(inv["error"])
                if inv.get("result_preview"):
                    st.markdown("_Result preview:_")
                    st.code(inv["result_preview"])


def render_agent_traces_for_subtasks(subtasks: list[SubtaskResult]) -> None:
    """Render one expandable trace block per subtask (manager + nested workers)."""
    if not subtasks:
        st.info("No agent traces yet.")
        return

    for sub in subtasks:
        traces = [
            tc["agent_trace"]
            for tc in (sub.tool_calls or [])
            if isinstance(tc, dict) and "agent_trace" in tc
        ]
        if not traces:
            continue
        with st.expander(f"🧠 {sub.agent_name} — agent reasoning trace", expanded=False):
            for i, trace in enumerate(traces):
                if i > 0:
                    st.divider()
                    st.caption(f"Nested worker: **{trace.get('agent_name', '?')}**")
                render_agent_trace(trace)
