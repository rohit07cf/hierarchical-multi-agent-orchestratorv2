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
