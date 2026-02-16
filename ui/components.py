"""Reusable UI components for the Streamlit application."""

from __future__ import annotations

import asyncio
from typing import Any

import streamlit as st

from models.agent_state import AgentState, HITLAction, HITLActionType, HITLCheckpointType
from models.streaming_models import StreamingModelResponseStep, StreamingStatus
from orchestration.hitl_manager import HITLManager


def render_message_input() -> str | None:
    """Render the user message input area.

    Returns:
        The user's input text if submitted, None otherwise.
    """
    with st.form("message_form", clear_on_submit=True):
        user_input = st.text_area(
            "Enter your request:",
            placeholder="e.g., What's the sentiment of 'I love this product'? Also multiply 5 * 3",
            height=100,
            key="user_input",
        )
        col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
        with col1:
            submitted = st.form_submit_button("Send", type="primary")
        with col2:
            use_manual = st.form_submit_button("Send (Manual Mode)")
        with col3:
            use_hitl = st.form_submit_button("Send (HITL Mode)")

    if submitted and user_input.strip():
        st.session_state["orchestration_mode"] = "auto"
        return user_input.strip()
    elif use_manual and user_input.strip():
        st.session_state["orchestration_mode"] = "manual"
        return user_input.strip()
    elif use_hitl and user_input.strip():
        st.session_state["orchestration_mode"] = "hitl"
        return user_input.strip()
    return None


def render_streaming_output(steps: list[StreamingModelResponseStep]) -> None:
    """Render streaming output steps in the UI.

    Args:
        steps: List of streaming steps to display.
    """
    if not steps:
        return

    for step in steps:
        icon = {
            StreamingStatus.STARTED: "🚀",
            StreamingStatus.IN_PROGRESS: "💬",
            StreamingStatus.TOOL_CALLED: "🔧",
            StreamingStatus.TOOL_COMPLETED: "✅",
            StreamingStatus.REASONING: "🧠",
            StreamingStatus.COMPLETED: "🏁",
            StreamingStatus.ERROR: "❌",
            StreamingStatus.HITL_PAUSED: "⏸️",
        }.get(step.status, "📋")

        if step.status == StreamingStatus.ERROR:
            st.error(f"{icon} [{step.name}] {step.message_fragment}")
        elif step.status == StreamingStatus.HITL_PAUSED:
            st.warning(f"{icon} [{step.name}] {step.message_fragment}")
        elif step.status in (StreamingStatus.TOOL_CALLED, StreamingStatus.TOOL_COMPLETED):
            st.info(f"{icon} [{step.name}] {step.message_fragment}")
        else:
            st.text(f"{icon} [{step.name}] {step.message_fragment}")


def render_hitl_controls(hitl_manager: HITLManager) -> HITLAction | None:
    """Render HITL control buttons for paused states.

    Shows checkpoint-specific UI:
    - DECOMPOSITION: Shows the planned subtasks for review
    - TOOL_EXECUTION: Shows the pending tool call for confirmation

    Args:
        hitl_manager: The HITLManager to query for paused states.

    Returns:
        The HITLAction if a button was clicked, None otherwise.
    """
    paused_states = hitl_manager.get_all_paused()
    if not paused_states:
        return None

    st.warning(f"⏸️ Execution paused — {len(paused_states)} state(s) awaiting review")

    for state in paused_states:
        checkpoint = state.checkpoint_type

        if checkpoint == HITLCheckpointType.DECOMPOSITION:
            return _render_decomposition_review(state)
        elif checkpoint == HITLCheckpointType.TOOL_EXECUTION:
            return _render_tool_confirmation(state)
        else:
            return _render_generic_review(state)

    return None


def _render_decomposition_review(state: AgentState) -> HITLAction | None:
    """Render the decomposition review HITL panel."""
    pending = state.pending_data
    decomposition = pending.get("decomposition", {})
    subtasks = decomposition.get("subtasks", [])
    reasoning = decomposition.get("reasoning", "")

    with st.expander("📋 Review Task Decomposition Plan", expanded=True):
        st.markdown(f"**Reasoning:** {reasoning}")
        st.markdown("**Planned subtasks:**")

        for i, subtask in enumerate(subtasks):
            agent = subtask.get("agent_name", "Unknown")
            desc = subtask.get("description", "")
            tools = ", ".join(subtask.get("tools_needed", []))
            st.markdown(f"  {i + 1}. **[{agent}]** {desc} _(tools: {tools})_")

        st.divider()
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✅ Approve Plan", key=f"approve_{state.state_id}", type="primary"):
                return HITLAction(
                    action=HITLActionType.APPROVE,
                    reason="User approved the decomposition plan",
                )

        with col2:
            revised = st.text_input(
                "Revise request:",
                key=f"revise_input_{state.state_id}",
                placeholder="Enter modified request...",
            )
            if st.button("✏️ Revise & Re-plan", key=f"revise_{state.state_id}"):
                return HITLAction(
                    action=HITLActionType.REVISE,
                    input=revised,
                    reason="User revised the input for re-decomposition",
                )

        with col3:
            if st.button("❌ Cancel", key=f"cancel_{state.state_id}"):
                return HITLAction(
                    action=HITLActionType.CANCEL,
                    reason="User cancelled the orchestration",
                )

    return None


def _render_tool_confirmation(state: AgentState) -> HITLAction | None:
    """Render the tool execution confirmation HITL panel."""
    pending = state.pending_data
    pending_tool = pending.get("pending_tool", {})
    completed = pending.get("completed_results", [])
    agent_name = pending_tool.get("agent_name", "Unknown")
    description = pending_tool.get("description", "")
    tools_needed = pending_tool.get("tools_needed", [])
    subtask_index = pending.get("subtask_index", 0)
    total_subtasks = len(pending.get("decomposition", {}).get("subtasks", []))

    with st.expander(
        f"🔧 Confirm Tool Execution — Step {subtask_index + 1}/{total_subtasks}",
        expanded=True,
    ):
        # Show progress so far
        if completed:
            st.markdown("**Completed so far:**")
            for r in completed:
                status_icon = "✅" if r.get("status") == "completed" else "❌"
                st.markdown(f"  {status_icon} [{r.get('agent_name')}] {r.get('subtask')}")
            st.divider()

        # Show pending tool
        st.markdown("**Next tool to execute:**")
        st.markdown(f"  - **Agent:** {agent_name}")
        st.markdown(f"  - **Task:** {description}")
        st.markdown(f"  - **Tools:** {', '.join(tools_needed)}")

        st.divider()
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(
                f"✅ Run {agent_name}",
                key=f"approve_{state.state_id}",
                type="primary",
            ):
                return HITLAction(
                    action=HITLActionType.APPROVE,
                    reason=f"User approved execution of {agent_name}: {description}",
                )

        with col2:
            revised = st.text_input(
                "Revise task:",
                key=f"revise_input_{state.state_id}",
                placeholder="Modify the subtask description...",
            )
            if st.button("✏️ Revise Task", key=f"revise_{state.state_id}"):
                return HITLAction(
                    action=HITLActionType.REVISE,
                    input=revised,
                    reason=f"User revised subtask for {agent_name}",
                )

        with col3:
            if st.button("⏭️ Skip", key=f"skip_{state.state_id}"):
                return HITLAction(
                    action=HITLActionType.APPROVE,
                    reason=f"User skipped {agent_name} (approved to continue)",
                )

    return None


def _render_generic_review(state: AgentState) -> HITLAction | None:
    """Render a generic HITL review panel for unknown checkpoint types."""
    with st.expander(f"Review: {state.tool_path} (Step {state.iteration_count})"):
        st.text(f"State ID: {state.state_id}")
        st.text(f"Current Tool: {state.tool or 'N/A'}")
        st.json(state.current_inputs)

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Approve", key=f"approve_{state.state_id}", type="primary"):
                return HITLAction(action=HITLActionType.APPROVE, reason="User approved")

        with col2:
            revised = st.text_input(
                "Revised input:",
                key=f"revise_input_{state.state_id}",
            )
            if st.button("Revise", key=f"revise_{state.state_id}"):
                return HITLAction(
                    action=HITLActionType.REVISE,
                    input=revised,
                    reason="User revised input",
                )

        with col3:
            if st.button("Cancel", key=f"cancel_{state.state_id}"):
                return HITLAction(action=HITLActionType.CANCEL, reason="User cancelled")

    return None


def render_model_selector() -> str:
    """Render a model selection dropdown.

    Returns:
        The selected model name.
    """
    models = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1", "gpt-4o", "gpt-4o-mini"]
    return st.selectbox("Model:", models, index=0, key="model_select") or models[0]


def render_conversation_history(messages: list[dict[str, str]]) -> None:
    """Render the conversation history in a chat-like format.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
    """
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            st.chat_message("user").markdown(content)
        else:
            st.chat_message("assistant").markdown(content)
