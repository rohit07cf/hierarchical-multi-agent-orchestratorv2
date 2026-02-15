"""Main Streamlit application for the Hierarchical Multi-Agent Orchestrator."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent_defs.supervisor import SupervisorAgent
from config.settings import get_settings
from models.agent_state import HITLAction, HITLActionType
from models.supervisor_output import SupervisorOutput
from orchestration.agent_tree import AgentTree
from orchestration.hitl_manager import HITLManager
from orchestration.streaming_handler import StreamingCallbackHandler
from ui.components import (
    render_conversation_history,
    render_hitl_controls,
    render_message_input,
    render_model_selector,
    render_streaming_output,
)
from ui.visualizations import (
    render_agent_state,
    render_agent_tree,
    render_reasoning_panel,
    render_subtask_table,
)
from utils.logging import setup_logging

logger = logging.getLogger(__name__)


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    defaults = {
        "messages": [],
        "orchestration_results": [],
        "streaming_steps": [],
        "agent_tree": AgentTree.build_default_tree(),
        "hitl_manager": HITLManager(persistence_dir=get_settings().hitl_persistence_dir),
        "streaming_handler": StreamingCallbackHandler(),
        "supervisor": None,
        "orchestration_mode": "auto",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_supervisor(model: str) -> SupervisorAgent:
    """Get or create the supervisor agent.

    Args:
        model: OpenAI model name to use.

    Returns:
        SupervisorAgent instance.
    """
    if st.session_state["supervisor"] is None or st.session_state.get("current_model") != model:
        st.session_state["supervisor"] = SupervisorAgent(model=model)
        st.session_state["current_model"] = model
    return st.session_state["supervisor"]


async def run_orchestration(user_input: str, model: str, mode: str) -> SupervisorOutput:
    """Run the supervisor orchestration asynchronously.

    Args:
        user_input: The user's request.
        model: OpenAI model to use.
        mode: 'auto' for handoff-based or 'manual' for explicit delegation.

    Returns:
        SupervisorOutput from the orchestration.
    """
    supervisor = get_supervisor(model)

    if mode == "manual":
        return await supervisor.orchestrate_manual(user_input)
    return await supervisor.orchestrate(user_input)


def render_sidebar() -> str:
    """Render the sidebar with configuration and agent tree visualization.

    Returns:
        Selected model name.
    """
    with st.sidebar:
        st.header("Configuration")
        model = render_model_selector()

        st.divider()
        st.header("Agent Hierarchy")
        render_agent_tree(st.session_state["agent_tree"])

        st.divider()
        st.header("System Info")
        tree = st.session_state["agent_tree"]
        all_agents = tree.get_all_agents()
        st.text(f"Total Agents: {len(all_agents)}")
        for agent_node in all_agents:
            st.text(f"  {agent_node.name}: {', '.join(agent_node.tools)}")

        st.divider()
        if st.button("Clear History"):
            st.session_state["messages"] = []
            st.session_state["orchestration_results"] = []
            st.session_state["streaming_steps"] = []
            st.rerun()

    return model


def render_main_content() -> None:
    """Render the main content area with chat and results."""
    # Conversation history
    render_conversation_history(st.session_state["messages"])

    # Latest orchestration results
    if st.session_state["orchestration_results"]:
        latest: SupervisorOutput = st.session_state["orchestration_results"][-1]

        # Reasoning panel
        with st.expander("Reasoning & Decomposition", expanded=False):
            render_reasoning_panel(latest)

        # Subtask results
        with st.expander("Subtask Results", expanded=False):
            render_subtask_table(latest.subtasks)

    # Streaming output
    if st.session_state["streaming_steps"]:
        with st.expander("Streaming Log", expanded=False):
            render_streaming_output(st.session_state["streaming_steps"])

    # HITL controls
    hitl_manager: HITLManager = st.session_state["hitl_manager"]
    hitl_action = render_hitl_controls(hitl_manager)
    if hitl_action:
        paused = hitl_manager.get_all_paused()
        if paused:
            hitl_manager.apply_action(paused[0].state_id, hitl_action)
            st.rerun()


def render_state_inspector() -> None:
    """Render the state inspector in an expander."""
    if not st.session_state["orchestration_results"]:
        return

    with st.expander("State Inspector (Debug)", expanded=False):
        latest: SupervisorOutput = st.session_state["orchestration_results"][-1]
        supervisor = st.session_state.get("supervisor")
        if supervisor:
            render_agent_state(supervisor.state)
        st.subheader("Raw Output")
        st.json(json.loads(latest.model_dump_json()))


def run_app() -> None:
    """Main Streamlit application entry point."""
    st.set_page_config(
        page_title="Multi-Agent Orchestrator",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    setup_logging(get_settings().log_level)
    init_session_state()

    st.title("Hierarchical Multi-Agent Orchestrator")
    st.caption(
        "Supervisor-driven task decomposition with HITL support, "
        "real-time streaming, and Temporal workflow durability"
    )

    model = render_sidebar()

    # Message input
    user_input = render_message_input()

    if user_input:
        # Add user message to history
        st.session_state["messages"].append({"role": "user", "content": user_input})

        mode = st.session_state.get("orchestration_mode", "auto")

        with st.spinner(f"Processing with {model} ({mode} mode)..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    run_orchestration(user_input, model, mode)
                )
                loop.close()

                # Store results
                st.session_state["orchestration_results"].append(result)
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": result.final_answer,
                })

                # Capture streaming steps from handler
                handler: StreamingCallbackHandler = st.session_state["streaming_handler"]
                st.session_state["streaming_steps"] = handler.history

            except Exception as e:
                st.error(f"Orchestration failed: {e}")
                logger.error("Orchestration error", exc_info=True)

        st.rerun()

    # Render main content
    render_main_content()
    render_state_inspector()


if __name__ == "__main__":
    run_app()
