"""HITL (Human-In-The-Loop) state management for pause/resume/revise workflows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from models.agent_state import AgentState, HITLAction, HITLActionType

logger = logging.getLogger(__name__)


class HITLManager:
    """Manages human-in-the-loop state for agent execution pause/resume cycles.

    Handles state capture, serialization, persistence, and restoration
    to enable users to review, revise, or cancel agent actions at
    any checkpoint during execution.
    """

    def __init__(self, persistence_dir: str | None = None) -> None:
        self._active_states: dict[str, AgentState] = {}
        self._persistence_dir = Path(persistence_dir) if persistence_dir else None
        if self._persistence_dir:
            self._persistence_dir.mkdir(parents=True, exist_ok=True)

    @property
    def active_states(self) -> dict[str, AgentState]:
        """Access all currently active (paused) agent states."""
        return self._active_states

    def has_paused_state(self, state_id: str | None = None) -> bool:
        """Check if there is a paused state, optionally for a specific ID."""
        if state_id:
            state = self._active_states.get(state_id)
            return state is not None and state.is_paused
        return any(s.is_paused for s in self._active_states.values())

    def capture_state(self, state: AgentState) -> str:
        """Capture an agent's execution state for HITL review.

        The state should already be paused (via state.pause()) with checkpoint_type
        and pending_data set before calling this method.  If not yet paused, a
        bare pause() is applied as a fallback.

        Args:
            state: The AgentState to capture (should already be paused).

        Returns:
            The state_id for later retrieval.
        """
        if not state.is_paused:
            state.pause()
        self._active_states[state.state_id] = state

        if self._persistence_dir:
            self._persist_state(state)

        logger.info("HITL state captured: %s (path: %s)", state.state_id, state.tool_path)
        return state.state_id

    def get_state(self, state_id: str) -> AgentState | None:
        """Retrieve a captured state by its ID.

        Args:
            state_id: The unique state identifier.

        Returns:
            The AgentState if found, None otherwise.
        """
        state = self._active_states.get(state_id)
        if state is None and self._persistence_dir:
            state = self._load_state(state_id)
            if state:
                self._active_states[state_id] = state
        return state

    def apply_action(self, state_id: str, action: HITLAction) -> AgentState | None:
        """Apply a HITL action to a paused state and resume execution.

        Args:
            state_id: The state to apply the action to.
            action: The HITL action (CANCEL, REVISE, APPROVE).

        Returns:
            The updated AgentState, or None if state not found.
        """
        state = self.get_state(state_id)
        if state is None:
            logger.warning("HITL state not found: %s", state_id)
            return None

        if not state.is_paused:
            logger.warning("HITL state not paused: %s", state_id)
            return None

        logger.info(
            "Applying HITL action %s to state %s (reason: %s)",
            action.action.value,
            state_id,
            action.reason or "none",
        )

        if action.action == HITLActionType.CANCEL:
            state.resume(action)
            self._cleanup_state(state_id)
            return state

        state.resume(action)

        if self._persistence_dir:
            self._persist_state(state)

        return state

    def approve(self, state_id: str, reason: str | None = None) -> AgentState | None:
        """Approve a paused state to continue execution.

        Args:
            state_id: The state to approve.
            reason: Optional reason for approval.

        Returns:
            The resumed AgentState.
        """
        return self.apply_action(
            state_id,
            HITLAction(action=HITLActionType.APPROVE, reason=reason),
        )

    def revise(self, state_id: str, revised_input: str, reason: str | None = None) -> AgentState | None:
        """Revise a paused state with new input before resuming.

        Args:
            state_id: The state to revise.
            revised_input: The new input to use.
            reason: Optional reason for the revision.

        Returns:
            The revised and resumed AgentState.
        """
        return self.apply_action(
            state_id,
            HITLAction(
                action=HITLActionType.REVISE,
                input=revised_input,
                reason=reason,
            ),
        )

    def cancel(self, state_id: str, reason: str | None = None) -> AgentState | None:
        """Cancel a paused state's execution.

        Args:
            state_id: The state to cancel.
            reason: Optional reason for cancellation.

        Returns:
            The cancelled AgentState.
        """
        return self.apply_action(
            state_id,
            HITLAction(action=HITLActionType.CANCEL, reason=reason),
        )

    def get_all_paused(self) -> list[AgentState]:
        """Get all currently paused states."""
        return [s for s in self._active_states.values() if s.is_paused]

    def _persist_state(self, state: AgentState) -> None:
        """Write state to disk for durability."""
        if not self._persistence_dir:
            return
        filepath = self._persistence_dir / f"{state.state_id}.json"
        filepath.write_text(
            json.dumps(state.to_serializable(), indent=2, default=str),
            encoding="utf-8",
        )
        logger.debug("State persisted to %s", filepath)

    def _load_state(self, state_id: str) -> AgentState | None:
        """Load state from disk."""
        if not self._persistence_dir:
            return None
        filepath = self._persistence_dir / f"{state_id}.json"
        if not filepath.exists():
            return None
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            return AgentState.from_serialized(data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to load state %s: %s", state_id, e)
            return None

    def _cleanup_state(self, state_id: str) -> None:
        """Remove a state from active tracking."""
        self._active_states.pop(state_id, None)
