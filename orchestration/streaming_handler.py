"""StreamingCallbackHandler — buffers orchestration events for the Streamlit UI.

The orchestrator pushes timeline events here via `push_orchestration_step()`
(and `emit_step()` for raw steps); the Streamlit "Streaming Log" panel renders
`.history`.
"""

from __future__ import annotations

import logging
from collections import deque

from models.streaming_models import StreamingModelResponseStep, StreamingStatus

try:
    from src.observability.metrics import registry as _M
except Exception:  # noqa: BLE001 — observability is optional for the UI bridge
    _M = None

logger = logging.getLogger(__name__)


class StreamingCallbackHandler:
    """Buffers streaming events for the UI.

    The orchestrator pushes events directly via `push_orchestration_step()`;
    `.history` exposes the buffered steps to the Streamlit panel.
    """

    def __init__(self, buffer_size: int = 100) -> None:
        self._buffer: deque[StreamingModelResponseStep] = deque(maxlen=buffer_size)
        self._is_active = True

    @property
    def history(self) -> list[StreamingModelResponseStep]:
        """Buffered history of streaming steps (rendered in Streamlit)."""
        return list(self._buffer)

    def emit_step(self, step: StreamingModelResponseStep) -> None:
        """Synchronously buffer a streaming step (preferred entry point)."""
        if not self._is_active:
            return
        self._buffer.append(step)
        if _M is not None:
            _M.STREAMING_QUEUE_DEPTH.set(len(self._buffer))

    def stop(self) -> None:
        """Stop emitting further events."""
        self._is_active = False

    def resume(self) -> None:
        """Resume event emission."""
        self._is_active = True

    def push_orchestration_step(
        self,
        agent_name: str,
        kind: str,
        message: str,
    ) -> None:
        """Append an orchestration timeline event to the streaming buffer."""
        status_map = {
            "task_decomposition": StreamingStatus.REASONING,
            "subtask_started": StreamingStatus.TOOL_CALLED,
            "subtask_complete": StreamingStatus.TOOL_COMPLETED,
            "orchestration_complete": StreamingStatus.COMPLETED,
            "error": StreamingStatus.ERROR,
            "info": StreamingStatus.IN_PROGRESS,
        }
        if _M is not None:
            _M.STREAMING_EVENTS.labels(kind).inc()
        self.emit_step(
            StreamingModelResponseStep(
                status=status_map.get(kind, StreamingStatus.IN_PROGRESS),
                name=agent_name,
                message_fragment=message,
            )
        )
