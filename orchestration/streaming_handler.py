"""StreamingCallbackHandler — buffers orchestration events for the Streamlit UI.

Originally implemented as an `openai-agents` `AgentHooks` subclass; the
refactored system does not depend on that SDK, so this module degrades to
a plain event buffer when the SDK is unavailable. The Streamlit
"Streaming Log" panel consumes `.history`, which works in both modes.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Any

from models.streaming_models import StreamingModelResponseStep, StreamingStatus

try:
    from src.observability.metrics import registry as _M
except Exception:  # noqa: BLE001 — observability is optional for the UI bridge
    _M = None

logger = logging.getLogger(__name__)

try:
    from agents import AgentHooks  # type: ignore[import-not-found]

    _Base: type = AgentHooks
except ImportError:
    class _Base:  # type: ignore[no-redef]
        """No-op replacement for `AgentHooks` when the SDK is absent."""


class StreamingCallbackHandler(_Base):  # type: ignore[misc, valid-type]
    """Buffers streaming events for the UI.

    When the `openai-agents` SDK is installed this class also satisfies
    the `AgentHooks` interface and can be wired into a Runner. The new
    architecture pushes events directly via `push_orchestration_step()`.
    """

    def __init__(self, buffer_size: int = 100) -> None:
        self._queue: asyncio.Queue[StreamingModelResponseStep] = asyncio.Queue()
        self._buffer: deque[StreamingModelResponseStep] = deque(maxlen=buffer_size)
        self._is_active = True

    @property
    def queue(self) -> asyncio.Queue[StreamingModelResponseStep]:
        """Async queue for consuming streaming updates."""
        return self._queue

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

    async def _emit(self, step: StreamingModelResponseStep) -> None:
        """Async emit, also feeding the queue for live consumers."""
        if not self._is_active:
            return
        self._buffer.append(step)
        try:
            self._queue.put_nowait(step)
        except asyncio.QueueFull:
            logger.warning("Streaming queue full, dropping event")

    def stop(self) -> None:
        """Stop emitting further events."""
        self._is_active = False

    def resume(self) -> None:
        """Resume event emission."""
        self._is_active = True

    # --- AgentHooks interface (only active when openai-agents is installed) ---

    async def on_start(self, context: Any, agent: Any) -> None:
        await self._emit(
            StreamingModelResponseStep(
                status=StreamingStatus.STARTED,
                name=getattr(agent, "name", "agent"),
                message_fragment=f"Agent {getattr(agent, 'name', 'agent')} started",
            )
        )

    async def on_end(self, context: Any, agent: Any, output: Any) -> None:
        await self._emit(
            StreamingModelResponseStep(
                status=StreamingStatus.COMPLETED,
                name=getattr(agent, "name", "agent"),
                message_fragment=f"Agent {getattr(agent, 'name', 'agent')} completed",
                metadata={"output": str(output)[:500]},
            )
        )

    async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
        await self._emit(
            StreamingModelResponseStep.tool_call(
                name=getattr(agent, "name", "agent"),
                tool_name=getattr(tool, "name", "tool"),
                params={},
            )
        )

    async def on_tool_end(
        self, context: Any, agent: Any, tool: Any, result: str
    ) -> None:
        await self._emit(
            StreamingModelResponseStep.tool_result(
                name=getattr(agent, "name", "agent"),
                tool_name=getattr(tool, "name", "tool"),
                result=result[:500] if result else "",
            )
        )

    async def on_handoff(self, context: Any, agent: Any, source: Any) -> None:
        await self._emit(
            StreamingModelResponseStep(
                status=StreamingStatus.IN_PROGRESS,
                name=getattr(agent, "name", "agent"),
                message_fragment=(
                    f"Handoff from {getattr(source, 'name', '?')} "
                    f"to {getattr(agent, 'name', '?')}"
                ),
            )
        )

    # --- Helpers used by the new architecture to push events ---

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
