"""Correlation context — the spine that ties every signal together.

A multi-agent run fans work out across the supervisor, managers, and
workers. Because Python `async` tasks share no implicit request scope,
we carry three identifiers through `contextvars` so a log line emitted
deep inside `simple_retriever` can still be joined to the originating
user request:

- ``request_id``  — one orchestration run (1:1 with ``OrchestratorState.state_id``).
- ``session_id``  — a conversation / user session spanning many requests.
- ``agent_path``  — the hierarchical breadcrumb, e.g.
  ``RootSupervisorAgent > BuildManagerAgent > ReviewAgent``.

``contextvars`` propagate correctly across ``await`` boundaries and into
tasks created with ``asyncio.create_task`` *after* the var is set, which
is exactly the propagation semantics a single orchestration needs.
Spans, metric exemplars, and log records all read from here, so the
three pillars stay correlated without threading IDs through every
function signature.
"""

from __future__ import annotations

import contextvars
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "hmao_request_id", default=None
)
_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "hmao_session_id", default=None
)
_agent_path: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "hmao_agent_path", default=()
)


@dataclass(frozen=True)
class Correlation:
    """Immutable snapshot of the current correlation context."""

    request_id: str | None
    session_id: str | None
    agent_path: tuple[str, ...]

    @property
    def agent_path_str(self) -> str:
        return " > ".join(self.agent_path)

    def as_log_fields(self) -> dict[str, str]:
        """Flatten into log/span attributes (skipping empties)."""
        fields: dict[str, str] = {}
        if self.request_id:
            fields["request_id"] = self.request_id
        if self.session_id:
            fields["session_id"] = self.session_id
        if self.agent_path:
            fields["agent_path"] = self.agent_path_str
        return fields


def get_correlation() -> Correlation:
    """Read the current correlation context (cheap, allocation-light)."""
    return Correlation(
        request_id=_request_id.get(),
        session_id=_session_id.get(),
        agent_path=_agent_path.get(),
    )


def new_request_id() -> str:
    return uuid.uuid4().hex


@contextmanager
def correlation_scope(
    *, request_id: str | None = None, session_id: str | None = None
) -> Iterator[Correlation]:
    """Bind a fresh request (and optional session) for the duration of a run.

    Used once at the top of ``orchestrate()``. Tokens are reset on exit so
    nested or sequential runs never leak IDs into one another.
    """
    rid = request_id or new_request_id()
    rt = _request_id.set(rid)
    st = _session_id.set(session_id) if session_id is not None else None
    try:
        yield get_correlation()
    finally:
        _request_id.reset(rt)
        if st is not None:
            _session_id.reset(st)


@contextmanager
def agent_scope(agent_name: str) -> Iterator[tuple[str, ...]]:
    """Push ``agent_name`` onto the hierarchical path for a nested call.

    The push/pop pairing reconstructs the manager → worker tree even
    though execution is a flat series of ``await``s.
    """
    parent = _agent_path.get()
    token = _agent_path.set(parent + (agent_name,))
    try:
        yield _agent_path.get()
    finally:
        _agent_path.reset(token)


def set_session_id(session_id: str) -> None:
    """Bind a session id outside a scope (e.g. from a Streamlit session)."""
    _session_id.set(session_id)
