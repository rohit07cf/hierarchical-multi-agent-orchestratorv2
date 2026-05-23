"""Distributed tracing: tracer setup, span helpers, semantic attributes."""

from src.observability.tracing import attributes
from src.observability.tracing.tracer import (
    current_trace_ids,
    get_tracer,
    setup_tracing,
    span,
    sync_span,
)

__all__ = [
    "attributes",
    "current_trace_ids",
    "get_tracer",
    "setup_tracing",
    "span",
    "sync_span",
]
