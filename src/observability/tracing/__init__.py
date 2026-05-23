"""Distributed tracing: tracer setup, span helpers, semantic attributes."""

from src.observability.tracing import attributes
from src.observability.tracing.tracer import (
    OTEL_AVAILABLE,
    SpanKind,
    current_trace_ids,
    get_tracer,
    setup_tracing,
    span,
    sync_span,
)

__all__ = [
    "attributes",
    "OTEL_AVAILABLE",
    "SpanKind",
    "current_trace_ids",
    "get_tracer",
    "setup_tracing",
    "span",
    "sync_span",
]
