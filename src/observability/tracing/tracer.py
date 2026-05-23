"""Tracer setup + the single ``span()`` helper used everywhere.

Design choices worth defending in an interview:

- **API-only by default.** We always call ``opentelemetry.trace.get_tracer``.
  Until ``setup_tracing()`` installs a real ``TracerProvider``, the OTel
  API returns a *no-op* tracer — spans cost virtually nothing and never
  raise. That is what lets the offline demo and the test suite run with
  zero observability wiring and zero new failure modes.
- **One span helper.** Every instrumentation site uses the same async
  context manager, so correlation IDs, error tagging, and status codes
  are applied identically — no copy-paste drift.
- **No vendor SDK.** Export is OTLP (the open standard). Swap Tempo,
  Jaeger, or Grafana Cloud behind the collector without touching code.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Iterator

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from src.observability.config import ObservabilityConfig
from src.observability.context import get_correlation
from src.observability.tracing import attributes as A

logger = logging.getLogger(__name__)

_INSTRUMENTATION_NAME = "hmao.observability"


def get_tracer() -> trace.Tracer:
    return trace.get_tracer(_INSTRUMENTATION_NAME)


def setup_tracing(config: ObservabilityConfig) -> None:
    """Install a real TracerProvider. Idempotent and import-guarded.

    Called once from ``init_observability()``. If the SDK pieces are not
    installed we log and leave the no-op tracer in place rather than
    crashing the app — observability must never take down the workload.
    """
    if not config.enabled or config.trace_exporter == "none":
        return
    # Don't clobber a provider another process component already set.
    if not isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider):
        return

    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )
        from opentelemetry.sdk.trace.sampling import (
            ParentBased,
            TraceIdRatioBased,
        )
    except ImportError:  # SDK not installed — stay on the no-op tracer.
        logger.warning("opentelemetry-sdk not installed; tracing disabled")
        return

    resource = Resource.create(
        {
            "service.name": config.service_name,
            "service.version": config.service_version,
            "deployment.environment": config.environment,
        }
    )
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(TraceIdRatioBased(config.trace_sample_ratio)),
    )

    if config.trace_exporter == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter: Any = OTLPSpanExporter(endpoint=config.otlp_endpoint, insecure=True)
        except ImportError:
            logger.warning("OTLP exporter missing; falling back to console")
            exporter = ConsoleSpanExporter()
    else:
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    logger.info(
        "Tracing enabled: exporter=%s endpoint=%s sample=%.2f",
        config.trace_exporter,
        config.otlp_endpoint,
        config.trace_sample_ratio,
    )


def _apply_correlation(span: Span) -> None:
    """Stamp the active correlation IDs onto a span for cross-signal joins."""
    if not span.is_recording():
        return
    corr = get_correlation()
    if corr.request_id:
        span.set_attribute(A.REQUEST_ID, corr.request_id)
    if corr.session_id:
        span.set_attribute(A.SESSION_ID, corr.session_id)
    if corr.agent_path:
        span.set_attribute(A.AGENT_PATH, corr.agent_path_str)


@asynccontextmanager
async def span(
    name: str,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
) -> AsyncIterator[Span]:
    """Async span context manager with uniform correlation + error tagging.

    On an exception the span records it, sets ERROR status, and re-raises
    — so a failed run is visible in the trace *and* still propagates.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name, kind=kind) as sp:
        _apply_correlation(sp)
        if attributes:
            for k, v in attributes.items():
                if v is not None:
                    sp.set_attribute(k, v)
        try:
            yield sp
        except Exception as exc:  # noqa: BLE001 — record then re-raise
            sp.record_exception(exc)
            sp.set_status(Status(StatusCode.ERROR, str(exc)))
            sp.set_attribute("hmao.error.type", type(exc).__name__)
            raise


@contextmanager
def sync_span(
    name: str,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Span]:
    """Synchronous twin of ``span()`` for non-async code paths."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name, kind=kind) as sp:
        _apply_correlation(sp)
        if attributes:
            for k, v in attributes.items():
                if v is not None:
                    sp.set_attribute(k, v)
        try:
            yield sp
        except Exception as exc:  # noqa: BLE001
            sp.record_exception(exc)
            sp.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def current_trace_ids() -> dict[str, str]:
    """Return ``trace_id``/``span_id`` of the active span as hex (for logs)."""
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return {}
    return {
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
    }
