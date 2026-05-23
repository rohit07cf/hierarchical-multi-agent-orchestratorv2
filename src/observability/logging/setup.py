"""Structured logging via structlog, correlated to traces and requests.

The whole point: every log line is a JSON object that already carries
``request_id``, ``session_id``, ``agent_path``, ``trace_id`` and
``span_id``. That is what lets an engineer paste a trace_id from Grafana
into Loki/CloudWatch and get the exact log lines for that one run —
across all seven agents — with no grep gymnastics.

Design notes:

- **One processor chain, two renderers.** JSON in production (machine
  parseable, ships to Loki/ELK); a coloured console renderer in dev so
  humans can read it. Toggle with ``OBS_LOG_JSON``.
- **stdlib bridge.** The existing code logs via ``logging`` (e.g.
  ``logger.info`` in the agents). We route stdlib records *through*
  structlog's formatter so legacy and new logs share one schema — no
  rewrite required.
- **Context injected by processors, not call sites.** Callers just write
  ``log.info("retrieved docs", count=3)``; the correlation/trace fields
  are merged in automatically. Call sites stay clean.
"""

from __future__ import annotations

import logging
from typing import Any

import structlog

from src.observability.config import ObservabilityConfig
from src.observability.context import get_correlation
from src.observability.tracing import current_trace_ids


def _add_correlation(_logger: Any, _method: str, event: dict) -> dict:
    """structlog processor: merge request/session/agent-path into every line."""
    event.update(get_correlation().as_log_fields())
    return event


def _add_trace_context(_logger: Any, _method: str, event: dict) -> dict:
    """structlog processor: merge active trace_id/span_id for cross-signal joins."""
    event.update(current_trace_ids())
    return event


def setup_logging(config: ObservabilityConfig) -> None:
    """Configure structlog + route stdlib logging through it. Idempotent."""
    level = getattr(logging, config.log_level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_correlation,
        _add_trace_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if config.log_json:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging (the agents' ``logging.getLogger`` calls) through
    # the same structlog renderer so all logs share one JSON schema.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Tame noisy third parties so production logs stay signal-dense.
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger (the app's preferred entry point)."""
    return structlog.get_logger(name)
