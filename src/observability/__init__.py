"""Observability package — one ``init_observability()`` wires up all three pillars.

Public surface (import from here, not from submodules, where possible):

    from src.observability import init_observability, get_logger
    from src.observability.middleware import observe_agent, observe_tool, observe_llm

Design philosophy
-----------------
- **Zero-overhead when off.** With ``OBS_ENABLED`` unset, no provider is
  installed, ``/metrics`` is not served, and every ``observe_*`` / ``span``
  becomes a no-op (the OTel API returns a no-op tracer; Prometheus
  counters increment cheaply but are never scraped). Tests and the
  offline demo are unaffected.
- **Standards only.** OpenTelemetry (traces) + Prometheus (metrics) +
  structlog (logs). No proprietary agent, no vendor lock-in — point the
  OTLP exporter at Tempo/Jaeger/Grafana Cloud via the collector.
- **Fail-open.** Nothing here may crash the workload. Exporter missing,
  port in use, SDK absent → log and continue.
"""

from __future__ import annotations

# Aliased: this package has a submodule named ``logging``; importing it
# below would otherwise shadow the stdlib module in this namespace.
import logging as _logging

from src.observability.config import ObservabilityConfig
from src.observability.context import (
    Correlation,
    agent_scope,
    correlation_scope,
    get_correlation,
)
from src.observability.logging import get_logger, setup_logging
from src.observability.metrics import start_metrics_server
from src.observability.tracing import setup_tracing, span, sync_span

logger = _logging.getLogger(__name__)

_initialized = False


def init_observability(config: ObservabilityConfig | None = None) -> ObservabilityConfig:
    """Initialise tracing + metrics + logging from config. Idempotent.

    Call once at process start (CLI ``main``, Streamlit bootstrap, or an
    API server's lifespan). Returns the resolved config so callers can
    log/inspect what was enabled.
    """
    global _initialized
    cfg = config or ObservabilityConfig.from_env()

    # Logging is always configured (structured logs are useful even with
    # tracing/metrics off); the heavy exporters are gated by cfg.enabled.
    setup_logging(cfg)

    if _initialized:
        return cfg

    if cfg.enabled:
        setup_tracing(cfg)
        if cfg.metrics_enabled:
            start_metrics_server(cfg.metrics_port)
        logger.info(
            "Observability initialised: service=%s env=%s tracing=%s metrics=:%d",
            cfg.service_name,
            cfg.environment,
            cfg.trace_exporter,
            cfg.metrics_port,
        )
    else:
        logger.info("Observability disabled (set OBS_ENABLED=true to enable)")

    _initialized = True
    return cfg


__all__ = [
    "ObservabilityConfig",
    "Correlation",
    "init_observability",
    "get_logger",
    "get_correlation",
    "correlation_scope",
    "agent_scope",
    "span",
    "sync_span",
]
