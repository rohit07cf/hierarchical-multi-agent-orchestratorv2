"""Prometheus metrics: registry, cost accounting, system sampling, HTTP server."""

from __future__ import annotations

import logging

from src.observability.metrics import registry
from src.observability.metrics.cost import estimate_cost_usd, estimate_tokens
from src.observability.metrics.registry import REGISTRY
from src.observability.metrics.system import sample_system_metrics

logger = logging.getLogger(__name__)

__all__ = [
    "registry",
    "REGISTRY",
    "estimate_cost_usd",
    "estimate_tokens",
    "sample_system_metrics",
    "start_metrics_server",
]


def start_metrics_server(port: int) -> bool:
    """Expose ``/metrics`` (Prometheus scrape target) on ``port``.

    Returns True if the server started. Failures are logged, not raised —
    a metrics port collision must never crash the orchestrator.
    """
    try:
        from prometheus_client import start_http_server

        start_http_server(port, registry=REGISTRY)
        logger.info("Prometheus metrics server listening on :%d/metrics", port)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not start metrics server on :%d — %s", port, exc)
        return False
