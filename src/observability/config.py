"""Observability configuration — env-driven, safe defaults.

A single place that decides *whether* and *how* the three signals
(traces, metrics, logs) are wired up. Everything is opt-in via env vars
so the offline demo and the test suite keep running with zero
observability overhead, while a production deployment turns it on with a
handful of variables.

    OBS_ENABLED=true
    OBS_SERVICE_NAME=hmao-orchestrator
    OBS_TRACE_EXPORTER=otlp|console|none
    OBS_OTLP_ENDPOINT=http://otel-collector:4317
    OBS_METRICS_ENABLED=true
    OBS_METRICS_PORT=9108
    OBS_LOG_JSON=true
    OBS_LOG_LEVEL=INFO
    OBS_TRACE_SAMPLE_RATIO=1.0
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ObservabilityConfig:
    """Resolved observability settings for one process."""

    enabled: bool = False
    service_name: str = "hmao-orchestrator"
    service_version: str = "0.1.0"
    environment: str = "dev"

    # Tracing
    trace_exporter: str = "console"  # otlp | console | none
    otlp_endpoint: str = "http://localhost:4317"
    trace_sample_ratio: float = 1.0

    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = 9108

    # Logging
    log_json: bool = True
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        return cls(
            enabled=_flag("OBS_ENABLED", False),
            service_name=os.environ.get("OBS_SERVICE_NAME", "hmao-orchestrator"),
            service_version=os.environ.get("OBS_SERVICE_VERSION", "0.1.0"),
            environment=os.environ.get("OBS_ENVIRONMENT", "dev"),
            trace_exporter=os.environ.get("OBS_TRACE_EXPORTER", "console"),
            otlp_endpoint=os.environ.get("OBS_OTLP_ENDPOINT", "http://localhost:4317"),
            trace_sample_ratio=float(os.environ.get("OBS_TRACE_SAMPLE_RATIO", "1.0")),
            metrics_enabled=_flag("OBS_METRICS_ENABLED", True),
            metrics_port=int(os.environ.get("OBS_METRICS_PORT", "9108")),
            log_json=_flag("OBS_LOG_JSON", True),
            log_level=os.environ.get("OBS_LOG_LEVEL", os.environ.get("LOG_LEVEL", "INFO")),
        )
