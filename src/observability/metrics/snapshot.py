"""In-process snapshot of the Prometheus registry — for the live UI panel.

The orchestrator's metrics live on an in-process ``CollectorRegistry``
(``registry.REGISTRY``). Counters/histograms/gauges are updated on every
run *regardless of ``OBS_ENABLED``* — that flag only controls whether the
``/metrics`` HTTP endpoint is served and traces are exported.

That means the Streamlit process already holds live, accumulating metric
state. This module reads it back without a Prometheus server, so the UI
can render an "Observability Metrics" panel out of the box. The same
numbers would appear in Grafana once the stack is running; this is just a
zero-dependency view of them.

The registry persists across Streamlit reruns (same Python process), so
the snapshot is cumulative "since app start".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.observability.metrics.registry import REGISTRY


@dataclass
class MetricsSnapshot:
    """Parsed view of the registry: counters, gauges, histograms by name."""

    counters: dict[str, list[tuple[dict[str, str], float]]] = field(default_factory=dict)
    gauges: dict[str, list[tuple[dict[str, str], float]]] = field(default_factory=dict)
    # name -> list of (labels, count, sum) for histogram aggregates.
    histograms: dict[str, list[tuple[dict[str, str], float, float]]] = field(
        default_factory=dict
    )

    # ---- Counter helpers ----
    def counter_total(self, name: str) -> float:
        return sum(v for _, v in self.counters.get(name, []))

    def counter_by_label(self, name: str, label: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for labels, value in self.counters.get(name, []):
            key = labels.get(label, "")
            out[key] = out.get(key, 0.0) + value
        return out

    # ---- Gauge helpers ----
    def gauge(self, name: str) -> float:
        vals = self.gauges.get(name, [])
        return vals[0][1] if vals else 0.0

    # ---- Histogram helpers ----
    def hist_count(self, name: str) -> float:
        return sum(c for _, c, _ in self.histograms.get(name, []))

    def hist_avg(self, name: str) -> float | None:
        rows = self.histograms.get(name, [])
        total_count = sum(c for _, c, _ in rows)
        total_sum = sum(s for _, _, s in rows)
        return (total_sum / total_count) if total_count else None

    def hist_avg_by_label(self, name: str, label: str) -> dict[str, float]:
        agg: dict[str, list[float]] = {}
        for labels, count, total in self.histograms.get(name, []):
            key = labels.get(label, "")
            bucket = agg.setdefault(key, [0.0, 0.0])
            bucket[0] += count
            bucket[1] += total
        return {k: (s / c) for k, (c, s) in agg.items() if c}

    @property
    def is_empty(self) -> bool:
        # A run has happened iff at least one orchestration was counted. (Keying
        # off agent_invocations_total alone hid the panel when agent-level
        # metrics regressed even though orchestration/LLM data was present.)
        if not (self.counters or self.histograms):
            return True
        return (
            self.counter_total("orchestration_total") == 0
            and self.counter_total("agent_invocations_total") == 0
        )


def collect_snapshot() -> MetricsSnapshot:
    """Walk ``REGISTRY`` once and bucket samples into a friendly snapshot."""
    snap = MetricsSnapshot()
    for metric in REGISTRY.collect():
        if metric.type == "counter":
            rows: list[tuple[dict[str, str], float]] = []
            for sample in metric.samples:
                # prometheus_client emits "<name>_total" for counter values.
                if sample.name.endswith("_total"):
                    rows.append((dict(sample.labels), sample.value))
            if rows:
                snap.counters[metric.name + "_total"] = rows
        elif metric.type == "gauge":
            snap.gauges[metric.name] = [
                (dict(s.labels), s.value) for s in metric.samples
            ]
        elif metric.type == "histogram":
            # Pair up _count and _sum per unique label set.
            counts: dict[tuple, tuple[dict[str, str], float]] = {}
            sums: dict[tuple, float] = {}
            for s in metric.samples:
                labels = {k: v for k, v in s.labels.items() if k != "le"}
                key = tuple(sorted(labels.items()))
                if s.name.endswith("_count"):
                    counts[key] = (labels, s.value)
                elif s.name.endswith("_sum"):
                    sums[key] = s.value
            snap.histograms[metric.name] = [
                (labels, count, sums.get(key, 0.0))
                for key, (labels, count) in counts.items()
            ]
    return snap
