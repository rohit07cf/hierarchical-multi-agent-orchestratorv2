"""Prometheus metric definitions — the system's RED + cost + quality signals.

Why these instruments, in production terms:

- **Counters** for things you rate() over time (requests, failures,
  tokens) — they only ever go up, so restarts are obvious.
- **Histograms** for latencies — they give you p50/p95/p99 *and* a
  ``_count``/``_sum`` for free, which is what you alert on. We pick
  buckets sized to *agent* work (sub-second tool calls up to multi-second
  LLM round-trips), because default buckets miss the tail that matters.
- **Gauges** for "right now" values (in-flight runs, queue depth, RSS).

Label cardinality is deliberately bounded: ``agent``, ``manager``,
``tool``, ``model``, ``operation``, ``status`` are all small, closed
sets. We never label by ``request_id`` or raw query text — that is the
classic way to blow up a Prometheus TSDB.

All metrics live on a private ``CollectorRegistry`` so importing this
module never mutates the global default registry (keeps tests hermetic
and avoids duplicate-timeseries errors on reload).
"""

from __future__ import annotations

# Fail-open: if prometheus_client is not installed (e.g. a deploy where
# deps haven't finished installing), fall back to no-op metric objects so
# importing this module — and therefore the whole app — never crashes.
# When the library is present (the normal case) behavior is unchanged.
try:
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the dep
    PROMETHEUS_AVAILABLE = False

    class _NoOpMetric:
        """Mimics Counter/Gauge/Histogram with zero-cost no-op methods."""

        def __init__(self, *args, **kwargs) -> None:
            pass

        def labels(self, *args, **kwargs) -> "_NoOpMetric":
            return self

        def inc(self, *args, **kwargs) -> None:
            pass

        def dec(self, *args, **kwargs) -> None:
            pass

        def observe(self, *args, **kwargs) -> None:
            pass

        def set(self, *args, **kwargs) -> None:
            pass

    class CollectorRegistry:  # type: ignore[no-redef]
        def collect(self):
            return []

    Counter = Gauge = Histogram = _NoOpMetric  # type: ignore[assignment,misc]

# Bucket sets tuned to this workload.
_FAST_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
_LLM_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0)
_ORCH_BUCKETS = (0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 120.0)

REGISTRY = CollectorRegistry()


def _counter(name: str, doc: str, labels: list[str] | None = None) -> Counter:
    return Counter(name, doc, labels or [], registry=REGISTRY)


def _gauge(name: str, doc: str, labels: list[str] | None = None) -> Gauge:
    return Gauge(name, doc, labels or [], registry=REGISTRY)


def _hist(name: str, doc: str, buckets, labels: list[str] | None = None) -> Histogram:
    return Histogram(name, doc, labels or [], buckets=buckets, registry=REGISTRY)


# ----------------------------- Orchestration -----------------------------
ORCHESTRATION_TOTAL = _counter(
    "orchestration_total",
    "Total orchestration runs started, by terminal status.",
    ["status"],
)
ORCHESTRATION_DURATION = _hist(
    "orchestration_duration_seconds",
    "End-to-end orchestration wall-clock latency.",
    _ORCH_BUCKETS,
    ["status"],
)
ORCHESTRATION_FAILURES = _counter(
    "orchestration_failures_total",
    "Orchestration runs that ended in error.",
    ["reason"],
)
ACTIVE_ORCHESTRATIONS = _gauge(
    "active_orchestrations",
    "Orchestration runs currently in flight.",
)
ORCHESTRATION_TURNS = _hist(
    "orchestration_turns",
    "Number of routing turns (manager invocations) per run.",
    (1, 2, 3, 4, 5, 6, 8),
)

# ------------------------------- Supervisor ------------------------------
ROUTING_DECISIONS_TOTAL = _counter(
    "routing_decisions_total",
    "Supervisor next-step decisions, by chosen manager and decision source.",
    ["next_manager", "source"],
)
ROUTING_LATENCY = _hist(
    "routing_latency_seconds",
    "Latency of a single supervisor routing decision.",
    _FAST_BUCKETS,
    ["source"],
)
MANAGER_SELECTION = _counter(
    "manager_selection_total",
    "How often each manager is selected (distribution).",
    ["manager"],
)

# --------------------------------- Agents --------------------------------
AGENT_INVOCATIONS = _counter(
    "agent_invocations_total",
    "Agent.handle() invocations, by agent and hierarchy layer.",
    ["agent", "layer"],
)
AGENT_DURATION = _hist(
    "agent_execution_duration_seconds",
    "Per-agent end-to-end handle() latency.",
    _LLM_BUCKETS,
    ["agent", "layer"],
)
AGENT_FAILURES = _counter(
    "agent_failures_total",
    "Agent invocations that raised or returned success=False.",
    ["agent", "layer"],
)
AGENT_TOOL_SELECTION = _counter(
    "agent_tool_selection_total",
    "Tools an agent selected vs skipped during reasoning.",
    ["agent", "tool", "decision"],  # decision = selected | skipped
)

# ---------------------------------- Tools --------------------------------
TOOL_DURATION = _hist(
    "tool_call_duration_seconds",
    "Deterministic tool execution latency.",
    _FAST_BUCKETS,
    ["tool", "agent"],
)
TOOL_FAILURES = _counter(
    "tool_failures_total",
    "Tool calls that raised.",
    ["tool", "agent"],
)
TOOL_USAGE = _counter(
    "tool_usage_total",
    "Tool invocation distribution.",
    ["tool"],
)

# ----------------------------------- LLM ---------------------------------
LLM_DURATION = _hist(
    "llm_request_duration_seconds",
    "LLM round-trip latency, by model and operation.",
    _LLM_BUCKETS,
    ["model", "operation", "mode"],
)
LLM_TOKENS_INPUT = _counter(
    "llm_tokens_input_total",
    "Prompt tokens consumed.",
    ["model", "operation"],
)
LLM_TOKENS_OUTPUT = _counter(
    "llm_tokens_output_total",
    "Completion tokens produced.",
    ["model", "operation"],
)
LLM_COST_ESTIMATE = _counter(
    "llm_cost_estimate_usd_total",
    "Estimated LLM spend in USD (input+output priced per model).",
    ["model", "operation"],
)
LLM_FAILURES = _counter(
    "llm_failures_total",
    "LLM calls that errored (and fell back to mock).",
    ["model", "operation"],
)

# ----------------------------------- RAG ---------------------------------
RETRIEVAL_DURATION = _hist(
    "retrieval_duration_seconds",
    "Retriever latency.",
    _FAST_BUCKETS,
)
RETRIEVAL_DOCS_RETURNED = _hist(
    "retrieval_documents_returned",
    "Documents returned per retrieval (top-k saturation signal).",
    (0, 1, 2, 3, 5, 8, 13),
)
RETRIEVAL_TOP_SCORE = _hist(
    "retrieval_top_score",
    "Relevance score of the best-matching document [0,1].",
    (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
)
RETRIEVAL_EMPTY = _counter(
    "retrieval_empty_results_total",
    "Retrievals that returned zero useful (score>0) documents.",
)
RETRIEVAL_CONTEXT_CHARS = _hist(
    "retrieval_context_chars",
    "Total characters of retrieved context fed downstream.",
    (0, 500, 1000, 2000, 4000, 8000, 16000),
)

# --------------------------------- Review --------------------------------
REVIEW_FINDINGS = _counter(
    "review_findings_total",
    "Review findings emitted, by severity.",
    ["severity"],
)
SECURITY_FINDINGS = _counter(
    "security_findings_total",
    "Security-category findings emitted, by severity.",
    ["severity"],
)

# -------------------------------- Streaming ------------------------------
STREAMING_EVENTS = _counter(
    "streaming_events_total",
    "Streaming/orchestration events emitted, by kind.",
    ["kind"],
)
STREAMING_QUEUE_DEPTH = _gauge(
    "streaming_queue_depth",
    "Current depth of the streaming event buffer.",
)

# ----------------------------------- HITL --------------------------------
HITL_PAUSED = _counter(
    "paused_orchestrations_total",
    "Orchestrations paused for human review, by checkpoint.",
    ["checkpoint"],
)
HITL_RESUMES = _counter(
    "resume_operations_total",
    "HITL resume operations, by action.",
    ["action"],  # approve | revise | cancel
)

# ---------------------------------- System -------------------------------
PROCESS_MEMORY_BYTES = _gauge(
    "process_memory_rss_bytes",
    "Resident set size of the orchestrator process.",
)
PROCESS_CPU_PERCENT = _gauge(
    "process_cpu_percent",
    "Process CPU utilisation percent (sampled).",
)
EVENT_LOOP_LAG = _hist(
    "event_loop_lag_seconds",
    "Asyncio event-loop scheduling lag (head-of-line blocking signal).",
    _FAST_BUCKETS,
)
