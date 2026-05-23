# Observability Architecture

Production-grade observability for the Hierarchical Multi-Agent
Orchestrator: **OpenTelemetry traces + Prometheus metrics + structured
JSON logs**, correlated by a single `request_id` and a hierarchical
`agent_path`. No vendor SDKs, no lock-in — point the OTLP exporter at
Tempo, Jaeger, or Grafana Cloud and the app never changes.

> **Zero-overhead when off.** Everything is gated by `OBS_ENABLED`. With
> it unset, the OTel API returns a no-op tracer, `/metrics` isn't served,
> and the offline demo + test suite run exactly as before.

---

## 1. Architecture Analysis

### The system, as an observability target

```
User Query
  └── RootSupervisorAgent           (plan → route loop → aggregate)
       ├── ResearchManagerAgent      (decides: RAG? Summarize? both?)
       │     ├── RAGAgent            → simple_retriever / load_kb
       │     └── SummarizerAgent     (pure synthesis)
       └── BuildManagerAgent         (decides: code? review? both?)
             ├── CodingAgent         → code_generation / template / file_context
             └── ReviewAgent         → code_review / security / test_gap
```

Every agent is a `ReasoningAgent` running the same 3-step contract:
**reason → execute tools → synthesize**. Execution is `async`,
sequential, and dynamically routed turn-by-turn by the supervisor.

### 1. Signals that already existed

| Signal | Where | Gap |
|---|---|---|
| `logging.info` per phase | `ReasoningAgent._log` | Plain text, no correlation, no trace IDs |
| `AgentTrace` (reasoning, tools, invocations) | `models/trace.py` | Rich but **in-process only** — dies with the request |
| `OrchestratorState.steps` timeline | `state_models.py` | Great for the UI; not exported, not queryable |
| `StreamingCallbackHandler` events | `orchestration/` | UI buffer only |

**Conclusion:** the system was *introspectable in the UI* but **blind in
production** — no cross-request aggregation, no latency/cost/error
metrics, no distributed trace, no way to debug a run after it finished.

### 2. Missing observability gaps

- No **trace** tying the 7 agents of one run together.
- No **latency breakdown** (which agent / tool / LLM call is slow?).
- No **cost or token** visibility — the #1 blind spot in LLM systems.
- No **failure analytics** — failures were logged and swallowed.
- No **RAG quality** signals (empty retrievals, low relevance).
- No **routing analytics** — loops and mis-routes were invisible.

### 3. Where instrumentation lives (and why)

| Boundary | File | Instrument |
|---|---|---|
| Orchestration lifecycle | `supervisor.orchestrate` | root span + lifecycle metrics |
| Routing decision | `supervisor._decide_next_manager` | span + routing metrics |
| Aggregation | `supervisor._aggregate` | span |
| Subtask dispatch | `execution_engine.run_one_task` | span |
| Agent reason→exec→synth | `agents/base.handle` | agent span + metrics |
| Tool call | `agents/base._call_tool` | tool span + metrics |
| LLM round-trip | `llm/client.reason/synthesize` | LLM span + token/cost |
| Retrieval | `rag_agent._tool_retrieve` | RAG span + quality metrics |
| Review findings | `review_agent._extra_data` | findings metrics |
| Streaming/HITL | `streaming_handler`, `supervisor` | event/pause metrics |

**Why these:** they are the *only* places control crosses an
abstraction boundary. Instrumenting at the boundary (not inside business
logic) keeps the agents clean and the spans meaningful.

### 4–6. Critical boundaries

- **Execution boundary** — `agent.handle()`: the unit of work that
  succeeds or fails.
- **Handoff boundary** — manager → worker delegation (a worker `handle`
  invoked as a "tool"). The span tree captures the parent-child link.
- **Context-propagation boundary** — the `await` chain. `contextvars`
  carry `request_id` / `session_id` / `agent_path` across every `await`
  without threading args.

### 7. Tracing strategy

One trace per orchestration run. The root span is opened in
`orchestrate()`; the **OTel context propagates implicitly** through
`async` calls, so every nested agent/tool/LLM span auto-parents. The
hierarchical `agent_path` contextvar gives the *same* tree as a flat,
queryable string attribute (`ResearchManagerAgent > RAGAgent`).

### 8. Production failure points → what catches them

| Failure | Caught by |
|---|---|
| Routing loop (same manager twice / never finishing) | `orchestration_turns` p95, routing span tree |
| Stuck orchestration | `active_orchestrations` gauge climbing, no span end |
| Cost blowout | `llm_cost_estimate_usd_total` by model |
| Silent RAG failure (empty / irrelevant) | `retrieval_empty_results_total`, `retrieval_top_score` |
| Tool regression | `tool_failures_total` by tool |
| Event-loop starvation (blocking call) | `event_loop_lag_seconds` |
| LLM provider degradation | `llm_failures_total`, `llm_request_duration_seconds` |

### 9–11. Spans vs Metrics vs Logs

- **Spans** — anything with a *parent and a duration*: orchestration,
  routing, subtask, agent, tool, LLM, retrieval, aggregation.
- **Metrics** — anything you *aggregate or alert on over time*: rates,
  latencies (histograms), counts, tokens, cost, gauges.
- **Logs** — *discrete events with detail*: reasoning summaries, errors,
  retries, routing rationale — each stamped with trace + correlation IDs.

The three are joined by `trace_id` (logs↔traces) and shared labels
(traces↔metrics), so one click in Grafana pivots across all three.

---

## 2. Observability Design

```
                ┌─────────────────────────────────────────┐
                │           Orchestrator process            │
                │                                           │
   contextvars  │   correlation: request_id / session_id /  │
   (the spine)  │                agent_path                 │
                │        │            │            │        │
                │     ┌──▼──┐     ┌───▼───┐    ┌───▼────┐    │
                │     │spans│     │metrics│    │ logs   │    │
                │     │ OTel│     │ Prom  │    │structlog│   │
                │     └──┬──┘     └───┬───┘    └───┬────┘    │
                └────────┼────────────┼────────────┼─────────┘
                    OTLP │      :9108 │ scrape     │ stdout (JSON)
                         ▼            ▼            ▼
                  ┌────────────┐  ┌──────────┐  ┌──────┐
                  │OTel Collect│  │Prometheus│  │ Loki │
                  └─────┬──────┘  └────┬─────┘  └──┬───┘
                        ▼              │           │
                     ┌──────┐          │           │
                     │ Tempo│          │           │
                     └──┬───┘          │           │
                        └──────► Grafana ◄─────────┘
                          (traces + metrics + logs, correlated)
```

**Package layout** (`src/observability/`):

```
observability/
  config.py            # env-driven ObservabilityConfig
  __init__.py          # init_observability() — wires all 3 pillars
  context/             # contextvars: request/session/agent_path
  tracing/             # tracer setup, span() helper, attribute constants
  metrics/             # Prometheus registry, cost model, system sampler
  logging/             # structlog + trace-correlation processors
  middleware/          # observe_agent / observe_tool / observe_llm / observe_retrieval
  dashboards/          # prometheus.yml, otel-collector.yaml, tempo.yaml, grafana/
```

---

## 3. Tracing Implementation

### Span hierarchy (one real run)

```
orchestration                              [SERVER]  request_id, query_preview, status, turns
├── supervisor.route_decision              turn=0, next_manager, source=llm|router_fallback
├── engine.run_task  (ResearchManagerAgent)
│   └── agent.handle (ResearchManagerAgent) layer=manager, selected_tools, success
│       ├── llm.request   operation=reason   tokens_in/out, cost_usd, mode
│       ├── tool.call     call_rag_agent
│       │   └── agent.handle (RAGAgent)       layer=worker
│       │       ├── llm.request reason
│       │       ├── tool.call simple_retriever
│       │       │   └── rag.retrieve          docs_returned, top_score, context_chars, empty
│       │       └── llm.request synthesize
│       ├── tool.call     call_summarizer_agent
│       │   └── agent.handle (SummarizerAgent)
│       └── llm.request   operation=synthesize
├── supervisor.route_decision              turn=1, next_manager=BuildManagerAgent
├── engine.run_task  (BuildManagerAgent)
│   └── agent.handle (BuildManagerAgent)
│       ├── tool.call call_coding_agent → agent.handle (CodingAgent) → tool.call code_generation_tool
│       └── tool.call call_review_agent → agent.handle (ReviewAgent) → 3× tool.call (review/security/test_gap)
├── supervisor.route_decision              turn=2 → finish
└── supervisor.aggregate                   inputs=2
```

### How it's wired

`init_observability()` installs a `TracerProvider` with a
`ParentBased(TraceIdRatioBased)` sampler and a `BatchSpanProcessor` →
OTLP. Every call site uses **one** async helper so correlation + error
tagging are identical everywhere:

```python
# src/observability/tracing/tracer.py
@asynccontextmanager
async def span(name, *, kind=SpanKind.INTERNAL, attributes=None):
    with get_tracer().start_as_current_span(name, kind=kind) as sp:
        _apply_correlation(sp)              # request/session/agent_path
        for k, v in (attributes or {}).items():
            if v is not None: sp.set_attribute(k, v)
        try:
            yield sp
        except Exception as exc:
            sp.record_exception(exc)
            sp.set_status(Status(StatusCode.ERROR, str(exc)))
            raise                            # record, then propagate
```

Agents never call `span()` directly — they call the `observe_*`
middleware, which adds the matching metrics in the same `finally`.

**Correlation IDs:** `request_id == OrchestratorState.state_id` (bound in
`correlation_scope` at the top of `orchestrate`). `agent_path` is pushed
in `agent_scope` inside `observe_agent`, so it reconstructs the tree even
across `await`.

---

## 4. Metrics Implementation

All metrics live on a private `CollectorRegistry` (hermetic, no global
mutation). Buckets are tuned to *agent* work, not web requests.

| Metric | Type | Why it matters / what it diagnoses |
|---|---|---|
| `orchestration_total{status}` | counter | Throughput + error ratio (SLO numerator/denominator) |
| `orchestration_duration_seconds{status}` | histogram | End-to-end p95/p99 — the user-facing latency SLO |
| `orchestration_failures_total{reason}` | counter | Crash-class breakdown for paging |
| `active_orchestrations` | gauge | Concurrency / stuck-run detection |
| `orchestration_turns` | histogram | **Routing-loop early warning** (creeps toward MAX_TURNS=6) |
| `routing_decisions_total{next_manager,source}` | counter | Mis-routing + LLM-vs-fallback ratio |
| `routing_latency_seconds{source}` | histogram | Planning overhead |
| `manager_selection_total{manager}` | counter | Workload distribution across managers |
| `agent_invocations_total{agent,layer}` | counter | Per-agent traffic |
| `agent_execution_duration_seconds{agent,layer}` | histogram | **Which agent is the bottleneck** |
| `agent_failures_total{agent,layer}` | counter | Per-agent reliability |
| `agent_tool_selection_total{agent,tool,decision}` | counter | Tool selection vs skip behavior drift |
| `tool_call_duration_seconds{tool,agent}` | histogram | Slow deterministic tools |
| `tool_failures_total{tool,agent}` | counter | Tool regressions |
| `tool_usage_total{tool}` | counter | Tool popularity / dead tools |
| `llm_request_duration_seconds{model,operation,mode}` | histogram | Provider latency by op |
| `llm_tokens_input_total{model,operation}` | counter | Prompt-size growth |
| `llm_tokens_output_total{model,operation}` | counter | Generation volume |
| `llm_cost_estimate_usd_total{model,operation}` | counter | **$ spend — the finance question** |
| `llm_failures_total{model,operation}` | counter | Provider degradation / fallback rate |
| `retrieval_duration_seconds` | histogram | Retriever latency |
| `retrieval_documents_returned` | histogram | Top-k saturation |
| `retrieval_top_score` | histogram | **Relevance quality** (silent-failure signal) |
| `retrieval_empty_results_total` | counter | **Empty/irrelevant retrievals** (hallucination risk) |
| `retrieval_context_chars` | histogram | Context-window pressure |
| `review_findings_total{severity}` | counter | Code-quality trend |
| `security_findings_total{severity}` | counter | Security-regression trend |
| `streaming_events_total{kind}` | counter | UI event throughput |
| `streaming_queue_depth` | gauge | Backpressure / consumer lag |
| `paused_orchestrations_total{checkpoint}` | counter | HITL pause rate |
| `resume_operations_total{action}` | counter | HITL approve/revise/cancel mix |
| `process_memory_rss_bytes` | gauge | Leak detection |
| `process_cpu_percent` | gauge | Saturation |
| `event_loop_lag_seconds` | histogram | **Async starvation (blocking call detector)** |

### Cost model (`metrics/cost.py`)

Per-model `$/1K tokens` table; real provider `usage` is preferred, with a
`~4 chars/token` heuristic fallback for mock mode / missing usage.
Unknown models price at `$0` but stay labeled so the gap is visible.

---

## 5. Logging Implementation

`structlog` emits JSON; **stdlib `logging` is routed through the same
formatter** so the agents' existing `logger.info` calls gain the schema
for free.

### Log schema (every line)

```json
{
  "timestamp": "2026-05-23T04:07:18.117Z",
  "level": "info",
  "event": "[RAGAgent] Tool simple_retriever [ok]: {...}",
  "request_id": "da55aa23-...",
  "session_id": "sess-1",
  "agent_path": "ResearchManagerAgent > RAGAgent",
  "trace_id": "aa2a3091...",   // present when a span is active
  "span_id":  "75d927f2..."
}
```

### Levels
- `DEBUG` — tool args, retrieved doc bodies (sampled).
- `INFO` — phase transitions, routing decisions, tool ok/err.
- `WARNING` — LLM fallback to mock, queue full, retries.
- `ERROR` — agent/tool exceptions (with stack via `format_exc_info`).

### Sampling strategy
- Logs: emit all WARN/ERROR; sample high-volume DEBUG (e.g. doc bodies).
- Traces: `ParentBased(TraceIdRatioBased)` — sample at the root, children
  inherit, so traces are never half-captured. Tail-sample errors/slow
  runs at the collector (keep the traces that matter).

### Production pitfalls (addressed)
- **PII / prompt leakage** → we log *previews* (≤200 chars) and never the
  full prompt at INFO.
- **Cardinality** → IDs go in log fields, never in metric labels.
- **Log-driven latency** → JSON serialization is cheap; heavy payloads
  are previews, not full objects.

---

## 6. Dashboard Design

### In-app panel (zero external dependencies)

The Streamlit UI ships a **📊 Observability Metrics** expander
([ui/observability_panel.py](../ui/observability_panel.py)) that reads the
in-process Prometheus registry directly via
[`metrics/snapshot.py`](../src/observability/metrics/snapshot.py) — no
Prometheus/Grafana required. It surfaces orchestration health (runs,
success rate, avg latency, avg turns, routing distribution), LLM cost &
tokens, per-agent performance, tool usage, and RAG quality. Metrics
accumulate for the life of the Streamlit process ("since app start").
This is the same data Grafana scrapes; the panel is just a dependency-free
view of it for demos.

> Metric *counters* update on every run regardless of `OBS_ENABLED` —
> that flag only gates the `/metrics` endpoint and trace export — which
> is why the panel works out of the box.

### Grafana dashboards

Two are shipped as importable JSON (`grafana/dashboards/`); all ten are
specified here with PromQL.

### 1. Orchestration Overview *(shipped)*
- **Active orchestrations** — `active_orchestrations`
- **Run rate** — `sum(rate(orchestration_total[5m]))*60`
- **Error %** — `100*sum(rate(orchestration_total{status="error"}[5m]))/clamp_min(sum(rate(orchestration_total[5m])),0.001)`
- **Latency p50/95/99** — `histogram_quantile(0.95, sum(rate(orchestration_duration_seconds_bucket[5m])) by (le))`
- **Turns p95** — `histogram_quantile(0.95, sum(rate(orchestration_turns_bucket[5m])) by (le))`
- *Learn:* throughput, health, and routing-loop onset at a glance.

### 2. Agent Performance
- **Latency p95 by agent** — `histogram_quantile(0.95, sum(rate(agent_execution_duration_seconds_bucket[5m])) by (le, agent))`
- **Invocations by layer** — `sum(rate(agent_invocations_total[5m])) by (layer)`
- **Failure rate by agent** — `sum(rate(agent_failures_total[5m])) by (agent) / sum(rate(agent_invocations_total[5m])) by (agent)`
- *Learn:* which agent is the bottleneck / least reliable.

### 3. Tool Performance
- **Tool latency p95** — `histogram_quantile(0.95, sum(rate(tool_call_duration_seconds_bucket[5m])) by (le, tool))`
- **Tool failure rate** — `sum(rate(tool_failures_total[5m])) by (tool)`
- **Usage distribution** — `sum(rate(tool_usage_total[5m])) by (tool)`
- *Learn:* dead tools, slow tools, flaky tools.

### 4. LLM Cost & Tokens *(shipped)*
- **Spend rate $/hr** — `sum(rate(llm_cost_estimate_usd_total[5m]))*3600`
- **Cost per run** — `sum(rate(llm_cost_estimate_usd_total[5m]))/clamp_min(sum(rate(orchestration_total[5m])),0.001)`
- **Tokens/sec in vs out** — `sum(rate(llm_tokens_input_total[5m])) by (operation)`
- **LLM p95 latency** — `histogram_quantile(0.95, sum(rate(llm_request_duration_seconds_bucket[5m])) by (le, operation))`

### 5. RAG Quality Signals
- **Empty-retrieval rate** — `sum(rate(retrieval_empty_results_total[5m]))`
- **Top-score distribution (heatmap)** — `sum(rate(retrieval_top_score_bucket[5m])) by (le)`
- **Docs returned p50** — `histogram_quantile(0.5, sum(rate(retrieval_documents_returned_bucket[5m])) by (le))`
- **Context size p95** — `histogram_quantile(0.95, sum(rate(retrieval_context_chars_bucket[5m])) by (le))`
- *Learn:* silent retrieval failure → hallucination risk.

### 6. Failure Analysis
- **Failures by class** — `sum(rate(orchestration_failures_total[5m])) by (reason)`
- **Agent vs tool vs LLM failures** — overlay the three `*_failures_total` rates
- **LLM fallback rate** — `sum(rate(llm_failures_total[5m])) by (model)`
- *Learn:* where breakage originates in the stack.

### 7. Streaming / Event Health
- **Event rate by kind** — `sum(rate(streaming_events_total[5m])) by (kind)`
- **Queue depth** — `streaming_queue_depth`

### 8. HITL Monitoring
- **Pause rate by checkpoint** — `sum(rate(paused_orchestrations_total[5m])) by (checkpoint)`
- **Resume action mix** — `sum(rate(resume_operations_total[5m])) by (action)`
- *Learn:* how often humans intervene and how (approve/revise/cancel).

### 9. Supervisor Decision Analytics
- **Decision distribution** — `sum(rate(routing_decisions_total[5m])) by (next_manager)`
- **LLM vs fallback ratio** — `sum(rate(routing_decisions_total[5m])) by (source)`
- **Routing latency p95** — `histogram_quantile(0.95, sum(rate(routing_latency_seconds_bucket[5m])) by (le, source))`

### 10. Trace Correlation
- A Tempo panel + a logs panel keyed on `request_id`. Paste a
  `request_id` (or click a span) → see the full trace and all log lines
  for that one run across all 7 agents.

---

## 7. Multi-Agent Trace Design

**Reconstruct a full run:** filter Tempo by `hmao.request_id`. The span
tree *is* the orchestration lifecycle (see §3).

**Debug a failed run:** the failed span carries `status=ERROR`,
`hmao.error.type`, and a recorded exception; its ancestors show exactly
which manager/worker/tool it sat under. Cross to logs via `trace_id`.

**Debug a stuck run:** `active_orchestrations` stays elevated and the
root span never ends. The last open child span names the agent/tool that
hung; `event_loop_lag_seconds` reveals if it's blocking the loop.

**Debug a routing loop:** `orchestration_turns` p95 rises toward 6, and
the trace shows repeated `supervisor.route_decision → engine.run_task`
pairs for the same manager (which the router guards against, but the LLM
path can still misbehave). `routing_decisions_total{source}` shows
whether the LLM or the fallback made the call.

---

## 8. RAG Observability

### Why RAG is different from a normal API
A normal API call is correct or it errors. A retrieval **succeeds (HTTP
200, low latency) while being useless** — wrong or empty documents. The
status code lies. So RAG needs *quality* signals, not just RED metrics:

- **Empty / irrelevant** — `retrieval_empty_results_total`,
  `retrieval_top_score` (we flag empty when `docs==0` *or* `top_score<=0`,
  catching top-k padding that hides a miss).
- **Saturation** — `retrieval_documents_returned` near `top_k` every time
  means you may be truncating relevant context.
- **Context pressure** — `retrieval_context_chars` trending up predicts
  token-cost growth and context-window overflow.
- **Embedding-drift placeholder** — score distribution shifting downward
  over time is the cheap proxy for drift before a real embedding store
  exists. (Swap `SimpleRetriever` for a vector store; the span/metric
  contract is unchanged.)

### Hallucination-correlation signal
Join `retrieval_top_score` (low) with downstream review findings or user
thumbs-down on the same `request_id`: **low relevance + confident answer
= likely hallucination.** The trace makes that join one query.

---

## 9. File Structure

```
src/observability/
├── config.py                    # ObservabilityConfig.from_env()
├── __init__.py                  # init_observability(), public API
├── context/
│   └── correlation.py           # contextvars + correlation_scope / agent_scope
├── tracing/
│   ├── attributes.py            # span names + semantic attribute keys
│   └── tracer.py                # setup_tracing(), span(), sync_span()
├── metrics/
│   ├── registry.py              # all Prometheus instruments
│   ├── cost.py                  # token + USD estimation
│   ├── system.py                # RSS/CPU/event-loop-lag sampler
│   └── __init__.py              # start_metrics_server()
├── logging/
│   └── setup.py                 # structlog config + correlation processors
├── middleware/
│   └── instrument.py            # observe_agent / observe_tool / observe_llm / observe_retrieval
└── dashboards/
    ├── prometheus.yml
    ├── otel-collector.yaml
    ├── tempo.yaml
    └── grafana/{provisioning,dashboards}/
```

Instrumented application files (minimal, boundary-only edits):
`orchestrator/supervisor.py`, `orchestrator/execution_engine.py`,
`agents/base.py`, `agents/rag_agent.py`, `agents/review_agent.py`,
`llm/client.py`, `orchestration/streaming_handler.py`, `main.py`,
`examples/demo_queries.py`.

---

## 10. Production-Readiness Improvements

**Implementation roadmap**

1. **MVP (quick wins, ~1 file each)**
   - Structured JSON logs with `request_id` + `agent_path` *(done)*.
   - Root orchestration span + `orchestration_total/duration` *(done)*.
   - `/metrics` endpoint *(done)*.
2. **Core**
   - Agent/tool/LLM spans + metrics, cost model *(done)*.
   - RAG quality + routing analytics *(done)*.
   - Grafana dashboards + Tempo correlation *(done)*.
3. **Hardening (future)**
   - Tail sampling at the collector (keep errors/slow, sample the rest).
   - Alerting rules (SLO burn, cost budget, empty-retrieval spike).
   - Exemplars linking Prometheus histograms → trace IDs.
   - `OrchestratorState` persistence + "rerun from step N".
   - Real embedding store → embedding-drift metric replaces the placeholder.
   - Per-tenant `session_id` cost attribution.

**Minimal viable observability:** logs (correlated) + orchestration RED +
LLM cost. Three signals answer "is it up, is it fast, what does it cost?"

---

## 11. Interview Talking Points

- **"Why OTel + Prometheus + structlog?"** Open standards, zero
  lock-in. The OTLP collector is the seam — swap Tempo/Jaeger/Grafana
  Cloud without touching app code. Proprietary agents (Datadog/New Relic)
  would couple the code to a vendor.
- **"How do you correlate three signals?"** One `request_id` (==
  `state_id`) in `contextvars`, injected into spans, metric context, and
  every log line; plus `trace_id`/`span_id` on logs. One click pivots
  across all three in Grafana.
- **"Why instrument at boundaries, not inside agents?"** Spans are only
  meaningful at unit-of-work edges. Boundary instrumentation via
  `observe_*` middleware keeps business logic clean and guarantees span +
  metric + log stay consistent (set in the same `finally`).
- **"How is this zero-overhead when off?"** The OTel *API* returns a
  no-op tracer until a provider is installed; metrics increment cheaply
  but aren't scraped. Gated by `OBS_ENABLED`. Tests prove the offline
  path is unchanged (39/39 pass).
- **"What if the observability libs aren't installed?"** Every core dep
  is import-guarded, so the package degrades to no-op telemetry and the
  app still imports and runs. This matters on platforms (e.g. Streamlit
  Community Cloud) that run the app once *before* installing newly-added
  requirements — without the guards, that pre-install window throws
  import tracebacks; with them, the app starts clean and lights up
  telemetry once the deps land. Verified by blocking all four libs and
  running a full orchestration.
- **"What's special about LLM/RAG observability?"** Cost is a
  first-class metric (per model/operation, real usage preferred). RAG
  needs *quality* signals because a retrieval can succeed and still be
  useless — `retrieval_empty_results`/`top_score` catch the silent
  failure that drives hallucination.
- **"How do you debug a routing loop / stuck run?"** `orchestration_turns`
  p95 + the routing span tree for loops; `active_orchestrations` + the
  last open span for stuck runs; `event_loop_lag` to confirm blocking.
- **Tradeoffs considered:** in-process Prometheus client (simple, but a
  multi-process WSGI deploy needs the pushgateway/multiprocess mode);
  char-heuristic tokens in mock mode (approximate, but real usage wins
  when present); cardinality discipline (IDs in logs/spans, never labels).

---

## 12. Final Architecture Diagram

```
┌──────────────────────── one orchestration run ────────────────────────┐
│ correlation_scope(request_id = state_id)                                │
│                                                                         │
│  orchestrate ──root span──► route ──► run_task ──► agent.handle ──┐     │
│      │  (metrics: active, total, duration, turns)                 │     │
│      │                                                            ▼     │
│      │                                            reason→tools→synth    │
│      │                                            (LLM span+cost,       │
│      │                                             tool span, RAG span) │
│      └──► aggregate ──► final answer                                    │
│                                                                         │
│  contextvars carry request_id / session_id / agent_path through await   │
└─────────────────────────────────────────────────────────────────────┬─┘
                                                                        │
   spans (OTLP)            metrics (:9108 scrape)        logs (stdout)  │
        │                          │                          │         │
        ▼                          ▼                          ▼         │
  OTel Collector ──► Tempo    Prometheus                     Loki       │
        └───────────────────────┴──────────► Grafana ◄─────────┘        │
                       (correlated traces • metrics • logs)             │
```

### Run it

```bash
# 1. Bring up the stack
docker compose -f docker-compose.observability.yml up -d
#    Grafana :3000  Prometheus :9090  Tempo :3200  Collector :4317

# 2. Run the app with observability on
OBS_ENABLED=true OBS_TRACE_EXPORTER=otlp \
OBS_OTLP_ENDPOINT=http://localhost:4317 \
streamlit run main.py

# Or the CLI demo with console spans:
OBS_ENABLED=true OBS_TRACE_EXPORTER=console python -m src.examples.demo_queries

# Offline / tests — unchanged, observability is a no-op:
pytest
```
