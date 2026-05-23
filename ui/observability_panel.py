"""Live observability dashboard rendered inside the Streamlit UI.

Reads the in-process Prometheus registry directly (see
``src.observability.metrics.snapshot``) so the same numbers Grafana would
show are visible in-app with zero external dependencies. Metrics
accumulate across runs for the life of the Streamlit process, giving a
"since app start" view.
"""

from __future__ import annotations

import streamlit as st

from src.observability.metrics.snapshot import MetricsSnapshot, collect_snapshot


def render_observability_panel() -> None:
    """Render orchestration / agent / tool / LLM / RAG metric panels."""
    snap = collect_snapshot()

    if snap.is_empty:
        st.info(
            "No metrics yet — run a query and they'll populate here. "
            "Metrics are collected in-process; enable the full stack with "
            "`OBS_ENABLED=true` to also export to Prometheus/Grafana/Tempo."
        )
        return

    _render_orchestration(snap)
    st.divider()
    _render_llm_cost(snap)
    st.divider()
    _render_agents(snap)
    st.divider()
    _render_tools_and_rag(snap)

    st.caption(
        "Cumulative since app start, read from the in-process Prometheus "
        "registry. The same series are scraped to Grafana when "
        "`OBS_ENABLED=true`."
    )


# --------------------------------------------------------------------------
def _render_orchestration(snap: MetricsSnapshot) -> None:
    st.subheader("Orchestration")
    by_status = snap.counter_by_label("orchestration_total", "status")
    total = sum(by_status.values())
    completed = by_status.get("completed", 0.0)
    errors = by_status.get("error", 0.0)
    avg_dur = snap.hist_avg("orchestration_duration_seconds")
    avg_turns = snap.hist_avg("orchestration_turns")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total runs", int(total))
    c2.metric(
        "Success rate",
        f"{(completed / total * 100):.0f}%" if total else "—",
    )
    c3.metric("Avg latency", f"{avg_dur * 1000:.0f} ms" if avg_dur else "—")
    c4.metric("Avg turns", f"{avg_turns:.1f}" if avg_turns else "—")

    if errors:
        st.error(f"⚠️ {int(errors)} run(s) ended in error.")

    routing = snap.counter_by_label("routing_decisions_total", "next_manager")
    if routing:
        st.caption("Routing decisions (manager picked per turn)")
        st.bar_chart(routing, horizontal=True, height=180)


def _render_llm_cost(snap: MetricsSnapshot) -> None:
    st.subheader("LLM Cost & Tokens")
    tokens_in = snap.counter_total("llm_tokens_input_total")
    tokens_out = snap.counter_total("llm_tokens_output_total")
    cost = snap.counter_total("llm_cost_estimate_usd_total")
    runs = sum(snap.counter_by_label("orchestration_total", "status").values())
    failures = snap.counter_total("llm_failures_total")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Est. cost", f"${cost:.4f}")
    c2.metric("Cost / run", f"${(cost / runs):.4f}" if runs else "—")
    c3.metric("Input tokens", f"{int(tokens_in):,}")
    c4.metric("Output tokens", f"{int(tokens_out):,}")

    if failures:
        st.warning(f"{int(failures)} LLM call(s) failed and fell back to mock.")

    by_op_in = snap.counter_by_label("llm_tokens_input_total", "operation")
    avg_lat = snap.hist_avg_by_label("llm_request_duration_seconds", "operation")
    if by_op_in or avg_lat:
        rows = []
        for op in sorted(set(by_op_in) | set(avg_lat)):
            rows.append(
                {
                    "Operation": op,
                    "Input tokens": int(by_op_in.get(op, 0.0)),
                    "Avg latency (ms)": round(avg_lat.get(op, 0.0) * 1000, 1),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)


def _render_agents(snap: MetricsSnapshot) -> None:
    st.subheader("Agent Performance")
    invs = snap.counter_by_label("agent_invocations_total", "agent")
    fails = snap.counter_by_label("agent_failures_total", "agent")
    avg_dur = snap.hist_avg_by_label("agent_execution_duration_seconds", "agent")

    if not invs:
        st.info("No agent activity recorded yet.")
        return

    # Preserve hierarchy order for readability.
    order = [
        "RootSupervisorAgent",
        "ResearchManagerAgent",
        "RAGAgent",
        "SummarizerAgent",
        "BuildManagerAgent",
        "CodingAgent",
        "ReviewAgent",
    ]
    names = [a for a in order if a in invs] + [a for a in invs if a not in order]

    rows = []
    for a in names:
        rows.append(
            {
                "Agent": a,
                "Invocations": int(invs.get(a, 0.0)),
                "Failures": int(fails.get(a, 0.0)),
                "Avg duration (ms)": round(avg_dur.get(a, 0.0) * 1000, 2),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def _render_tools_and_rag(snap: MetricsSnapshot) -> None:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Tool Usage")
        usage = snap.counter_by_label("tool_usage_total", "tool")
        fails = snap.counter_by_label("tool_failures_total", "tool")
        if usage:
            rows = [
                {
                    "Tool": t,
                    "Calls": int(c),
                    "Failures": int(fails.get(t, 0.0)),
                }
                for t, c in sorted(usage.items(), key=lambda kv: -kv[1])
            ]
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.info("No tool calls yet.")

    with col_right:
        st.subheader("RAG Quality")
        retrievals = snap.hist_count("retrieval_documents_returned")
        avg_docs = snap.hist_avg("retrieval_documents_returned")
        avg_score = snap.hist_avg("retrieval_top_score")
        empties = snap.counter_total("retrieval_empty_results_total")

        if retrievals:
            c1, c2 = st.columns(2)
            c1.metric("Retrievals", int(retrievals))
            c2.metric("Empty/irrelevant", int(empties))
            c3, c4 = st.columns(2)
            c3.metric("Avg docs", f"{avg_docs:.1f}" if avg_docs is not None else "—")
            c4.metric(
                "Avg top score",
                f"{avg_score:.2f}" if avg_score is not None else "—",
            )
            if empties:
                st.warning(
                    "Empty/irrelevant retrievals detected — possible "
                    "hallucination risk."
                )
        else:
            st.info("No retrievals yet.")

        findings = snap.counter_by_label("review_findings_total", "severity")
        if findings:
            st.caption("Review findings by severity")
            st.bar_chart(findings, height=150)
