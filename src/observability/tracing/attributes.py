"""Semantic attribute & span-name constants.

Centralising the keys keeps span attributes consistent across every
instrumentation site, which is what makes traces queryable. We follow
the OpenTelemetry GenAI semantic conventions where they exist
(``gen_ai.*``) and namespace everything else under ``hmao.*`` (the
project's own domain) so dashboards can filter on stable keys instead of
ad-hoc strings.
"""

from __future__ import annotations

# ---- Span names (the hierarchy a trace renders as) ----
SPAN_ORCHESTRATION = "orchestration"
SPAN_ROUTING = "supervisor.route_decision"
SPAN_AGGREGATION = "supervisor.aggregate"
SPAN_SUBTASK = "engine.run_task"
SPAN_AGENT = "agent.handle"
SPAN_AGENT_REASON = "agent.reason"
SPAN_AGENT_SYNTHESIZE = "agent.synthesize"
SPAN_TOOL = "tool.call"
SPAN_LLM = "llm.request"
SPAN_RETRIEVAL = "rag.retrieve"
SPAN_HITL_PAUSE = "hitl.pause"

# ---- Correlation ----
REQUEST_ID = "hmao.request_id"
SESSION_ID = "hmao.session_id"
AGENT_PATH = "hmao.agent_path"

# ---- Orchestration ----
ORCH_QUERY_PREVIEW = "hmao.orchestration.query_preview"
ORCH_QUERY_LEN = "hmao.orchestration.query_chars"
ORCH_STATUS = "hmao.orchestration.status"
ORCH_TURNS = "hmao.orchestration.turns"
ORCH_MANAGERS = "hmao.orchestration.managers_invoked"

# ---- Routing ----
ROUTE_NEXT_MANAGER = "hmao.route.next_manager"
ROUTE_DECISION_SOURCE = "hmao.route.source"  # llm | router_fallback
ROUTE_REASONING = "hmao.route.reasoning"
ROUTE_TURN = "hmao.route.turn"

# ---- Agent ----
AGENT_NAME = "hmao.agent.name"
AGENT_LAYER = "hmao.agent.layer"  # supervisor | manager | worker
AGENT_PARENT = "hmao.agent.parent"
AGENT_SELECTED_TOOLS = "hmao.agent.selected_tools"
AGENT_SKIPPED_TOOLS = "hmao.agent.skipped_tools"
AGENT_SUCCESS = "hmao.agent.success"

# ---- Tool ----
TOOL_NAME = "hmao.tool.name"
TOOL_SUCCESS = "hmao.tool.success"
TOOL_RATIONALE = "hmao.tool.rationale"
TOOL_RESULT_PREVIEW = "hmao.tool.result_preview"

# ---- LLM (GenAI conventions) ----
LLM_MODEL = "gen_ai.request.model"
LLM_OPERATION = "gen_ai.operation.name"  # reason | synthesize | route | aggregate
LLM_MODE = "hmao.llm.mode"  # real | mock
LLM_TOKENS_IN = "gen_ai.usage.input_tokens"
LLM_TOKENS_OUT = "gen_ai.usage.output_tokens"
LLM_COST_USD = "hmao.llm.cost_usd"

# ---- RAG ----
RAG_QUERY = "hmao.rag.query_preview"
RAG_TOP_K = "hmao.rag.top_k"
RAG_DOCS_RETURNED = "hmao.rag.docs_returned"
RAG_TOP_SCORE = "hmao.rag.top_score"
RAG_EMPTY = "hmao.rag.empty"
RAG_CONTEXT_CHARS = "hmao.rag.context_chars"

# ---- Review ----
REVIEW_FINDINGS = "hmao.review.findings"
REVIEW_BLOCKERS = "hmao.review.blockers"
REVIEW_APPROVED = "hmao.review.approved"

# ---- Layer lookup, so spans/metrics agree on what each agent is ----
AGENT_LAYERS: dict[str, str] = {
    "RootSupervisorAgent": "supervisor",
    "ResearchManagerAgent": "manager",
    "BuildManagerAgent": "manager",
    "RAGAgent": "worker",
    "SummarizerAgent": "worker",
    "CodingAgent": "worker",
    "ReviewAgent": "worker",
}


def layer_for(agent_name: str) -> str:
    return AGENT_LAYERS.get(agent_name, "worker")
