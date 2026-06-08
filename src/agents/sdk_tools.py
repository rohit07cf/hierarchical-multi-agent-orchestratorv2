"""SDK-native tools, shared run context, and trace-capture helpers.

This module is the bridge between the deterministic pure functions in
``src/tools/*`` and the OpenAI Agents SDK. It provides:

- ``RunContext`` — the typed object threaded through every agent/sub-run
  via ``Runner.run(..., context=...)``. It replaces the old per-manager
  instance caches (``_retrieved_docs`` / ``_generated_code``) and collects
  per-agent traces so the Streamlit UI can still render them.
- ``@function_tool`` wrappers around the existing pure tool functions.
- ``traced_as_tool`` — wraps ``agent.as_tool(...)`` with a
  ``custom_output_extractor`` that captures each sub-agent's ``AgentTrace``
  into ``RunContext.traces`` (nested sub-run items do NOT surface in the
  parent ``new_items``, so this is how worker/manager traces are recovered).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agents import (
    Agent,
    ItemHelpers,
    MessageOutputItem,
    RunContextWrapper,
    RunResult,
    ToolCallItem,
    ToolCallOutputItem,
    function_tool,
)

from src.models.trace import AgentTrace, LLMMode, ToolInvocation
from src.observability.middleware import observe_retrieval
from src.tools.code_generation_tool import generate_skeleton
from src.tools.code_review_tool import review_code
from src.tools.document_loader import load_knowledge_base
from src.tools.file_context_tool import find_related_files
from src.tools.security_review_tool import scan_security
from src.tools.simple_retriever import SimpleRetriever
from src.tools.template_loader import list_templates, load_template
from src.tools.test_gap_tool import scan_test_gaps

_PREVIEW = 400


# ----------------- Shared run context -----------------


@dataclass
class RunContext:
    """State threaded through a single orchestration run.

    The SAME instance is passed to every nested ``as_tool`` sub-run (the SDK
    forwards ``context.context`` unchanged), so writes here are visible across
    the whole Supervisor → Manager → Worker tree.
    """

    documents: list[dict] = field(default_factory=list)  # RAG → Summarizer hand-off
    generated_code: str = ""  # Coding → Review hand-off
    traces: list[dict] = field(default_factory=list)  # per-agent AgentTrace dicts


# Lazily-built singleton retriever over the local knowledge base.
_KB: dict[str, str] | None = None
_RETRIEVER: SimpleRetriever | None = None


def _retriever() -> SimpleRetriever:
    global _KB, _RETRIEVER
    if _RETRIEVER is None:
        _KB = load_knowledge_base()
        _RETRIEVER = SimpleRetriever(_KB)
    return _RETRIEVER


# ----------------- Function tools (wrap src/tools/*) -----------------


@function_tool
async def simple_retriever(
    ctx: RunContextWrapper[RunContext], query: str, top_k: int = 2
) -> dict:
    """Return the top-k local knowledge-base documents whose tokens overlap the query.

    Args:
        query: The search query.
        top_k: How many documents to return.
    """
    async with observe_retrieval(query, top_k) as obs:
        docs = _retriever().retrieve(query, top_k=top_k)
        top_score = max((d.score for d in docs), default=0.0)
        obs.record(
            docs_returned=len(docs),
            top_score=top_score,
            context_chars=sum(len(d.text) for d in docs),
        )
    payload = [{"name": d.name, "score": d.score, "text": d.text} for d in docs]
    # Hand-off to the summarizer. Keep the best docs across a multi-search RAG
    # run: a later zero-overlap search must not clobber relevant docs an
    # earlier search already found.
    relevant = [d for d in payload if d["score"] > 0.0]
    existing_relevant = any(d.get("score", 0.0) > 0.0 for d in ctx.context.documents)
    if relevant or not existing_relevant:
        ctx.context.documents = relevant or payload
    return {"count": len(payload), "documents": payload}


@function_tool
def load_knowledge_base_tool() -> dict:
    """List every knowledge-base document name (useful for exploratory queries)."""
    kb = _retriever()._documents  # noqa: SLF001 — read-only name listing
    return {"count": len(kb), "names": sorted(kb.keys())}


@function_tool
def code_generation_tool(
    ctx: RunContextWrapper[RunContext], query: str, language: str = "python"
) -> dict:
    """Generate a boilerplate code skeleton for the request.

    Args:
        query: What to generate.
        language: Target language (default python).
    """
    result = generate_skeleton(query, language=language)
    ctx.context.generated_code = result.get("body", "")  # hand-off to review
    return result


@function_tool
def template_loader(ctx: RunContextWrapper[RunContext], template_name: str) -> dict:
    """Load a known named code template (e.g. fastapi_upload_endpoint, redis_memory_tool).

    Args:
        template_name: The template to load.
    """
    result = load_template(template_name)
    if result.get("found"):
        ctx.context.generated_code = result.get("body", "")
    return result


@function_tool
def list_templates_tool() -> dict:
    """List the names of all available code templates."""
    return {"available": list_templates()}


@function_tool
def file_context_tool(query: str, max_files: int = 3) -> dict:
    """Find existing project files related to the query, to ground generated code.

    Args:
        query: Topic to search project files for.
        max_files: Max files to return.
    """
    return find_related_files(query, max_files=max_files)


@function_tool
def code_review_tool(code: str) -> dict:
    """Review code for bugs, clarity, and missing error handling.

    Args:
        code: The source code to review.
    """
    return review_code(code).model_dump(mode="json")


@function_tool
def security_review_tool(code: str) -> dict:
    """Scan code for security issues (secrets, shell injection, eval).

    Args:
        code: The source code to scan.
    """
    return scan_security(code)


@function_tool
def test_gap_tool(code: str) -> dict:
    """Analyze code for untested functions/classes and test-coverage gaps.

    Args:
        code: The source code to analyze.
    """
    return scan_test_gaps(code)


# ----------------- Dynamic instructions (context injection) -----------------


def summarizer_instructions(ctx: RunContextWrapper[RunContext], agent: Agent) -> str:
    """SummarizerAgent has no tools; inject retrieved docs from the run context."""
    base = (
        "You are SummarizerAgent. Produce a concise 2-3 sentence summary that "
        "directly answers the query using only the provided text."
    )
    docs = ctx.context.documents
    if docs:
        joined = "\n\n".join(f"### {d['name']}\n{d['text']}" for d in docs)
        return f"{base}\n\nRetrieved documents:\n{joined}"
    return base


def review_instructions(ctx: RunContextWrapper[RunContext], agent: Agent) -> str:
    """ReviewAgent reviews the code generated upstream (from the run context)."""
    base = (
        "You are ReviewAgent. Review the code below by calling code_review_tool, "
        "security_review_tool, and test_gap_tool (call all that apply, then "
        "summarize the findings). Default to running all three as a quality gate."
    )
    code = ctx.context.generated_code
    if code:
        return f"{base}\n\nCode under review:\n```\n{code}\n```"
    return base


# ----------------- Trace capture across as_tool sub-runs -----------------


def _args(item: ToolCallItem) -> dict[str, Any]:
    raw = getattr(item, "raw_item", None)
    arguments = getattr(raw, "arguments", None)
    if isinstance(arguments, str):
        try:
            return json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return {}
    return arguments or {}


def trace_from_run(agent_name: str, available_tools: list[str], result: RunResult) -> dict:
    """Build an ``AgentTrace`` dict from a (sub-)run's items.

    Used by ``traced_as_tool`` to reconstruct each sub-agent's trace, and by
    the result mapper for the supervisor's own top-level run.
    """
    items = result.new_items
    outputs = {
        it.call_id: it for it in items if isinstance(it, ToolCallOutputItem)
    }
    calls = [it for it in items if isinstance(it, ToolCallItem)]
    reasoning = " ".join(
        ItemHelpers.text_message_output(it)
        for it in items
        if isinstance(it, MessageOutputItem)
    )
    selected = [c.tool_name for c in calls]
    invocations: list[ToolInvocation] = []
    for c in calls:
        out = outputs.get(c.call_id)
        preview = "" if out is None else str(out.output)[:_PREVIEW]
        invocations.append(
            ToolInvocation(
                tool_name=c.tool_name,
                arguments=_args(c),
                rationale="",
                result=getattr(out, "output", None),
                result_preview=preview,
                success=out is not None,
            )
        )
    return AgentTrace(
        agent_name=agent_name,
        llm_mode=LLMMode.REAL,
        reasoning=reasoning,
        available_tools=available_tools,
        selected_tools=selected,
        skipped_tools=[t for t in available_tools if t not in selected],
        tool_invocations=invocations,
        final_response=str(result.final_output or ""),
    ).model_dump(mode="json")


def _retrieval_line(result: RunResult) -> str:
    """Terse output for the RAG ``as_tool``, read from the documents the
    retriever actually wrote to the shared context.

    This is the source of truth — independent of whether the RAG agent stopped
    after one tool or looped and produced a (possibly "no docs") prose message.
    Reading ``final_output`` instead would mis-report retrieved docs as missing.
    """
    ctx = result.context_wrapper.context
    docs = ctx.documents if isinstance(ctx, RunContext) else []
    relevant = [d for d in docs if d.get("score", 0.0) > 0.0]
    if not relevant:
        return "No knowledge-base documents matched the query."
    names = ", ".join(d["name"] for d in relevant)
    return f"Retrieved {len(relevant)} document(s): {names}."


def _final_text(result: RunResult) -> str:
    out = result.final_output
    return out if isinstance(out, str) else str(out)


def traced_as_tool(
    agent: Agent,
    *,
    tool_name: str,
    tool_description: str,
    available_tools: list[str],
    output: str = "final",
    needs_approval: bool = False,
):
    """Expose ``agent`` as a tool that also records its trace into ``RunContext``.

    The ``custom_output_extractor`` receives the sub-run's ``RunResult`` (whose
    ``context_wrapper.context`` is the shared ``RunContext``), so it both
    captures the agent's trace and returns the parent-visible string.
    """

    async def extractor(result: RunResult) -> str:
        ctx = result.context_wrapper.context
        if isinstance(ctx, RunContext):
            ctx.traces.append(trace_from_run(agent.name, available_tools, result))
        return _retrieval_line(result) if output == "retrieval" else _final_text(result)

    return agent.as_tool(
        tool_name=tool_name,
        tool_description=tool_description,
        custom_output_extractor=extractor,
        needs_approval=needs_approval,
    )
