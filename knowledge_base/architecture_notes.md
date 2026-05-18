# Architecture Notes

## Hierarchical Multi-Agent Orchestrator

This system uses a 3-layer hierarchical orchestration pattern:

- **Layer 1 — RootSupervisorAgent:** receives the user query, generates an
  execution plan, decomposes work, routes subtasks to manager agents and
  aggregates the final response.
- **Layer 2 — Manager agents:** ResearchManagerAgent and BuildManagerAgent
  coordinate worker agents and own the contract between Layer 1 and Layer 3.
- **Layer 3 — Worker agents:** RAGAgent, SummarizerAgent, CodingAgent and
  ReviewAgent perform the actual unit of work.

## Design Goals

- Deterministic routing decisions so the orchestrator is testable without
  paid LLM access.
- Structured Pydantic models everywhere — `ExecutionPlan`, `AgentTask`,
  `OrchestratorState`, `ReviewResult` — so the orchestration trace is
  inspectable.
- Stateful execution: every subtask emits an `ExecutionStep` recorded in
  `OrchestratorState`, surfaced through the Streamlit state inspector.
- HITL friendliness: every layer pauses cleanly between subtasks so a human
  can approve, revise or cancel.

## Why three layers?

Three layers gives enough structure to demonstrate hierarchical
orchestration without producing the over-engineered "agent of agents of
agents" pattern that becomes impossible to debug. Managers exist purely to
own a single domain (research vs. build) and pick the right worker.
