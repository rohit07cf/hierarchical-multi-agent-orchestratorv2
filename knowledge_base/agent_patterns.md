# Agent Orchestration Patterns

## Supervisor / Manager / Worker

The orchestrator follows the classic supervisor-worker pattern with an
intermediate manager layer:

- The supervisor focuses on *planning and aggregation*.
- Managers focus on *coordination* inside a single capability domain.
- Workers focus on a single tool-grade primitive.

## Routing

Routing is deterministic and rule-based:

| Signal in query                             | Manager invoked           |
|---------------------------------------------|---------------------------|
| "summarize", "explain", "search", "context" | ResearchManagerAgent      |
| "build", "implement", "endpoint", "code"    | BuildManagerAgent         |
| "review", "audit", "security"               | BuildManagerAgent         |

Queries that mention both research and build keywords trigger both
managers in sequence (Research → Build) so that retrieved context can feed
the coding workflow.

## Execution plan

The plan is a list of `AgentTask` objects. Each task carries:

- `agent_name`
- `description`
- `tools_needed`
- `depends_on`

## Observability

Every step emits an `ExecutionStep` with a `kind` of
`task_decomposition`, `subtask_started`, `subtask_complete` or
`orchestration_complete`. These map directly onto the Streamlit
"Execution Timeline" panel.

## Mock-friendly design

If `OPENAI_API_KEY` is not set, all worker agents fall back to
deterministic logic so the demo runs offline. This makes the project
interview-friendly and testable in CI.
