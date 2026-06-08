"""Curated example queries shown in the UI to onboard new users.

Each example is mapped to the agent path it exercises so a first-time user
can see, at a glance, what the orchestrator can do without having to read the
code or guess. Grouped by the manager/worker path they trigger:

- Research & Summarize → ResearchManagerAgent (RAGAgent + SummarizerAgent)
- Build & Review       → BuildManagerAgent (CodingAgent + ReviewAgent)
- Combined             → both managers in one plan
- Reflective / Direct  → answered without retrieval or codegen
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExampleQuery:
    """A single suggested query.

    Attributes:
        label: Short button label.
        query: The full text submitted to the orchestrator.
        path: Human-readable routing path, shown as a button tooltip.
    """

    label: str
    query: str
    path: str


@dataclass(frozen=True)
class ExampleCategory:
    """A named group of example queries."""

    icon: str
    title: str
    description: str
    examples: list[ExampleQuery]


EXAMPLE_CATEGORIES: list[ExampleCategory] = [
    ExampleCategory(
        icon="🔍",
        title="Research & Summarize",
        description="Retrieve from the knowledge base, then summarize.",
        examples=[
            ExampleQuery(
                label="Summarize this project's architecture",
                query="Summarize the architecture of this project.",
                path="ResearchManager → Summarizer",
            ),
            ExampleQuery(
                label="Find orchestration patterns in the KB",
                query="Search the knowledge base for agent orchestration "
                "patterns and summarize the key ideas.",
                path="ResearchManager → RAG → Summarizer",
            ),
            ExampleQuery(
                label="Summarize RAG retrieval docs",
                query="Find documents about RAG retrieval and give me a "
                "short summary of how it works.",
                path="ResearchManager → RAG → Summarizer",
            ),
        ],
    ),
    ExampleCategory(
        icon="🛠️",
        title="Build & Review",
        description="Generate code, then review it for quality and security.",
        examples=[
            ExampleQuery(
                label="FastAPI upload endpoint + review",
                query="Build a FastAPI endpoint for uploading documents and "
                "review the solution.",
                path="BuildManager → Coding → Review",
            ),
            ExampleQuery(
                label="Redis memory tool + review",
                query="Generate a simple Redis memory tool and review it for "
                "production concerns.",
                path="BuildManager → Coding → Review",
            ),
            ExampleQuery(
                label="Rate limiter + security review",
                query="Write a Python rate limiter decorator and review it "
                "for security and test gaps.",
                path="BuildManager → Coding → Review",
            ),
        ],
    ),
    ExampleCategory(
        icon="🔀",
        title="Combined (Research + Build)",
        description="Both managers in one plan — research informs the build.",
        examples=[
            ExampleQuery(
                label="Research patterns → implement guidance",
                query="Search the knowledge base for agent orchestration "
                "patterns and generate implementation guidance.",
                path="ResearchManager + BuildManager",
            ),
            ExampleQuery(
                label="Research retries → build HTTP client",
                query="Research best practices for retry logic, then "
                "implement a resilient HTTP client and review it.",
                path="ResearchManager + BuildManager",
            ),
        ],
    ),
    ExampleCategory(
        icon="💬",
        title="Reflective / Direct",
        description="Answered directly — no retrieval or code generation.",
        examples=[
            ExampleQuery(
                label="How does the router decide?",
                query="Explain how a hierarchical multi-agent orchestrator "
                "routes a request to the right manager.",
                path="RootSupervisor (direct)",
            ),
            ExampleQuery(
                label="A reflective prompt",
                query="The meaning of life is to live, to understand, and to "
                "create something meaningful to share with others.",
                path="RootSupervisor (direct)",
            ),
        ],
    ),
]
