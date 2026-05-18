"""AgentTree and AgentNode for hierarchical agent management and visualization."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AgentNode:
    """A node in the hierarchical agent tree.

    Represents a single agent with its children, tools, and metadata.
    Supports building tree structures for visualization and traversal.
    """

    def __init__(self, name: str, agent: Any = None) -> None:
        self.name = name
        self.agent = agent
        self.children: list[AgentNode] = []
        self.tools: list[str] = []

    def add_child(self, child: AgentNode) -> AgentNode:
        """Add a child node to this agent node.

        Args:
            child: The child AgentNode to add.

        Returns:
            The added child node (for chaining).
        """
        self.children.append(child)
        return child

    def add_tool(self, tool_name: str) -> None:
        """Register a tool name with this agent node.

        Args:
            tool_name: Name of the tool to register.
        """
        if tool_name not in self.tools:
            self.tools.append(tool_name)

    def find(self, path: str) -> AgentNode | None:
        """Find a node by dot-separated path (e.g., 'Supervisor.MathAgent').

        Args:
            path: Dot-separated path from this node.

        Returns:
            The found AgentNode, or None if not found.
        """
        parts = path.split(".", 1)
        if parts[0] != self.name:
            return None
        if len(parts) == 1:
            return self

        remaining = parts[1]
        for child in self.children:
            child_parts = remaining.split(".", 1)
            if child.name == child_parts[0]:
                if len(child_parts) == 1:
                    return child
                return child.find(f"{child.name}.{child_parts[1]}")
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert node tree to a serializable dictionary."""
        return {
            "name": self.name,
            "tools": self.tools,
            "children": [c.to_dict() for c in self.children],
        }

    def __repr__(self) -> str:
        tools_str = ", ".join(self.tools) if self.tools else "none"
        children_str = ", ".join(c.name for c in self.children) if self.children else "none"
        return f"AgentNode(name={self.name!r}, tools=[{tools_str}], children=[{children_str}])"


class AgentTree:
    """Hierarchical tree structure for managing the agent supervisor-child relationships.

    Provides tree construction, traversal, search, and visualization
    capabilities for the multi-agent hierarchy.
    """

    def __init__(self, root: AgentNode) -> None:
        self.root = root

    def find_agent(self, path: str) -> AgentNode | None:
        """Find an agent node by its dot-separated path.

        Args:
            path: Dot-separated path like 'Supervisor.MathAgent'.

        Returns:
            The found AgentNode, or None.
        """
        return self.root.find(path)

    def get_all_agents(self) -> list[AgentNode]:
        """Get a flat list of all agent nodes in the tree."""
        result: list[AgentNode] = []
        self._collect_nodes(self.root, result)
        return result

    def _collect_nodes(self, node: AgentNode, result: list[AgentNode]) -> None:
        """Recursively collect all nodes in the tree."""
        result.append(node)
        for child in node.children:
            self._collect_nodes(child, result)

    def to_digraph(self) -> Any:
        """Convert the agent tree to a Graphviz Digraph for visualization.

        Returns:
            A graphviz.Digraph object representing the agent hierarchy.

        Raises:
            ImportError: If graphviz is not installed.
        """
        try:
            import graphviz
        except ImportError:
            logger.warning("graphviz not installed; cannot create Digraph")
            raise

        dot = graphviz.Digraph(
            comment="Agent Hierarchy",
            format="png",
            graph_attr={
                "rankdir": "TB",
                "splines": "ortho",
                "bgcolor": "#1e1e2e",
                "fontcolor": "#cdd6f4",
                "pad": "0.5",
            },
            node_attr={
                "shape": "box",
                "style": "rounded,filled",
                "fontname": "Helvetica",
                "fontsize": "11",
                "margin": "0.3,0.15",
            },
            edge_attr={
                "color": "#6c7086",
                "arrowsize": "0.8",
            },
        )

        self._add_nodes_to_graph(dot, self.root)
        return dot

    def _add_nodes_to_graph(self, dot: Any, node: AgentNode) -> None:
        """Recursively add nodes and edges to the Graphviz digraph."""
        # Color scheme by role: root supervisor / manager / worker.
        if node.name == "RootSupervisorAgent":
            fillcolor = "#89b4fa"
            fontcolor = "#1e1e2e"
        elif node.name.endswith("ManagerAgent"):
            fillcolor = "#f9e2af"
            fontcolor = "#1e1e2e"
        else:
            fillcolor = "#a6e3a1"
            fontcolor = "#1e1e2e"

        tools_label = "\\n".join(node.tools) if node.tools else "no tools"
        label = f"{node.name}\\n──────\\n{tools_label}"

        dot.node(
            node.name,
            label=label,
            fillcolor=fillcolor,
            fontcolor=fontcolor,
        )

        for child in node.children:
            self._add_nodes_to_graph(dot, child)
            dot.edge(node.name, child.name)

    def visualize(self) -> Any:
        """Render the agent tree visualization.

        Returns:
            The rendered Graphviz Digraph (displays in Jupyter/Streamlit).
        """
        digraph = self.to_digraph()
        return digraph

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire tree to a dictionary."""
        return self.root.to_dict()

    @classmethod
    def build_default_tree(cls) -> AgentTree:
        """Build the 3-layer hierarchical agent tree.

        Returns:
            AgentTree shaped as RootSupervisorAgent → managers → workers.
        """
        root = AgentNode(name="RootSupervisorAgent")
        root.add_tool("router")
        root.add_tool("execution_engine")

        research = AgentNode(name="ResearchManagerAgent")
        rag = AgentNode(name="RAGAgent")
        rag.add_tool("simple_retriever")
        rag.add_tool("load_knowledge_base")
        summarizer = AgentNode(name="SummarizerAgent")
        summarizer.add_tool("llm_summarize")
        research.add_child(rag)
        research.add_child(summarizer)

        build = AgentNode(name="BuildManagerAgent")
        coding = AgentNode(name="CodingAgent")
        coding.add_tool("llm_codegen")
        review = AgentNode(name="ReviewAgent")
        review.add_tool("review_code")
        build.add_child(coding)
        build.add_child(review)

        root.add_child(research)
        root.add_child(build)

        return cls(root=root)
