"""Tool selection prompt templates for guiding agents to choose the right tools."""

from __future__ import annotations

TOOL_SELECTION_PROMPT_TEMPLATE = """Given the following task, select the most appropriate tool and parameters.

Task: {task}

Available tools:
{tool_descriptions}

Respond with:
1. SELECTED TOOL: The name of the tool to use
2. REASONING: Why this tool is the best choice
3. PARAMETERS: The exact parameters to pass

Important:
- Choose the most specific tool for the job
- Ensure parameter types match the tool's requirements
- If no tool fits, explain why and suggest alternatives"""


def get_tool_selection_prompt(task: str, tool_descriptions: dict[str, str]) -> str:
    """Generate a tool selection prompt for a given task.

    Args:
        task: Description of the task requiring tool selection.
        tool_descriptions: Mapping of tool names to their descriptions.

    Returns:
        Formatted tool selection prompt string.
    """
    formatted_tools = "\n".join(
        f"  - {name}: {desc}" for name, desc in tool_descriptions.items()
    )
    return TOOL_SELECTION_PROMPT_TEMPLATE.format(
        task=task,
        tool_descriptions=formatted_tools,
    )
