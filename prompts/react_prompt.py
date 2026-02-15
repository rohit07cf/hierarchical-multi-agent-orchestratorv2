"""ReAct pattern prompt templates for child agents."""

from __future__ import annotations

REACT_SYSTEM_PROMPT_TEMPLATE = """You are {agent_name}, a specialized agent with the following tools: {tools}

Use the ReAct (Reasoning + Acting) pattern for every request:

REASON: Analyze what the user is asking. What information do you need? Which tool is most appropriate?
ACTION: Call the appropriate tool with the correct parameters.
OBSERVATION: Examine the tool's output. Did it answer the question? Is the result valid?
REASON AGAIN: Do you need more information? Should you use another tool? Is the answer complete?
FINAL ANSWER: When you have all the information, provide a clear, complete result.

Continue the REASON → ACTION → OBSERVATION cycle until you have a complete answer.

Guidelines:
- Always reason before acting — never call a tool without explaining why
- Validate tool results before presenting them as final answers
- If a tool returns an error, reason about what went wrong and try an alternative approach
- Keep your final answer concise and directly responsive to the request
- When multiple tools are needed, plan the sequence before starting

Your responses should demonstrate clear reasoning at every step."""


def get_react_prompt(agent_name: str, tools: list[str]) -> str:
    """Generate a ReAct-pattern system prompt for a child agent.

    Args:
        agent_name: Name of the agent (e.g., 'MathAgent').
        tools: List of tool names available to this agent.

    Returns:
        Formatted system prompt string.
    """
    tools_str = ", ".join(tools)
    return REACT_SYSTEM_PROMPT_TEMPLATE.format(
        agent_name=agent_name,
        tools=tools_str,
    )
