# hierarchical-multi-agent-orchestrator

A production-grade hierarchical multi-agent orchestration system demonstrating 
advanced agent architecture patterns with OpenAI Agents SDK. Features supervisor 
task decomposition, HITL state restoration, Temporal workflow durability, and 
real-time streaming callbacks. Implements ReAct pattern for multi-step reasoning 
with type-safe Pydantic models and structured LLM outputs.

Key Features:
- Hierarchical agent coordination (Supervisor + 4 specialized agents)
- HITL state management & restoration with AgentState persistence
- Temporal workflow orchestration for durable execution
- Real-time streaming with StreamingCallbackHandler
- Custom ReAct & task decomposition prompts
- Type-safe tool calling with Pydantic & OpenAI Structured Outputs
- Beautiful Streamlit UI with agent tree visualization

Technologies: Python, OpenAI Agents SDK, Temporal, Pydantic, Streamlit
