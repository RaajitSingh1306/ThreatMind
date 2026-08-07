"""
LangGraph ReAct agent for ThreatMind threat intelligence.

Graph:
    START → agent_node → (tool_node | END)
              ↑               │
              └───────────────┘

The agent uses a ReAct loop: reason → act (tool call) → observe → repeat.
Terminates when the LLM issues a final answer without tool calls.
"""

from __future__ import annotations

import os
from typing import Annotated, Any

import structlog
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from src.agent.prompts import REACT_SYSTEM_PROMPT
from src.agent.tools import ALL_TOOLS

load_dotenv()
log = structlog.get_logger()


# ── State ──────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls_made: list[str]
    reasoning_trace: list[str]


# ── LLM client (free providers) ────────────────────────────────────────────────
def _build_llm_with_tools():
    """
    Build an LLM callable that supports tool calling.
    Tries LangChain bindings for Groq / Together / Ollama.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        try:
            from langchain_groq import ChatGroq  # noqa: PLC0415

            llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            )
            return llm.bind_tools(ALL_TOOLS)
        except ImportError:
            log.warning("langchain-groq not installed, falling back to direct HTTP")

    if provider == "ollama":
        try:
            from langchain_community.chat_models import ChatOllama  # noqa: PLC0415

            llm = ChatOllama(
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            )
            return llm.bind_tools(ALL_TOOLS)
        except ImportError:
            log.warning("langchain-community not installed")

    # Fallback: Together via OpenAI-compat endpoint
    try:
        from langchain_openai import ChatOpenAI  # noqa: PLC0415

        llm = ChatOpenAI(
            api_key=os.getenv("TOGETHER_API_KEY", ""),
            base_url="https://api.together.xyz/v1",
            model=os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3-70b-chat-hf"),
        )
        return llm.bind_tools(ALL_TOOLS)
    except ImportError:
        pass

    raise RuntimeError("No LLM backend available. Install langchain-groq, langchain-community, or langchain-openai.")


# ── Nodes ──────────────────────────────────────────────────────────────────────
def agent_node(state: AgentState) -> AgentState:
    """LLM reasoning step — may produce tool calls or a final answer."""
    llm = _build_llm_with_tools()
    from langchain_core.messages import SystemMessage  # noqa: PLC0415

    messages = [SystemMessage(content=REACT_SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)

    tool_names = [tc.get("name", "") for tc in (response.tool_calls or [])]
    reasoning = state.get("reasoning_trace", [])
    if tool_names:
        reasoning.append(f"Calling tools: {tool_names}")

    return {
        "messages": [response],
        "tool_calls_made": state.get("tool_calls_made", []) + tool_names,
        "reasoning_trace": reasoning,
    }


def should_continue(state: AgentState) -> str:
    """Route: if last message has tool calls → tools, else → END."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


# ── Graph ──────────────────────────────────────────────────────────────────────
def build_graph() -> Any:
    tool_node = ToolNode(ALL_TOOLS)

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    return builder.compile()


def run_agent(query: str) -> dict[str, Any]:
    """
    Run the ThreatMind ReAct agent for a natural language query.
    Returns structured response with answer, sources, and reasoning trace.
    """
    from langchain_core.messages import HumanMessage  # noqa: PLC0415

    graph = build_graph()
    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "tool_calls_made": [],
        "reasoning_trace": [],
    }

    log.info("Agent invoked", query=query[:100])
    final_state = graph.invoke(initial_state)

    last_message = final_state["messages"][-1]
    answer = last_message.content if hasattr(last_message, "content") else str(last_message)

    # Extract CVE sources from tool outputs
    sources = []
    for msg in final_state["messages"]:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            import re  # noqa: PLC0415

            sources.extend(re.findall(r"CVE-\d{4}-\d+", msg.content))

    return {
        "answer": answer,
        "sources": list(dict.fromkeys(sources)),  # dedup, preserve order
        "tool_calls": final_state.get("tool_calls_made", []),
        "reasoning_trace": " → ".join(final_state.get("reasoning_trace", [])),
    }
