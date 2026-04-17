"""
State and routing schemas for the OpenWorm assistant.

AssistantState uses TypedDict — the idiomatic LangGraph state type.
State access uses state["key"] / state.get("key", default).

QueryTypeSchema is a Pydantic BaseModel because it is only used
transiently inside the classifier node (for LLM structured output
parsing) and its .query_type string is what gets stored in state.
"""

from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field
from typing import TypedDict
from typing_extensions import Dict, List, Literal, Tuple


class QueryTypeSchema(BaseModel):
    """LLM classifier output — question or task."""

    query_type: Literal["undefined", "question", "task"] = Field(
        default="undefined",
    )


class AssistantState(TypedDict, total=False):
    query: str
    query_type: str  # "undefined", "question", or "task"
    messages: List[AnyMessage]
    context_summary: str
    message_for_user: str
    # Retrieved corpus references: {query: [(Document, score), ...]}
    reference_material: Dict[str, List[Tuple]]
    # Base64-encoded plot image from tool execution (e.g. HH voltage trace)
    plot_base64: str
    # Voltage trace data for client-side rendering (fallback when sandbox
    # matplotlib is unavailable). Dict with keys: t_ms, v_mv (lists of floats)
    plot_data: dict
    # Orchestrator: accumulated evidence from previous RAG/tool steps
    gathered_evidence: List[str]
    # Orchestrator: number of loops completed (prevents infinite cycling)
    loop_count: int
    # Orchestrator: names of MCP tools successfully called this query
    tools_used: List[str]
