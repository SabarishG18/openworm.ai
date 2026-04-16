"""
OpenWorm.ai FastAPI backend
Wraps the RAG pipeline and MCP tool calls behind HTTP endpoints.
"""

import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# Load .env from repo root
load_dotenv(Path(__file__).resolve().parent / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
VS_CONFIG = os.getenv("GEN_RAG_VS_CONFIG", str(REPO_ROOT / "vector-stores.json"))
MCP_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8543/mcp")

app = FastAPI(title="OpenWorm.ai Backend")

# Simple in-memory assistant cache by model name
_ASSISTANTS: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_async(coro):
    """Run an async coroutine from sync code."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def get_assistant(chat_model: str):
    """Create or reuse an OpenWorm assistant for the given model."""
    from openworm_ai.assistant import OpenWormAssistant

    if chat_model not in _ASSISTANTS:
        assistant = OpenWormAssistant(
            vs_config_file=VS_CONFIG,
            chat_model=chat_model,
            mcp_url=MCP_URL,
        )
        _run_async(assistant.setup())
        _ASSISTANTS[chat_model] = assistant
    return _ASSISTANTS[chat_model]


def call_mcp_tool(tool_name: str, params: dict):
    """Call an MCP tool and return the parsed result."""
    from fastmcp import Client

    async def _call():
        async with Client(MCP_URL) as client:
            result = await client.call_tool(tool_name, params)
            texts = []
            for block in result.content:
                texts.append(block.text if hasattr(block, "text") else str(block))
            return "\n".join(texts)

    return _run_async(_call())


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str
    chat_model: str
    messages: list[ChatMessage] = []


class MCPToolRequest(BaseModel):
    tool_name: str
    params: dict


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "vector_store_config": VS_CONFIG,
        "mcp_url": MCP_URL,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    """Run the OpenWorm assistant graph and return final state."""
    from langchain_core.messages import AIMessage, HumanMessage

    assistant = get_assistant(req.chat_model)

    history = []
    for msg in req.messages:
        if msg.role == "user":
            history.append(HumanMessage(content=msg.content))
        else:
            history.append(AIMessage(content=msg.content))

    final_state = _run_async(
        assistant.run_graph_invoke_state(
            {
                "query": req.query,
                "messages": history,
            }
        )
    )

    return final_state


@app.post("/tool/run")
def run_tool(req: MCPToolRequest):
    """Run an MCP tool and return its raw text output."""
    output = call_mcp_tool(req.tool_name, req.params)
    return {"output": output}