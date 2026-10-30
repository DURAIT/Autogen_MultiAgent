"""
Nexus AI Support - Multi-Agent Customer Support Chatbot
=======================================================
Framework : AutoGen (autogen-agentchat >= 0.4) + SelectorGroupChat
Frontend  : Streamlit dashboard
Agents    : IT, Admin, HR, Accounts, Compliance
Tools     : served by a separate MCP server file, mcp_server.py (FastMCP over stdio).
            This app launches it automatically and the agents reach the tools through
            AutoGen's McpWorkbench. Keep both files in the same folder.

Setup
-----
    pip install -U streamlit python-dotenv "mcp<2" autogen-agentchat "autogen-ext[openai,mcp]"

    # .env (same folder as this file)
    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-4o-mini        # optional

Run
---
    streamlit run customer_support.py
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import (
    TextMessage,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
)
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
load_dotenv()  # loads OPENAI_API_KEY (and optional OPENAI_MODEL) from .env

MODEL_NAME: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
MEMORY_FILE = DATA_DIR / "long_term_memory.json"   # per-agent long-term memory
MCP_SERVER_FILE = Path(__file__).parent / "mcp_server.py"   # the MCP tool server
HISTORY_FILE = DATA_DIR / "chat_history.json"      # saved conversations
MAX_MEMORIES_PER_AGENT = 50

# --------------------------------------------------------------------------- #
# Console logging (terminal) + per-request capture (dashboard)
# --------------------------------------------------------------------------- #
logger = logging.getLogger("customer_support")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:  # Streamlit re-runs the script; avoid duplicate handlers
    _console = logging.StreamHandler(sys.stderr)
    _console.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(_console)


class ListLogHandler(logging.Handler):
    """Collects formatted log lines so they can be shown in the Streamlit UI."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: List[str] = []
        self.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


# --------------------------------------------------------------------------- #
# Department definitions
# --------------------------------------------------------------------------- #
DEPARTMENTS: Dict[str, Dict[str, str]] = {
    "IT": {
        "agent": "IT_Agent",
        "label": "IT Support",
        "icon": "🖥️",
        "description": "Handles VPN, network, laptops, software, passwords, email and any "
        "technical/IT problems.",
    },
    "Admin": {
        "agent": "Admin_Agent",
        "label": "Administration",
        "icon": "💼",
        "description": "Handles office facilities, meeting rooms, parking, travel bookings, "
        "stationery, courier and visitor management.",
    },
    "HR": {
        "agent": "HR_Agent",
        "label": "Human Resources",
        "icon": "👥",
        "description": "Handles leave, payroll queries, onboarding, benefits, insurance, "
        "recruitment, appraisals and workplace policies.",
    },
    "Accounts": {
        "agent": "Accounts_Agent",
        "label": "Accounts",
        "icon": "💲",
        "description": "Handles invoices, expense claims, reimbursements, budgets, purchase "
        "orders, vendor payments and tax questions.",
    },
    "Compliance": {
        "agent": "Compliance_Agent",
        "label": "Compliance",
        "icon": "⚖️",
        "description": "Handles legal, data privacy, code of conduct, audits, contracts, "
        "NDAs, regulatory and risk questions.",
    },
}
AGENT_TO_DEPT: Dict[str, str] = {v["agent"]: k for k, v in DEPARTMENTS.items()}

# --------------------------------------------------------------------------- #
# Long-term memory (persisted JSON -> AutoGen ListMemory)
# --------------------------------------------------------------------------- #
def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read %s; starting fresh", path.name)
        return default


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_memory_store() -> Dict[str, List[str]]:
    return _read_json(MEMORY_FILE, {})


def remember(dept: str, user_question: str, answer: str) -> None:
    """Persist a short note in the responding agent's long-term memory."""
    store = load_memory_store()
    note = (
        f"[{datetime.now():%Y-%m-%d %H:%M}] Employee asked: {user_question.strip()[:200]} "
        f"| {dept} agent answered: {answer.strip().replace(chr(10), ' ')[:300]}"
    )
    store.setdefault(dept, []).append(note)
    store[dept] = store[dept][-MAX_MEMORIES_PER_AGENT:]
    _write_json(MEMORY_FILE, store)
    logger.info("MEMORY saved for %s agent (%d notes)", dept, len(store[dept]))


async def build_memory(dept: str) -> ListMemory:
    """Create a ListMemory for an agent and pre-load its persisted notes."""
    memory = ListMemory(name=f"{dept}_long_term_memory")
    for note in load_memory_store().get(dept, []):
        await memory.add(MemoryContent(content=note, mime_type=MemoryMimeType.TEXT))
    return memory


# --------------------------------------------------------------------------- #
# Multi-agent team
# --------------------------------------------------------------------------- #
SELECTOR_PROMPT = """You are the routing coordinator of the Nexus AI Support desk.
Read the conversation and choose the ONE specialist best suited to answer the
employee's latest question.

Specialists:
{roles}

Conversation:
{history}

Select exactly one agent from {participants}. Return only the agent name."""


async def build_team(
    model_client: OpenAIChatCompletionClient, workbench: McpWorkbench
) -> SelectorGroupChat:
    """Create the 5 department agents (memory + MCP tools) and the selector team."""
    agents: List[AssistantAgent] = []
    for key, info in DEPARTMENTS.items():
        system_message = (
            f"You are the {info['label']} specialist at Nexus, a friendly workplace support "
            f"desk. Your scope: {info['description']}\n"
            f"Rules:\n"
            f"- ALWAYS call search_knowledge_base with department='{key}' before answering.\n"
            f"- If the issue is unresolved or the employee asks to raise/escalate a ticket, call "
            f"create_support_ticket with department='{key}'; use get_ticket_status to check a "
            f"ticket id.\n"
            f"- Write a clear, concise, friendly answer in your own words from the search "
            f"results (use short numbered steps when helpful). Never paste raw search output.\n"
            f"- Use your memory notes of earlier conversations to personalise the reply.\n"
            f"- If the knowledge base has no answer, say so and suggest how to escalate."
        )
        agents.append(
            AssistantAgent(
                name=info["agent"],
                description=info["description"],
                model_client=model_client,
                workbench=workbench,       # all tools come from the MCP server
                memory=[await build_memory(key)],
                reflect_on_tool_use=True,  # write an answer from the tool results
                system_message=system_message,
            )
        )

    return SelectorGroupChat(
        participants=agents,
        model_client=model_client,
        selector_prompt=SELECTOR_PROMPT,
        termination_condition=MaxMessageTermination(max_messages=2),  # task + 1 answer
        max_turns=1,  # exactly one specialist answers each question
        allow_repeated_speaker=True,
    )


def _short(text: Any, limit: int = 160) -> str:
    text = str(text).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."


async def handle_query(
    user_text: str,
    recent_history: List[Dict[str, Any]],
    on_route: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Route one user question through the SelectorGroupChat and return the result."""
    log_handler = ListLogHandler()
    logger.addHandler(log_handler)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.removeHandler(log_handler)
        raise RuntimeError("OPENAI_API_KEY not found. Add it to your .env file.")

    model_client = OpenAIChatCompletionClient(model=MODEL_NAME, api_key=api_key)
    routed_dept: Optional[str] = None
    answer = ""
    try:
        logger.info("USER  %s", _short(user_text, 300))
        context = "\n".join(
            f"{'Employee' if m['role'] == 'user' else 'Assistant'}: {_short(m['content'], 300)}"
            for m in recent_history[-6:]
        )
        task = (
            (f"Earlier conversation:\n{context}\n\n" if context else "")
            + f"Employee's current question: {user_text}"
        )

        # Launch mcp_server.py (stdio) and connect the agents to it.
        if not MCP_SERVER_FILE.exists():
            raise RuntimeError(f"MCP server file not found: {MCP_SERVER_FILE}")
        mcp_params = StdioServerParams(
            command=sys.executable,
            args=[str(MCP_SERVER_FILE)],
            read_timeout_seconds=60,
        )
        async with McpWorkbench(mcp_params) as workbench:
            tool_names = [t["name"] for t in await workbench.list_tools()]
            logger.info("MCP   connected, tools: %s", ", ".join(tool_names))

            team = await build_team(model_client, workbench)
            async for event in team.run_stream(task=task):
                if isinstance(event, TaskResult):
                    logger.info("TASK  finished (%s)", event.stop_reason)
                    continue

                source = getattr(event, "source", "")
                kind = type(event).__name__

                # First time we see a department agent => the selector has routed the task.
                if source in AGENT_TO_DEPT and routed_dept is None:
                    routed_dept = AGENT_TO_DEPT[source]
                    logger.info("ROUTE selector -> %s (%s)", source, DEPARTMENTS[routed_dept]["label"])
                    if on_route:
                        on_route(routed_dept)

                if isinstance(event, ToolCallRequestEvent):
                    for call in event.content:
                        logger.info("AGENT %s calls MCP tool %s(%s)", source, call.name, _short(call.arguments))
                elif isinstance(event, ToolCallExecutionEvent):
                    for res in event.content:
                        logger.info("MCP   result: %s", _short(res.content))
                elif isinstance(event, TextMessage):
                    if source in AGENT_TO_DEPT:
                        answer = event.content
                        logger.info("AGENT %s answered: %s", source, _short(answer, 200))
                elif kind == "SelectorEvent":
                    logger.info("SELECT speaker chosen: %s", _short(getattr(event, "content", "")))
        logger.info("MCP   server session closed")

        if routed_dept and answer:
            remember(routed_dept, user_text, answer)

        return {
            "dept": routed_dept,
            "answer": answer or "Sorry, I couldn't produce an answer. Please try again.",
            "logs": log_handler.lines,
        }
    finally:
        await model_client.close()  # always close the model client
        logger.info("CLIENT model client closed")
        logger.removeHandler(log_handler)


# --------------------------------------------------------------------------- #
# Chat-history persistence (sessions shown under "Recent Conversations")
# --------------------------------------------------------------------------- #
def save_session() -> None:
    sessions = _read_json(HISTORY_FILE, {})
    msgs = st.session_state.messages
    if not msgs:
        sessions.pop(st.session_state.session_id, None)
    else:
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), "New chat")
        sessions[st.session_state.session_id] = {
            "title": _short(first_user, 38),
            "updated": datetime.now().isoformat(timespec="seconds"),
            "messages": msgs,
        }
    _write_json(HISTORY_FILE, sessions)


# --------------------------------------------------------------------------- #
# Streamlit dashboard
# --------------------------------------------------------------------------- #
CSS = """
<style>
.stApp { background: linear-gradient(135deg,#eef2fb 0%,#f7f9fe 100%); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }
.block-container { padding-top: 1.2rem; max-width: 1100px; }

/* Sidebar */
section[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid #e6eaf3; }
.brand { display:flex; align-items:center; gap:12px; margin-bottom:18px; }
.brand .logo { width:42px;height:42px;border-radius:12px;background:#e8efff;color:#1f3fae;
  display:flex;align-items:center;justify-content:center;font-weight:800;font-size:22px; }
.brand .t1 { font-weight:700; font-size:16px; line-height:1.1; color:#1b2440; }
.brand .t2 { font-size:15px; color:#5b6784; }
.sec { font-size:11.5px; letter-spacing:.06em; color:#5b6784; margin:6px 0 8px; }
.dept { display:flex; align-items:center; gap:12px; padding:9px 10px; border-radius:10px;
  font-size:14.5px; color:#1b2440; margin-bottom:4px; }
.dept .ico { width:32px;height:32px;border-radius:9px;background:#eef3ff;
  display:flex;align-items:center;justify-content:center;font-size:16px; }
.dept.active { background:#dfe9fb; font-weight:600; }

/* Header */
.topbar { background:#fff; border:1px solid #e6eaf3; border-radius:14px 14px 0 0;
  padding:14px 22px; display:flex; justify-content:space-between; align-items:center; }
.topbar h1 { font-size:21px; margin:0; color:#1b2440; font-weight:700; }
.welcome { background:#fff; border:1px solid #e6eaf3; border-top:none;
  padding:12px 22px; display:flex; justify-content:space-between; align-items:center;
  border-radius:0 0 14px 14px; margin-bottom:14px; }
.welcome .a { font-size:15px; color:#1b2440; } .welcome .b { font-size:12.5px; color:#6a7591; }
.pill { background:#e6f7ec; color:#1a7f45; border:1px solid #bfe8cf; padding:6px 14px;
  border-radius:999px; font-size:13.5px; font-weight:500; }
.pill:before { content:"●"; margin-right:7px; color:#22a35a; }

/* Chat */
div[data-testid="stChatMessage"] { background:#f1f3f8; border-radius:16px; padding:12px 16px;
  border:1px solid #e6eaf3; }
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
  background:#2f6fe4; color:#fff; border:none; }
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) * { color:#fff !important; }
.route { display:inline-block; background:#e6eefc; color:#1f3fae; border-radius:10px;
  padding:5px 12px; font-size:13px; margin-bottom:8px; }
.route b { background:#2f6fe4; color:#fff; border-radius:6px; padding:1px 6px; margin-right:8px; font-size:11px; }
div[data-testid="stChatInput"] { border-radius:16px; }
</style>
"""


def route_chip(dept: Optional[str]) -> str:
    if not dept:
        return ""
    return (
        f'<div class="route"><b>AI</b>Analyzing your request and connecting you to '
        f'{html.escape(DEPARTMENTS[dept]["label"])}...</div>'
    )


def render_sidebar() -> None:
    sb = st.sidebar
    sb.markdown(
        '<div class="brand"><div class="logo">N</div><div><div class="t1">Nexus</div>'
        '<div class="t2">AI Support</div></div></div><div class="sec">DEPARTMENTS</div>',
        unsafe_allow_html=True,
    )
    active = st.session_state.active_dept
    items = "".join(
        f'<div class="dept {"active" if key == active else ""}">'
        f'<span class="ico">{d["icon"]}</span>{d["label"]}</div>'
        for key, d in DEPARTMENTS.items()
    )
    sb.markdown(items, unsafe_allow_html=True)

    sb.markdown("---")
    sb.markdown('<div class="sec">RECENT CONVERSATIONS</div>', unsafe_allow_html=True)
    sessions = _read_json(HISTORY_FILE, {})
    recent = sorted(sessions.items(), key=lambda kv: kv[1]["updated"], reverse=True)[:6]
    if not recent:
        sb.caption("No conversations yet.")
    for sid, s in recent:
        if sb.button(f"💬 {s['title']}", key=f"sess_{sid}", use_container_width=True):
            st.session_state.session_id = sid
            st.session_state.messages = s["messages"]
            st.session_state.active_dept = next(
                (m.get("dept") for m in reversed(s["messages"]) if m.get("dept")), None
            )
            st.rerun()

    sb.markdown("---")
    with sb.expander("⚙️ Settings & Clear Chat"):
        st.caption(f"Model: `{MODEL_NAME}`")
        st.session_state.show_logs = st.checkbox("Show console log", st.session_state.show_logs)
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.console_logs = []
            st.session_state.active_dept = None
            st.session_state.session_id = uuid.uuid4().hex
            st.rerun()
        if st.button("🧠 Clear long-term memory", use_container_width=True):
            _write_json(MEMORY_FILE, {})
            logger.info("MEMORY cleared for all agents")
            st.success("Agent memory cleared.")


def render_header() -> None:
    st.markdown(
        """
        <div class="topbar"><h1>AI Customer Support Assistant</h1></div>
        <div class="welcome">
          <div><div class="a">Hello! <b>Welcome to Nexus AI Support.</b></div>
          <div class="b">Your intelligent workplace support, powered by multi-agent AI</div></div>
          <div class="pill">AI System Online</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Nexus AI Support", page_icon="🤖", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    ss = st.session_state
    ss.setdefault("messages", [])
    ss.setdefault("session_id", uuid.uuid4().hex)
    ss.setdefault("active_dept", None)
    ss.setdefault("console_logs", [])
    ss.setdefault("show_logs", True)

    render_sidebar()
    render_header()

    with st.chat_message("assistant"):
        st.write("Hello! Welcome to Nexus AI Support. How can I assist you today?")

    for m in ss.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "assistant":
                st.markdown(route_chip(m.get("dept")), unsafe_allow_html=True)
            st.markdown(m["content"])

    if ss.show_logs and ss.console_logs:
        with st.expander("🖥️ Console Log", expanded=False):
            st.code("\n".join(ss.console_logs[-80:]), language="text")

    prompt = st.chat_input("Ask anything about IT, Admin, HR, Accounts or Compliance...")
    if not prompt:
        return

    history_before = list(ss.messages)
    ss.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.empty()
        status.markdown(
            '<div class="route"><b>AI</b>Analyzing your request and routing it...</div>',
            unsafe_allow_html=True,
        )

        def on_route(dept: str) -> None:
            status.markdown(route_chip(dept), unsafe_allow_html=True)

        try:
            result = asyncio.run(handle_query(prompt, history_before, on_route))
        except Exception as exc:  # surfaces config/API problems in the UI
            logger.exception("Request failed")
            ss.messages.pop()  # remove the unanswered user message
            status.empty()
            st.error(f"Something went wrong: {exc}")
            return

    ss.active_dept = result["dept"]
    ss.console_logs.extend(result["logs"])
    ss.messages.append(
        {"role": "assistant", "content": result["answer"], "dept": result["dept"]}
    )
    save_session()
    st.rerun()


main()
