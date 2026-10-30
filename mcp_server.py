"""
Nexus AI Support - MCP Tool Server
==================================
Standalone MCP server (FastMCP, stdio transport) that exposes every support tool:

    search_knowledge_base(query, department)
    create_support_ticket(department, summary, priority)
    get_ticket_status(ticket_id)

You normally do NOT run this by hand: customer_support.py launches it automatically
through AutoGen's McpWorkbench. To test it on its own:

    pip install "mcp<2"
    python mcp_server.py                      # waits for an MCP client on stdio
    npx @modelcontextprotocol/inspector python mcp_server.py   # optional visual tester

Note: stdout is reserved for the MCP protocol, so all logging goes to stderr.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
TICKETS_FILE = DATA_DIR / "tickets.json"

VALID_DEPARTMENTS = ["IT", "Admin", "HR", "Accounts", "Compliance"]

logger = logging.getLogger("mcp_server")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)  # never stdout (MCP protocol channel)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | MCP-SERVER | %(message)s", "%H:%M:%S")
    )
    logger.addHandler(_handler)


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read %s; starting fresh", path.name)
        return default


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Knowledge base + tools
# --------------------------------------------------------------------------- #
KNOWLEDGE_BASE: Dict[str, List[Dict[str, Any]]] = {
    "IT": [
        {
            "title": "VPN keeps disconnecting",
            "keywords": ["vpn", "disconnect", "disconnecting", "drops", "connection", "remote"],
            "content": "1) Switch from Wi-Fi to a wired or 5 GHz network. 2) Quit and reopen the "
            "VPN client, then choose a different gateway (e.g. 'Chennai-2'). 3) Update the "
            "client to v5.4+ from the Software Portal. 4) Disable third-party firewalls/"
            "antivirus 'web shields' that interrupt tunnels. 5) If it still drops, raise a "
            "ticket at helpdesk.nexus.example (include client logs: Help > Export Logs).",
        },
        {
            "title": "Password reset / account locked",
            "keywords": ["password", "reset", "locked", "account", "login", "mfa", "forgot"],
            "content": "Use https://pwd.nexus.example to self-reset using MFA. Accounts unlock "
            "automatically after 30 minutes; for urgent unlocks call the IT desk on ext. 4000.",
        },
        {
            "title": "New laptop / hardware request",
            "keywords": ["laptop", "hardware", "monitor", "keyboard", "mouse", "new", "request", "device"],
            "content": "Raise a 'Hardware Request' ticket with manager approval. Standard "
            "delivery is 5 working days. Faulty devices are replaced on-site within 24 hours.",
        },
        {
            "title": "Software installation",
            "keywords": ["software", "install", "installation", "application", "app", "license"],
            "content": "Approved apps are in the Software Portal (self-service). For licensed "
            "tools (Adobe, JetBrains, etc.) request a licence via the IT Catalogue.",
        },
    ],
    "Admin": [
        {
            "title": "Meeting room booking",
            "keywords": ["meeting", "room", "book", "booking", "conference", "reserve"],
            "content": "Book rooms via the Outlook 'Rooms' calendar or the Workplace app. "
            "Bookings auto-release if nobody checks in within 10 minutes. Rooms with video "
            "conferencing: Everest, Kailash, Nilgiri.",
        },
        {
            "title": "Parking and access cards",
            "keywords": ["parking", "access", "card", "badge", "id", "lost", "visitor"],
            "content": "Parking passes: apply on the Workplace app (approval in 1 day). Lost "
            "access cards: report to the front desk; a replacement costs INR 200 and is "
            "issued the same day. Visitors must be pre-registered 24h in advance.",
        },
        {
            "title": "Business travel booking",
            "keywords": ["travel", "flight", "hotel", "booking", "cab", "trip", "business"],
            "content": "Submit a travel request in the Travel Portal at least 7 days before "
            "departure with manager approval. Economy class for flights under 6 hours.",
        },
        {
            "title": "Stationery and courier",
            "keywords": ["stationery", "courier", "pen", "supplies", "pantry", "post"],
            "content": "Stationery requests go through the Admin Catalogue (delivered weekly "
            "on Wednesdays). Courier pickups are at 4 PM daily from reception.",
        },
    ],
    "HR": [
        {
            "title": "Leave policy",
            "keywords": ["leave", "vacation", "holiday", "sick", "casual", "pto", "days", "time off"],
            "content": "Annual entitlement: 18 paid leaves, 12 sick leaves, 8 casual leaves. "
            "Apply in the HR portal (Leave > Apply); manager approval is needed. Unused paid "
            "leave carries forward up to 10 days.",
        },
        {
            "title": "Payroll and payslips",
            "keywords": ["payroll", "payslip", "salary", "pay", "ctc", "tax", "form16"],
            "content": "Salary is credited on the last working day. Payslips and Form 16 are "
            "in the HR portal under 'My Pay'. For discrepancies email payroll@nexus.example "
            "before the 20th of the month.",
        },
        {
            "title": "Health insurance and benefits",
            "keywords": ["insurance", "health", "medical", "benefits", "dependent", "claim"],
            "content": "Group health cover of INR 5 lakh includes spouse, children and "
            "parents. Add dependants within 30 days of joining or a life event. Cashless "
            "claims use the e-card in the Benefits app.",
        },
        {
            "title": "Work-from-home / hybrid policy",
            "keywords": ["wfh", "work", "home", "hybrid", "remote", "office", "attendance"],
            "content": "Hybrid model: 3 days in office, 2 days remote. Extra WFH days need "
            "manager approval and must be logged in the HR portal.",
        },
    ],
    "Accounts": [
        {
            "title": "Expense reimbursement",
            "keywords": ["expense", "reimbursement", "claim", "receipt", "reimburse", "refund"],
            "content": "Submit claims within 30 days in the Finance portal with receipts. "
            "Manager approval, then payment in the next payroll cycle. Meals limit: INR 1,000 "
            "per day while travelling.",
        },
        {
            "title": "Vendor invoices and payments",
            "keywords": ["invoice", "vendor", "payment", "supplier", "payable", "po", "purchase"],
            "content": "Vendor invoices must quote a valid PO number. Standard payment terms "
            "are Net-30. Send invoices to ap@nexus.example; status can be checked in the "
            "Procurement portal.",
        },
        {
            "title": "Budget and purchase orders",
            "keywords": ["budget", "purchase", "order", "approval", "cost", "centre", "center"],
            "content": "POs above INR 50,000 need department-head approval; above INR 5 lakh "
            "also need Finance Controller approval. Budget reports are in the Finance "
            "dashboard (updated monthly).",
        },
        {
            "title": "Tax and TDS",
            "keywords": ["tax", "tds", "gst", "form", "declaration", "investment"],
            "content": "Submit investment declarations in the Finance portal by 15 January. "
            "TDS certificates are issued quarterly.",
        },
    ],
    "Compliance": [
        {
            "title": "Data privacy and protection",
            "keywords": ["data", "privacy", "gdpr", "dpdp", "personal", "protection", "breach"],
            "content": "Personal data may only be used for the purpose it was collected. "
            "Suspected data breaches must be reported to privacy@nexus.example within 24 "
            "hours. Never send customer data to personal email or unapproved tools.",
        },
        {
            "title": "Code of conduct and whistleblowing",
            "keywords": ["conduct", "ethics", "whistleblower", "harassment", "report", "misconduct", "posh"],
            "content": "Report violations confidentially via the Ethics Helpline (1800-000-"
            "ETHICS) or ethics@nexus.example. Retaliation against reporters is prohibited.",
        },
        {
            "title": "Contracts, NDAs and legal review",
            "keywords": ["contract", "nda", "legal", "agreement", "review", "signature", "sign"],
            "content": "All external contracts and NDAs need Legal review (SLA: 3 working "
            "days). Only authorised signatories may sign. Use the Contract Hub templates.",
        },
        {
            "title": "Audits and gifts policy",
            "keywords": ["audit", "gift", "gifts", "hospitality", "conflict", "interest", "regulatory"],
            "content": "Gifts above INR 2,000 must be declared. Declare any conflict of "
            "interest to Compliance. Internal audits run quarterly; respond to requests "
            "within 5 working days.",
        },
    ],
}

_STOPWORDS = {
    "the", "a", "an", "is", "are", "my", "me", "i", "to", "of", "for", "and", "or", "in",
    "on", "can", "you", "how", "do", "does", "what", "with", "it", "this", "that", "help",
    "please", "keeps", "need", "want", "about", "get",
}


def search_knowledge_base(query: str, department: str) -> str:
    """Search a department's internal knowledge base and return the most relevant articles.

    Use this tool whenever an employee asks a company-specific question (policies, how-to
    steps, contacts, SLAs, troubleshooting). Always call it before answering so that the
    reply is grounded in official company information rather than guesses.

    Args:
        query: Natural-language question or keywords describing what the employee needs,
            for example "VPN keeps disconnecting" or "how many sick leaves do I get".
        department: Which department's knowledge base to search. Must be exactly one of
            "IT", "Admin", "HR", "Accounts" or "Compliance".

    Returns:
        The text of up to two best-matching knowledge-base articles, or a message saying
        that nothing relevant was found (with the list of available topics).
    """
    logger.info("TOOL search_knowledge_base(query=%r, department=%r)", query, department)
    articles = KNOWLEDGE_BASE.get(department)
    if articles is None:
        return f"Unknown department '{department}'. Valid: {', '.join(KNOWLEDGE_BASE)}."

    tokens = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if t not in _STOPWORDS}
    scored = []
    for art in articles:
        text_words = set(re.findall(r"[a-z0-9]+", (art["title"] + " " + art["content"]).lower()))
        score = 3 * len(tokens & set(art["keywords"])) + len(tokens & text_words)
        if score > 0:
            scored.append((score, art))

    if not scored:
        topics = "; ".join(a["title"] for a in articles)
        return f"No matching article found in {department}. Available topics: {topics}."

    scored.sort(key=lambda s: s[0], reverse=True)
    return "\n\n".join(f"[{a['title']}]\n{a['content']}" for _, a in scored[:2])


def create_support_ticket(department: str, summary: str, priority: str = "Medium") -> str:
    """Create a support ticket for a department when the knowledge base cannot solve the issue.

    Use this only after searching the knowledge base, or when the employee explicitly asks
    to raise, log or escalate a ticket.

    Args:
        department: Department that owns the ticket: "IT", "Admin", "HR", "Accounts" or
            "Compliance".
        summary: One or two sentences describing the problem and what was already tried.
        priority: "Low", "Medium", "High" or "Critical". Defaults to "Medium".

    Returns:
        A confirmation containing the new ticket id (for example "IT-1A2B3C").
    """
    if department not in VALID_DEPARTMENTS:
        return f"Unknown department '{department}'. Valid: {', '.join(VALID_DEPARTMENTS)}."
    tickets = _read_json(TICKETS_FILE, {})
    ticket_id = f"{department.upper()}-{uuid.uuid4().hex[:6].upper()}"
    tickets[ticket_id] = {
        "department": department,
        "summary": summary,
        "priority": priority,
        "status": "Open",
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(TICKETS_FILE, tickets)
    logger.info("TOOL create_support_ticket -> %s", ticket_id)
    return f"Ticket {ticket_id} created with priority {priority}. Status: Open."


def get_ticket_status(ticket_id: str) -> str:
    """Look up the current status of an existing support ticket.

    Args:
        ticket_id: The ticket id returned when it was created, for example "IT-1A2B3C".

    Returns:
        The ticket's department, priority, status, creation time and summary, or a message
        saying the ticket was not found.
    """
    ticket = _read_json(TICKETS_FILE, {}).get(ticket_id.strip().upper())
    logger.info("TOOL get_ticket_status(%s) found=%s", ticket_id, bool(ticket))
    if not ticket:
        return f"No ticket found with id '{ticket_id}'."
    return (
        f"Ticket {ticket_id.upper()} | {ticket['department']} | priority {ticket['priority']} | "
        f"status {ticket['status']} | created {ticket['created']} | {ticket['summary']}"
    )


# --------------------------------------------------------------------------- #
# MCP server
# --------------------------------------------------------------------------- #
mcp = FastMCP("nexus-support-tools")
for _tool in (search_knowledge_base, create_support_ticket, get_ticket_status):
    mcp.tool()(_tool)  # type hints + docstring become the tool schema/description

if __name__ == "__main__":
    logger.info("Starting MCP server with tools: search_knowledge_base, "
                "create_support_ticket, get_ticket_status")
    mcp.run(transport="stdio")
