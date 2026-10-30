# Nexus AI Support – Multi-Agent Customer Support Chatbot

A multi-agent workplace support assistant built with **Microsoft AutoGen**, **MCP (Model Context Protocol)** and **Streamlit**.

An employee asks a question in plain language. A `SelectorGroupChat` coordinator reads it, picks the department specialist best suited to answer (IT, Admin, HR, Accounts or Compliance), and that agent uses tools served by an MCP server to produce a grounded answer.

---

## Table of Contents

1. [Features](#features)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Running the App](#running-the-app)
8. [Using the Dashboard](#using-the-dashboard)
9. [Example Questions](#example-questions)
10. [How It Works](#how-it-works)
11. [The MCP Server and Its Tools](#the-mcp-server-and-its-tools)
12. [Memory, Chat History and Logging](#memory-chat-history-and-logging)
13. [Customization](#customization)
14. [Tech Stack](#tech-stack)

---

## Features

- **Five specialist agents**: IT, Admin, HR, Accounts and Compliance, each with its own skill set and scope.
- **Automatic routing**: `SelectorGroupChat` assigns each question to the right agent.
- **MCP tool server**: all tools live in a separate `mcp_server.py` and are consumed through AutoGen's `McpWorkbench`.
- **Reflection on tool use**: agents are created with `reflect_on_tool_use=True`, so they write a natural answer from the tool results instead of dumping raw search output.
- **Long-term memory**: every agent keeps its own persistent memory across restarts.
- **Chat history**: conversations are saved and can be reopened from the sidebar.
- **Console log**: routing decisions, tool calls and results are printed to the terminal and shown in the dashboard.
- **Streamlit dashboard**: department sidebar, chat bubbles, routing indicator and "AI System Online" status.
- **Safe shutdown**: the model client and the MCP session are closed after every request.

---

## Architecture

```
                    ┌────────────────────────────────────────┐
                    │          Streamlit Dashboard           │
                    │         (customer_support.py)          │
                    └───────────────────┬────────────────────┘
                                        │ user question
                                        ▼
                    ┌────────────────────────────────────────┐
                    │          SelectorGroupChat             │
                    │   (LLM picks the best-fit specialist)  │
                    └───┬─────────┬─────────┬─────────┬──────┘
                        ▼         ▼         ▼         ▼         ▼
                     IT_Agent  Admin_Agent HR_Agent Accounts_  Compliance_
                                                     Agent      Agent
                        │  (each agent: system prompt + long-term memory)
                        └─────────────────────┬─────────────────────┘
                                              │ McpWorkbench (stdio)
                                              ▼
                    ┌────────────────────────────────────────┐
                    │        MCP Server (mcp_server.py)      │
                    │  • search_knowledge_base               │
                    │  • create_support_ticket               │
                    │  • get_ticket_status                   │
                    └────────────────────────────────────────┘
```

---

## Project Structure

```
your-project/
├── customer_support.py        # Streamlit app, agents, team, memory, chat history
├── mcp_server.py              # MCP tool server (knowledge base + ticket tools)
├── README.md                  # This file
├── requirements.txt           # Python dependencies (you create this, see below)
├── .env                       # Your API key (you create this, see below)
└── data/                      # Created automatically on first run
    ├── long_term_memory.json  # Per-agent persistent memory
    ├── chat_history.json      # Saved conversations
    └── tickets.json           # Tickets created through the MCP tool
```

> `customer_support.py` and `mcp_server.py` must stay in the **same folder**, because the app launches the server from that location.

---

## Prerequisites

- **Python 3.10 or newer**
- **An OpenAI API key** from <https://platform.openai.com/api-keys>
- **pip** (comes with Python)
- Internet access for package installation and OpenAI API calls

---

## Installation

### Step 1 – Get the project files

Place `customer_support.py` and `mcp_server.py` together in a new folder, for example:

```bash
mkdir nexus-support
cd nexus-support
```

### Step 2 – Create a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3 – Create `requirements.txt`

Create a file named `requirements.txt` with the following content:

```text
streamlit
python-dotenv
mcp<2
autogen-agentchat
autogen-ext[openai,mcp]
```

> The `mcp<2` pin is required. The newest major release of `mcp` is not compatible with `autogen-ext`'s MCP support yet.

### Step 4 – Install the dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

---

## Configuration

### Step 5 – Create the `.env` file

In the project folder, create a file named `.env`:

```env
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o-mini
```

| Variable         | Required | Description                                              |
| ---------------- | -------- | -------------------------------------------------------- |
| `OPENAI_API_KEY` | Yes      | Your OpenAI API key, loaded with `load_dotenv()`.        |
| `OPENAI_MODEL`   | No       | Model used by the selector and all agents. Defaults to `gpt-4o-mini`. |

> Never commit `.env` to version control. Add it to your `.gitignore`.

---

## Running the App

### Step 6 – Start the dashboard

```bash
streamlit run customer_support.py
```

Streamlit prints a local URL (normally <http://localhost:8501>) and opens it in your browser.

You do **not** need to start the MCP server yourself. The app launches `mcp_server.py` automatically for every request and shuts it down afterwards.

### Step 7 – (Optional) Test the MCP server on its own

To inspect the tools without the chatbot, use the MCP Inspector (requires Node.js):

```bash
npx @modelcontextprotocol/inspector python mcp_server.py
```

---

## Using the Dashboard

| Area                        | What it does                                                                                              |
| --------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Departments (sidebar)**   | Lists the five departments. The department that answered the latest question is highlighted.              |
| **Recent Conversations**    | Shows your latest saved chats. Click one to reload it.                                                    |
| **Settings & Clear Chat**   | Expands to show the model name, a *Show console log* toggle, *Clear chat* and *Clear long-term memory*.   |
| **Header**                  | Title bar with the "AI System Online" status badge.                                                       |
| **Chat area**               | Your messages (blue bubbles) and the assistant's replies, each tagged with the routing message.           |
| **Console Log (expander)**  | Live trace of routing, MCP tool calls, tool results, memory writes and shutdown events.                   |
| **Message box**             | Type a question and press Enter.                                                                          |

### Typical flow

1. Type a question in the message box.
2. The chat shows *"Analyzing your request and connecting you to IT Support..."* while the selector decides.
3. The chosen agent searches the knowledge base through MCP and writes the answer.
4. The sidebar highlights the department, and the exchange is saved to chat history and to that agent's memory.

---

## Example Questions

| Department     | Try asking                                                       |
| -------------- | ---------------------------------------------------------------- |
| **IT**         | "My VPN keeps disconnecting. Can you help me resolve this?"      |
| **IT**         | "I forgot my password and my account is locked."                 |
| **Admin**      | "How do I book a meeting room with video conferencing?"          |
| **Admin**      | "I lost my access card. What should I do?"                       |
| **HR**         | "How many sick leaves do I get in a year?"                       |
| **HR**         | "What is the hybrid work-from-home policy?"                      |
| **Accounts**   | "How do I claim reimbursement for my travel expenses?"           |
| **Accounts**   | "What are the payment terms for vendor invoices?"                |
| **Compliance** | "How do I report a suspected data breach?"                       |
| **Compliance** | "Does an NDA need legal review before signing?"                  |

Ticket examples (handled by any agent through MCP):

- "Please raise a high priority ticket, my VPN still drops after trying those steps."
- "What is the status of ticket IT-1A2B3C?"

---

## How It Works

1. **Request received.** `handle_query()` in `customer_support.py` builds a task from the latest question plus the last few chat messages for context.
2. **MCP connection.** The app starts `mcp_server.py` over stdio using `StdioServerParams` and opens an `McpWorkbench`. The available tool names are logged.
3. **Team creation.** `build_team()` creates five `AssistantAgent` instances. Each one gets:
   - a department-specific system prompt,
   - its own `ListMemory` pre-loaded from `data/long_term_memory.json`,
   - the shared MCP workbench,
   - `reflect_on_tool_use=True`.
4. **Routing.** `SelectorGroupChat` uses a selector prompt and the LLM to choose exactly one agent (`max_turns=1`).
5. **Tool use.** The chosen agent calls `search_knowledge_base` (and optionally the ticket tools) and then writes a final answer based on the results.
6. **Persistence.** The answer is appended to the chat history file and summarised into that agent's long-term memory.
7. **Cleanup.** The MCP session and the model client are closed in a `finally` block.

---

## The MCP Server and Its Tools

`mcp_server.py` is a FastMCP server using the stdio transport. Tool names, parameters and descriptions are generated from each function's **type hints and docstring**, which is how the agents know what each tool is for.

| Tool                    | Parameters                                    | Purpose                                                                 |
| ----------------------- | --------------------------------------------- | ----------------------------------------------------------------------- |
| `search_knowledge_base` | `query: str`, `department: str`               | Finds the best-matching knowledge-base articles for a department.       |
| `create_support_ticket` | `department: str`, `summary: str`, `priority: str = "Medium"` | Creates a ticket (saved in `data/tickets.json`) and returns its id. |
| `get_ticket_status`     | `ticket_id: str`                              | Returns the department, priority, status and summary of a ticket.       |

Valid `department` values: `IT`, `Admin`, `HR`, `Accounts`, `Compliance`.
Valid `priority` values: `Low`, `Medium`, `High`, `Critical`.

The knowledge base is a Python dictionary inside `mcp_server.py` with four articles per department.

---

## Memory, Chat History and Logging

### Long-term memory
- Stored per agent in `data/long_term_memory.json`.
- After each answer, the responding agent saves a short note of the question and answer (capped at the latest 50 notes per agent).
- On every request, the notes are loaded into the agent's `ListMemory`, so agents can personalise replies using earlier conversations.
- Clear it from **Settings & Clear Chat → Clear long-term memory**.

### Chat history
- Every conversation is saved to `data/chat_history.json` with a title taken from the first question.
- The six most recent conversations appear under **Recent Conversations**.
- **Clear chat** starts a fresh conversation.

### Console log
- Terminal output (stderr) and the dashboard expander show the same trace, for example:

```text
10:39:27 | USER  My VPN keeps disconnecting. Can you help me resolve this?
10:39:27 | MCP   connected, tools: search_knowledge_base, create_support_ticket, get_ticket_status
10:39:28 | ROUTE selector -> IT_Agent (IT Support)
10:39:28 | AGENT IT_Agent calls MCP tool search_knowledge_base({"query": "VPN keeps disconnecting", "department": "IT"})
10:39:29 | AGENT IT_Agent answered: Here are some steps to fix VPN disconnections...
10:39:29 | MEMORY saved for IT agent (3 notes)
10:39:29 | CLIENT model client closed
```

- The MCP server writes its own lines (prefixed `MCP-SERVER`) to the terminal.

---

## Customization

### Add or edit knowledge-base articles
Open `mcp_server.py` and edit the `KNOWLEDGE_BASE` dictionary. Each article has a `title`, a list of `keywords` and the `content` text.

### Add a new MCP tool
1. In `mcp_server.py`, write a function with **type hints and a clear docstring**.
2. Add it to the tuple in the registration loop near the bottom:

```python
for _tool in (search_knowledge_base, create_support_ticket, get_ticket_status, my_new_tool):
    mcp.tool()(_tool)
```

3. (Optional) Mention the new tool in the agents' `system_message` in `customer_support.py` so they know when to use it.

### Add a new department agent
1. Add an entry to `DEPARTMENTS` in `customer_support.py` (key, agent name, label, icon, description).
2. Add the department key to `VALID_DEPARTMENTS` and add its articles to `KNOWLEDGE_BASE` in `mcp_server.py`.

### Change the model
Set `OPENAI_MODEL` in `.env` (for example `gpt-4o`).

### Connect to a real data source
Replace the body of `search_knowledge_base` in `mcp_server.py` with a call to your vector database, wiki, or ticketing system. The agents and dashboard need no changes.

### Use an existing external MCP server
In `handle_query()`, point `StdioServerParams` (or `SseServerParams`) at your own server instead of `mcp_server.py`.

---

## Tech Stack

| Component        | Technology                                               |
| ---------------- | -------------------------------------------------------- |
| Agent framework  | AutoGen (`autogen-agentchat`, `autogen-ext`)             |
| Multi-agent routing | `SelectorGroupChat`                                   |
| Tool protocol    | Model Context Protocol (FastMCP server, stdio transport) |
| LLM provider     | OpenAI (`OpenAIChatCompletionClient`)                    |
| Frontend         | Streamlit                                                |
| Configuration    | `python-dotenv`                                          |
| Storage          | Local JSON files in `data/`                              |
