# dwagents

> **Unofficial package.** `dwagents` is a community-maintained integration and is **not** an official release from [doubleword.ai](https://doubleword.ai) or the [LangChain Deep Agents](https://github.com/langchain-ai/deepagents) project. It provides support for using both together, but any issues with `dwagents` itself should be filed against this repo — not against the upstream projects.

LangChain Deep Agents with [doubleword.ai](https://doubleword.ai) batch inference. All LLM calls go through doubleword.ai's autobatcher for 50-75% cost savings. Designed for background agents where cost matters more than latency.

Built on [LangChain Deep Agents](https://github.com/langchain-ai/deepagents) and Doubleword.ai's [autobatcher](https://github.com/doublewordai/autobatcher). Neither project endorses or maintains this package.

## Installation

```bash
# From PyPI:
pip install dwagents

# Or from source:
pip install git+https://github.com/phoughton/dwagents.git

# For development (after cloning the repo):
pip install -e ".[dev]"
```

## Configuration

Set your doubleword.ai API key as an environment variable:

```bash
export DOUBLEWORD_API_KEY="your-key-here"
```

Optional settings (with defaults):

```bash
export DOUBLEWORD_BASE_URL="https://api.doubleword.ai/v1/"
export DOUBLEWORD_MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
export DOUBLEWORD_BATCH_WINDOW_SECONDS="10.0"
export DOUBLEWORD_BATCH_SIZE="1000"
export DOUBLEWORD_POLL_INTERVAL_SECONDS="5.0"
export DOUBLEWORD_COMPLETION_WINDOW="1h"
```

Every setting above can also be overridden by a CLI flag on `dwagents run` (see `dwagents run --help` for the full list, or the `examples/parallel_agents.py` template) or by a Python kwarg via `model_kwargs={"model_name": ..., "batch_size": ..., ...}` passed to `create_supervisor` or `run_agents_parallel`. Precedence is **CLI flag > env var > built-in default**. `--api-key` is available but prefer the env var — CLI flags can leak into shell history and `ps` output.

## Usage

Most uses of this library want the **Parallel agents** pattern below or the **Command-line runner**. Skip to *Basic agent with tools* if you only need a single supervisor.

### Parallel agents sharing one batch window

`run_agents_parallel` spins up one supervisor per prompt and sends all of their LLM calls through a single shared batch client, so they collate into the same batch window. This is the high-leverage pattern for workloads where you have N independent tasks and want them all to ride one batch.

```python
import asyncio
from dwagents import (
    ToolCallLogger,
    connect_mcp,
    print_message_trail,
    run_agents_parallel,
    wrap_with_retry,
)
from dwagents.tools.example_tools import calculator, web_search


async def main():
    prompts = {
        "a": "What's the sum of the first 20 prime numbers?",
        "b": "Briefly explain the halting problem in one paragraph.",
    }

    # Optional — pull tools from one or more MCP servers instead of/on top of
    # bundled tools. Omit and just pass local tools if you don't need MCP.
    # mcp_tools = await connect_mcp({"files": {"transport": "streamable_http", "url": "…/mcp"}})
    # mcp_tools = [wrap_with_retry(t) for t in mcp_tools]

    results = await run_agents_parallel(
        prompts,
        tools=[web_search, calculator],
        system_prompt="You are a helpful assistant. Use tools when useful.",
        callbacks_factory=lambda name: [ToolCallLogger(name)],
    )
    for name, result in results.items():
        if isinstance(result, Exception):
            print(f"[{name}] FAILED: {result}")
            continue
        print_message_trail(name, result["messages"])


asyncio.run(main())
```

`ToolCallLogger` prints each LLM turn, tool call, and tool result prefixed with the agent name so concurrent activity stays legible. `print_message_trail` is a post-run walker that shows the full ordered message history per agent.

See `examples/parallel_agents.py` for a runnable template that loads prompt files from a directory and is easy to point at your own MCP server.

### Command-line runner

For the common case — "run every prompt file in a directory, in parallel, with logging" — there's a bundled CLI. After installing the package, it's on your `PATH` as `dwagents`:

```bash
# Runs with bundled example_tools (web_search, calculator)
dwagents run --prompts-dir examples/prompts

# Point at one or more MCP servers (repeat --mcp-server for multiple)
dwagents run \
    --prompts-dir examples/prompts \
    --mcp-server files=https://my.mcp.server/mcp \
    --system-prompt-file my_system_prompt.txt \
    --completion-window 1h
```

The CLI wires `ToolCallLogger` by default and prints a full message trail per agent. It's the fastest way to see whether a new MCP server, prompt set, or model is behaving. For anything beyond its shape, copy `examples/parallel_agents.py` and edit directly.

If a remote MCP server needs authentication, pass the bearer token via env var (recommended — doesn't leak into shell history) or flag, or use `--mcp-header` for non-bearer schemes. The `NAME` matches the name used in `--mcp-server NAME=URL` and binds credentials to that specific server:

```bash
# Safer: token from env, URL on the command line.
export DWAGENTS_MCP_BEARER_FILES="secret-token"
dwagents run --prompts-dir examples/prompts \
    --mcp-server files=https://mcp.example.com/mcp

# Or inline (leaks into shell history / ps output):
dwagents run --prompts-dir examples/prompts \
    --mcp-server files=https://mcp.example.com/mcp \
    --mcp-bearer-token files=secret-token

# Arbitrary headers for non-bearer schemes:
dwagents run --prompts-dir examples/prompts \
    --mcp-server crm=https://mcp.example.com/crm \
    --mcp-header crm=X-API-Key:abcdef
```

#### Local command tools

Expose a local CLI program as an agent tool with `--tool-command NAME=COMMAND`. The declared command and any agent-supplied `args` string are both tokenised with `shlex.split` and run via `subprocess` with `shell=False` — shell metacharacters (`|`, `>`, `*`, …) are **not** interpreted. Composes with `--mcp-server`.

```bash
# Expose git and ripgrep as tools; let the agents drive them.
dwagents run --prompts-dir examples/prompts \
    --tool-command git_log='git log --oneline -n 20' \
    --tool-command rg='rg --json' \
    --tool-command-description rg="Structured ripgrep search. Pass '-n PATTERN PATH' via args."
```

- `NAME` must be a valid Python identifier (e.g. `git_log`, not `git-log`).
- Agent-supplied arguments arrive through a single `args` string, which is appended to the declared command's argv.
- Per-tool timeout defaults to 30 s; override with `--tool-command-timeout NAME=SECONDS`.
- If you need pipes, redirection, or globs, wrap in a shell explicitly: `--tool-command pipeline="sh -c 'git log | head'"`.

### Basic agent with tools

```python
from langchain_core.tools import tool
from dwagents import create_supervisor


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # Replace with a real search implementation
    import requests
    resp = requests.get("https://api.example.com/search", params={"q": query})
    return resp.text


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression like '2 + 2' or '100 / 7'.
    """
    # Guard eval by stripping builtins so the tool can only do arithmetic.
    result = eval(expression, {"__builtins__": {}})
    return str(result)


agent = create_supervisor(
    tools=[web_search, calculator],
    system_prompt="You are a research assistant. Use tools to answer questions accurately.",
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "What is the population of France divided by 3?"}]
})
```

The agent will:
1. Call the LLM (autobatcher) to decide it needs `web_search`
2. Execute `web_search("population of France")`
3. Call the LLM (autobatcher) with the search result, decide it needs `calculator`
4. Execute `calculator("67390000 / 3")`
5. Call the LLM (autobatcher) to compose the final answer

### Custom tools with structured input

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from dwagents import create_supervisor


class DatabaseQuery(BaseModel):
    table: str = Field(description="The database table to query")
    filters: dict = Field(description="Column filters as key-value pairs")
    limit: int = Field(default=10, description="Max rows to return")


@tool(args_schema=DatabaseQuery)
def query_database(table: str, filters: dict, limit: int = 10) -> str:
    """Query a database table with filters.

    Use this to look up records in the application database.
    """
    # Replace with real database logic
    import json
    return json.dumps({
        "table": table,
        "filters": filters,
        "limit": limit,
        "results": [{"id": 1, "name": "example"}],
    })


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email notification.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Email body text.
    """
    # Replace with real email logic
    return f"Email sent to {to}"


agent = create_supervisor(
    tools=[query_database, send_email],
    system_prompt=(
        "You are an operations assistant. You can query the database "
        "and send email notifications when issues are found."
    ),
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Check for overdue orders and email the ops team"}]
})
```

### Using tools from an MCP server

Connect to any MCP server and use its tools as agent tools. For HTTP-based MCP servers you can use the bundled `connect_mcp` helper — it wraps `MultiServerMCPClient` with a connect-time retry loop and pairs well with `wrap_with_retry` for per-tool resilience:

```python
import asyncio
from dwagents import connect_mcp, create_supervisor, wrap_with_retry


async def main():
    tools = await connect_mcp({
        "files": {"transport": "streamable_http", "url": "https://my.mcp.server/mcp"},
    })
    tools = [wrap_with_retry(t) for t in tools]  # transient errors retry; terminal ones surface as "Error: …"

    agent = create_supervisor(
        tools=tools,
        system_prompt="You have access to filesystem tools over MCP.",
    )
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "List files in /data and summarise the first .txt"}]
    })
    print(result["messages"][-1].content)


asyncio.run(main())
```

For stdio-based MCP servers, or when you want full control, use `MultiServerMCPClient` directly. The example below mixes stdio and `streamable_http` transports, adds a plain tool, and shows how to pass a bearer token via `headers` for authenticated servers (SSE is also supported via `"transport": "sse"`):

```python
import asyncio
import os
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dwagents import create_supervisor


@tool
def notify_slack(channel: str, message: str) -> str:
    """Post a message to a Slack channel."""
    return f"Posted to #{channel}"


async def main():
    client = MultiServerMCPClient(
        {
            "filesystem": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/data"],
            },
            "crm": {
                "transport": "streamable_http",
                "url": "https://mcp.example.com/crm",
                # Bearer-token auth: set MCP_CRM_TOKEN in the environment.
                "headers": {"Authorization": f"Bearer {os.environ.get('MCP_CRM_TOKEN', '')}"},
            },
        }
    )
    mcp_tools = await client.get_tools()

    agent = create_supervisor(
        tools=mcp_tools + [notify_slack],
        system_prompt="You can read files, query the CRM, and post to Slack.",
    )

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Check the CSV files and notify #data-alerts of anything odd."}]
    })
    print(result["messages"][-1].content)


asyncio.run(main())
```

### Multi-agent with subagents

```python
from langchain_core.tools import tool
from dwagents import create_supervisor


@tool
def search_docs(query: str) -> str:
    """Search internal documentation."""
    return f"[Doc results for: {query}]"


@tool
def run_sql(query: str) -> str:
    """Run a read-only SQL query against the analytics database."""
    return f"[SQL results for: {query}]"


@tool
def create_chart(data: str, chart_type: str) -> str:
    """Create a chart from data.

    Args:
        data: JSON string of the data to plot.
        chart_type: One of 'bar', 'line', 'pie'.
    """
    return f"[Chart created: {chart_type}]"


agent = create_supervisor(
    tools=[],
    system_prompt=(
        "You are a supervisor that delegates research and analysis tasks. "
        "Use the researcher for finding information and the analyst for data work."
    ),
    subagents=[
        {
            "name": "researcher",
            "description": "Searches documentation and gathers information.",
            "system_prompt": "You search docs to find relevant information.",
            "tools": [search_docs],
        },
        {
            "name": "analyst",
            "description": "Queries databases and creates visualizations.",
            "system_prompt": "You run SQL queries and create charts from the results.",
            "tools": [run_sql, create_chart],
        },
    ],
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Show me a chart of monthly revenue trends"}]
})
```

Each subagent that doesn't specify its own `model` reuses the supervisor's `ChatDoublewordBatch`, so all their LLM calls collate into the same batch window.

### Overriding model settings

```python
from dwagents import create_supervisor

agent = create_supervisor(
    tools=[],
    system_prompt="You are a helpful assistant.",
    model_kwargs={
        "model_name": "gpt-4o-mini",
        "batch_window_seconds": 5.0,
        "completion_window": "1h",
    },
)
```

### Using the models directly

```python
from dwagents.models import ChatDoublewordBatch, create_realtime_model

# Batch model (all calls go through autobatcher — default for agents)
batch_model = ChatDoublewordBatch(model_name="gpt-4o")

# Real-time model (standard ChatOpenAI pointed at doubleword.ai, no batching)
realtime_model = create_realtime_model(model="gpt-4o")
```

## How it works

`ChatDoublewordBatch` is a LangChain `BaseChatModel` that wraps doubleword.ai's [autobatcher](https://github.com/doublewordai/autobatcher). All LLM calls are transparently collected and submitted as batch API calls:

1. Requests accumulate over a configurable time window (default 10s)
2. When the window closes (or batch size limit is hit), they're submitted as a single batch
3. Results are polled and returned to callers as they complete

This gives 50-75% cost savings compared to real-time API calls. The trade-off is latency (~10s+ per call), which is acceptable for background agents.

## Tests

```bash
pytest -v
```

(`testpaths = ["tests"]` is configured in `pyproject.toml`, so `pytest` alone picks up the test suite.)

## Docker Container

Here is the [Dockerfile](https://github.com/phoughton/python_dev_container) for this template repo (in a separate repo).
