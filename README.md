# dwagents

LangChain Deep Agents with [doubleword.ai](https://doubleword.ai) batch inference. All LLM calls go through doubleword.ai's autobatcher for 50-75% cost savings. Designed for background agents where cost matters more than latency.

Built on [LangChain Deep Agents](https://github.com/langchain-ai/deepagents) and [autobatcher](https://github.com/doublewordai/autobatcher).

## Installation

```bash
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
export DOUBLEWORD_MODEL="gpt-4o"
export DOUBLEWORD_BATCH_WINDOW_SECONDS="10.0"
export DOUBLEWORD_BATCH_SIZE="1000"
export DOUBLEWORD_COMPLETION_WINDOW="24h"
```

## Usage

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

Connect to any MCP server and use its tools as agent tools. This example uses a stdio server, but HTTP (`streamable_http`) and SSE (`sse`) transports also work.

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from dwagents import create_supervisor


async def main():
    client = MultiServerMCPClient(
        {
            "filesystem": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/workspace"],
            },
        }
    )
    tools = await client.get_tools()

    agent = create_supervisor(
        tools=tools,
        system_prompt=(
            "You are a file management assistant. "
            "Use the filesystem tools to read, write, and organize files."
        ),
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": "List the files in /tmp/workspace and summarize any .txt files"}]
    })
    print(result["messages"][-1].content)


asyncio.run(main())
```

You can connect to multiple MCP servers at once and mix MCP tools with regular tools:

```python
import asyncio
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dwagents import create_supervisor


@tool
def notify_slack(channel: str, message: str) -> str:
    """Post a message to a Slack channel.

    Args:
        channel: The Slack channel name.
        message: The message to post.
    """
    return f"Posted to #{channel}"


async def main():
    client = MultiServerMCPClient(
        {
            "filesystem": {
                "transport": "stdio",
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/data"],
            },
            "database": {
                "transport": "streamable_http",
                "url": "http://localhost:8080/mcp",
            },
        }
    )
    mcp_tools = await client.get_tools()

    agent = create_supervisor(
        tools=mcp_tools + [notify_slack],
        system_prompt="You are a data ops agent with access to files, a database, and Slack.",
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": "Check the CSV files for anomalies, query the database for context, and post a summary to #data-alerts"}]
    })
    print(result["messages"][-1].content)


asyncio.run(main())
```

### Streamable HTTP with bearer token authentication

When connecting to a remote MCP server over HTTP that requires authentication, pass a bearer token in the `headers` field:

```python
import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from dwagents import create_supervisor


async def main():
    client = MultiServerMCPClient(
        {
            "inventory": {
                "transport": "streamable_http",
                "url": "https://mcp.example.com/inventory",
                "headers": {
                    "Authorization": f"Bearer {os.environ['MCP_INVENTORY_TOKEN']}",
                },
            },
            "crm": {
                "transport": "streamable_http",
                "url": "https://mcp.example.com/crm",
                "headers": {
                    "Authorization": f"Bearer {os.environ['MCP_CRM_TOKEN']}",
                },
            },
        }
    )
    tools = await client.get_tools()

    agent = create_supervisor(
        tools=tools,
        system_prompt=(
            "You are a supply chain agent. Use the inventory tools to check "
            "stock levels and the CRM tools to look up customer orders."
        ),
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": "Check if order #4521 can be fulfilled from current stock"}]
    })
    print(result["messages"][-1].content)


asyncio.run(main())
```

Set the tokens as environment variables:

```bash
export MCP_INVENTORY_TOKEN="your-inventory-api-token"
export MCP_CRM_TOKEN="your-crm-api-token"
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

Each subagent gets its own `ChatDoublewordBatch` instance, so batch windows are independent.

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
pytest tests/ -v
```

## Docker Container

Here is the [Dockerfile](https://github.com/phoughton/python_dev_container) for this template repo (in a separate repo).
