"""Helpers for using MCP servers with dwagents.

Two utilities that cover the common pattern of "connect to one or more MCP
servers, then wrap the resulting tools so transient failures don't kill the
run":

- :func:`connect_mcp` is a thin retry-wrapper around
  ``MultiServerMCPClient.get_tools()``. Its ``servers`` argument is the same
  dict shape that ``MultiServerMCPClient`` accepts, so users who outgrow this
  helper can switch to the underlying client with no code changes.
- :func:`wrap_with_retry` wraps a :class:`~langchain_core.tools.BaseTool` so
  transient errors retry with jittered backoff and a terminal failure returns
  an ``"Error: …"`` string instead of raising — which keeps an agent's run
  alive and surfaces the failure through the normal tool-result channel.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient


def wrap_with_retry(tool: BaseTool, *, max_retries: int = 5) -> BaseTool:
    """Wrap a tool so transient failures retry; a terminal failure returns an error string.

    The wrapped tool keeps the original's name, description, and ``args_schema``
    so the LLM sees it as the same tool. On failure after ``max_retries``
    attempts, the wrapper returns ``"Error: <tool> failed after N attempts: …"``
    instead of raising, so the agent loop can continue and the error surfaces
    as a normal tool result.
    """

    async def _acoroutine(**kwargs: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                return await tool.ainvoke(kwargs)
            except Exception as e:
                last_error = e
                wait = 2 * attempt + random.uniform(0, 2)
                print(
                    f"  tool '{tool.name}' attempt {attempt}/{max_retries} "
                    f"failed ({str(e)[:80]}), retrying in {wait:.1f}s...",
                    flush=True,
                )
                await asyncio.sleep(wait)
        return f"Error: tool '{tool.name}' failed after {max_retries} attempts: {last_error}"

    def _sync(**kwargs: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                return tool.invoke(kwargs)
            except Exception as e:
                last_error = e
                wait = 2 * attempt + random.uniform(0, 2)
                print(
                    f"  tool '{tool.name}' attempt {attempt}/{max_retries} "
                    f"failed ({str(e)[:80]}), retrying in {wait:.1f}s...",
                    flush=True,
                )
                time.sleep(wait)
        return f"Error: tool '{tool.name}' failed after {max_retries} attempts: {last_error}"

    return StructuredTool.from_function(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        coroutine=_acoroutine,
        func=_sync,
    )


async def connect_mcp(
    servers: dict[str, dict[str, Any]],
    *,
    max_retries: int = 5,
) -> list[BaseTool]:
    """Connect to one or more MCP servers and return all of their tools.

    Args:
        servers: Mapping of server name to the server config dict accepted by
            ``langchain_mcp_adapters.client.MultiServerMCPClient``. Example::

                {
                    "files": {"transport": "streamable_http", "url": "https://…/mcp"},
                }

        max_retries: How many times to retry the initial connection on failure.

    Returns:
        Flat list of all tools exposed across the configured servers.

    Raises:
        The last connection exception if every attempt fails.
    """
    for attempt in range(1, max_retries + 1):
        try:
            client = MultiServerMCPClient(servers)
            return await client.get_tools()
        except Exception as e:
            print(
                f"MCP connection attempt {attempt}/{max_retries} failed: {e}",
                flush=True,
            )
            if attempt == max_retries:
                raise
            await asyncio.sleep(2 * attempt)
    # Unreachable — the loop either returns or raises on the last attempt.
    raise RuntimeError("connect_mcp exited retry loop without resolving")
