"""Tests for dwagents.mcp — connect_mcp + wrap_with_retry."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from dwagents.mcp import connect_mcp, wrap_with_retry


class _Args(BaseModel):
    query: str


def _make_fake_tool(name: str, *, ainvoke=None, invoke=None) -> BaseTool:
    """Build a minimal BaseTool fake with controllable invoke behaviour."""
    tool = MagicMock(spec=BaseTool)
    tool.name = name
    tool.description = f"{name} tool"
    tool.args_schema = _Args
    tool.ainvoke = ainvoke or AsyncMock(return_value="ok")
    tool.invoke = invoke or MagicMock(return_value="ok")
    return tool


class TestWrapWithRetry:
    def test_async_success_on_first_try_no_retry(self):
        tool = _make_fake_tool("f", ainvoke=AsyncMock(return_value="ok"))
        wrapped = wrap_with_retry(tool, max_retries=3)

        result = asyncio.run(wrapped.ainvoke({"query": "x"}))

        assert result == "ok"
        tool.ainvoke.assert_awaited_once()

    def test_async_transient_then_success(self):
        call_count = {"n": 0}

        async def flaky(_):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise RuntimeError("boom")
            return "recovered"

        tool = _make_fake_tool("f", ainvoke=AsyncMock(side_effect=flaky))
        wrapped = wrap_with_retry(tool, max_retries=3)

        # Patch sleep so the test doesn't actually wait.
        with patch("dwagents.mcp.asyncio.sleep", new=AsyncMock()):
            result = asyncio.run(wrapped.ainvoke({"query": "x"}))

        assert result == "recovered"
        assert call_count["n"] == 2

    def test_async_terminal_failure_surfaced_as_error_string(self):
        tool = _make_fake_tool(
            "f", ainvoke=AsyncMock(side_effect=RuntimeError("nope"))
        )
        wrapped = wrap_with_retry(tool, max_retries=2)

        with patch("dwagents.mcp.asyncio.sleep", new=AsyncMock()):
            result = asyncio.run(wrapped.ainvoke({"query": "x"}))

        assert isinstance(result, str)
        assert result.startswith("Error: tool 'f' failed after 2 attempts")
        assert "nope" in result

    def test_sync_path_also_retries_and_surfaces(self):
        call_count = {"n": 0}

        def flaky(_):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise RuntimeError("boom")
            return "recovered"

        tool = _make_fake_tool("f", invoke=MagicMock(side_effect=flaky))
        wrapped = wrap_with_retry(tool, max_retries=3)

        with patch("dwagents.mcp.time.sleep", new=MagicMock()):
            result = wrapped.invoke({"query": "x"})

        assert result == "recovered"
        assert call_count["n"] == 2

    def test_preserves_name_description_and_schema(self):
        tool = _make_fake_tool("original_name")
        wrapped = wrap_with_retry(tool)

        assert wrapped.name == "original_name"
        assert wrapped.description == "original_name tool"
        assert wrapped.args_schema is _Args


class TestConnectMcp:
    def test_returns_tools_from_client(self):
        fake_tools = [_make_fake_tool("t1"), _make_fake_tool("t2")]
        mock_client = MagicMock()
        mock_client.get_tools = AsyncMock(return_value=fake_tools)

        with patch(
            "dwagents.mcp.MultiServerMCPClient", return_value=mock_client
        ) as MockClient:
            servers = {"s": {"transport": "streamable_http", "url": "http://x/"}}
            result = asyncio.run(connect_mcp(servers, max_retries=1))

        MockClient.assert_called_once_with(servers)
        assert result is fake_tools

    def test_retries_transient_then_succeeds(self):
        fake_tools = [_make_fake_tool("t1")]
        call_count = {"n": 0}

        def _client_factory(_servers):
            mock = MagicMock()

            async def _get_tools():
                call_count["n"] += 1
                if call_count["n"] < 2:
                    raise RuntimeError("network")
                return fake_tools

            mock.get_tools = _get_tools
            return mock

        with patch(
            "dwagents.mcp.MultiServerMCPClient", side_effect=_client_factory
        ), patch("dwagents.mcp.asyncio.sleep", new=AsyncMock()):
            result = asyncio.run(connect_mcp({"s": {}}, max_retries=3))

        assert result is fake_tools
        assert call_count["n"] == 2

    def test_raises_after_exhausting_retries(self):
        def _client_factory(_servers):
            mock = MagicMock()
            mock.get_tools = AsyncMock(side_effect=RuntimeError("always fails"))
            return mock

        with patch(
            "dwagents.mcp.MultiServerMCPClient", side_effect=_client_factory
        ), patch("dwagents.mcp.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(RuntimeError, match="always fails"):
                asyncio.run(connect_mcp({"s": {}}, max_retries=2))
