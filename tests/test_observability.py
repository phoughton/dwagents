"""Tests for dwagents.observability — ToolCallLogger + print_message_trail."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from dwagents.observability import ToolCallLogger, print_message_trail


def _llm_result_with(message: AIMessage) -> LLMResult:
    return LLMResult(generations=[[ChatGeneration(message=message)]])


class TestToolCallLogger:
    def test_on_llm_end_logs_tool_calls_when_present(self, capsys):
        logger = ToolCallLogger("agent-a")
        msg = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "search", "args": {"q": "dogs"}}],
        )
        logger.on_llm_end(_llm_result_with(msg))

        out = capsys.readouterr().out
        assert "[agent-a]" in out
        assert "tool_call search" in out
        assert "dogs" in out

    def test_on_llm_end_logs_text_when_no_tool_calls(self, capsys):
        logger = ToolCallLogger("agent-b")
        msg = AIMessage(content="Here is your answer: 42.")
        logger.on_llm_end(_llm_result_with(msg))

        out = capsys.readouterr().out
        assert "[agent-b]" in out
        assert "LLM -> text" in out
        assert "42" in out

    def test_on_llm_end_notes_empty_response(self, capsys):
        logger = ToolCallLogger("agent-c")
        msg = AIMessage(content="")
        logger.on_llm_end(_llm_result_with(msg))

        out = capsys.readouterr().out
        assert "empty response" in out

    def test_on_tool_end_flags_error_strings(self, capsys):
        logger = ToolCallLogger("agent-d")
        logger.on_tool_end("Error: something exploded", name="writer")

        out = capsys.readouterr().out
        assert "ERR" in out
        assert "writer" in out

    def test_on_tool_start_logs_name_and_args(self, capsys):
        logger = ToolCallLogger("agent-e")
        logger.on_tool_start({"name": "calc"}, "2 + 2")

        out = capsys.readouterr().out
        assert "-> tool calc" in out
        assert "2 + 2" in out

    def test_on_tool_error_logs_exception(self, capsys):
        logger = ToolCallLogger("agent-f")
        logger.on_tool_error(RuntimeError("kaboom"), name="t")

        out = capsys.readouterr().out
        assert "!! tool t" in out
        assert "RuntimeError" in out
        assert "kaboom" in out


class TestPrintMessageTrail:
    def test_renders_full_trail(self, capsys):
        messages = [
            HumanMessage(content="please do X"),
            AIMessage(
                content="",
                tool_calls=[{"id": "1", "name": "do_x", "args": {"k": "v"}}],
            ),
            ToolMessage(content="done", tool_call_id="1"),
            AIMessage(content="All done."),
        ]
        print_message_trail("my-agent", messages)

        out = capsys.readouterr().out
        assert "[my-agent]" in out
        assert "USER" in out and "please do X" in out
        assert "AI  ->" in out and "do_x" in out
        assert "TOOL" in out and "done" in out
        assert "All done." in out

    def test_flags_error_tool_messages(self, capsys):
        print_message_trail(
            "x", [ToolMessage(content="Error: bad", tool_call_id="1")]
        )
        out = capsys.readouterr().out
        assert "ERR" in out
