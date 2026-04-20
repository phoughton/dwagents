"""Per-agent tool-call logging and post-run message-trail printing.

:class:`ToolCallLogger` is a :class:`~langchain_core.callbacks.BaseCallbackHandler`
that prints every LLM boundary, tool invocation, tool result, and tool error
with a stable ``[agent_name]`` prefix so concurrent agents' activity stays
legible when interleaved. :func:`print_message_trail` walks the final message
list after a run for a full post-mortem.

Typical wiring with :func:`dwagents.run_agents_parallel`::

    from dwagents import ToolCallLogger, print_message_trail, run_agents_parallel

    results = await run_agents_parallel(
        prompts,
        tools=tools,
        callbacks_factory=lambda name: [ToolCallLogger(name)],
    )
    for name, result in results.items():
        if isinstance(result, Exception):
            print(f"[{name}] FAILED: {result}")
            continue
        print_message_trail(name, result["messages"])
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

_PREVIEW_LEN = 300


def _preview(text: Any, limit: int = _PREVIEW_LEN) -> str:
    s = text if isinstance(text, str) else repr(text)
    s = s.replace("\n", " ")
    return s if len(s) <= limit else s[:limit] + f"... ({len(s)} chars)"


class ToolCallLogger(BaseCallbackHandler):
    """Print per-agent LLM and tool activity to stdout."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name

    def _log(self, msg: str) -> None:
        print(f"[{self.agent_name}] {msg}", flush=True)

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        turn = len(messages[0]) if messages else 0
        self._log(f"LLM call -> {turn} messages in history")

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self._log(f"LLM call (text) -> {len(prompts)} prompt(s)")

    def on_llm_end(self, response, **kwargs) -> None:
        try:
            gen = response.generations[0][0]
            message = getattr(gen, "message", None)
        except (IndexError, AttributeError):
            return
        if message is None:
            return
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            for tc in tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                self._log(f"  LLM -> tool_call {name}({_preview(args)})")
        else:
            content = getattr(message, "content", "") or ""
            if content.strip():
                self._log(f"  LLM -> text: {_preview(content)}")
            else:
                self._log("  LLM -> (empty response, no tool calls)")

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        name = (serialized or {}).get("name", "?")
        self._log(f"  -> tool {name}({_preview(input_str)})")

    def on_tool_end(self, output, **kwargs) -> None:
        name = kwargs.get("name", "?")
        text = output if isinstance(output, str) else getattr(output, "content", repr(output))
        prefix = "ERR " if isinstance(text, str) and text.startswith("Error:") else ""
        self._log(f"  <- tool {name} {prefix}{_preview(text)}")

    def on_tool_error(self, error, **kwargs) -> None:
        name = kwargs.get("name", "?")
        self._log(f"  !! tool {name} raised {type(error).__name__}: {error}")


def print_message_trail(agent_name: str, messages: list[BaseMessage]) -> None:
    """Print the complete ordered message trail for one agent.

    Useful as a post-mortem when you want to see exactly which tools each agent
    called, in what order, and what they returned. Pairs well with
    :class:`ToolCallLogger` — the logger shows events as they happen, this
    shows the final shape.
    """
    print(f"\n--- [{agent_name}] message trail ({len(messages)} messages) ---", flush=True)
    for i, m in enumerate(messages):
        if isinstance(m, HumanMessage):
            print(f"  {i:02d} USER: {_preview(m.content)}")
        elif isinstance(m, AIMessage):
            content = (m.content or "").strip()
            if content:
                print(f"  {i:02d} AI  : {_preview(content)}")
            tool_calls = m.tool_calls or []
            for tc in tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?")
                args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                print(f"  {i:02d} AI  -> {name}({_preview(args)})")
            if not content and not tool_calls:
                print(f"  {i:02d} AI  : (empty)")
        elif isinstance(m, ToolMessage):
            prefix = "ERR " if isinstance(m.content, str) and m.content.startswith("Error:") else ""
            print(f"  {i:02d} TOOL: {prefix}{_preview(m.content)}")
        else:
            print(f"  {i:02d} {type(m).__name__}: {_preview(getattr(m, 'content', repr(m)))}")
    print(f"--- [{agent_name}] end ---\n", flush=True)
