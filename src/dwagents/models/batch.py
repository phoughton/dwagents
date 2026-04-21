from __future__ import annotations

import asyncio
from typing import Any

from autobatcher import BatchOpenAI
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field, PrivateAttr

from dwagents.config import settings
from dwagents.models._openai_compat import install_patch as _install_openai_compat

_install_openai_compat()


class EmptyLLMResponseError(RuntimeError):
    """Raised when the batch LLM returns empty content and no tool calls twice in a row.

    LangGraph's agent loop treats an `AIMessage(content="", tool_calls=[])` as a
    terminal state and silently exits. `_agenerate` retries once on an empty
    response; if the second attempt is also empty, this is raised so the failure
    surfaces instead of the agent stopping silently.
    """


def _messages_to_openai(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Convert LangChain messages to OpenAI-format dicts."""
    result = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            entry: dict[str, Any] = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["args"]
                            if isinstance(tc["args"], str)
                            else __import__("json").dumps(tc["args"]),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            result.append(entry)
        elif isinstance(msg, ToolMessage):
            result.append({
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            })
        else:
            result.append({"role": "user", "content": str(msg.content)})
    return result


def _parse_tool_calls(raw_tool_calls: list) -> list[dict[str, Any]]:
    """Convert OpenAI tool_calls to LangChain format."""
    import json

    result = []
    for tc in raw_tool_calls:
        func = tc.function
        args_str = func.arguments
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = args_str
        result.append({
            "id": tc.id,
            "name": func.name,
            "args": args,
        })
    return result


def _is_empty_generation(result: ChatResult) -> bool:
    """True iff the result has no text content and no tool calls.

    LangGraph's tools-edge routes to END whenever `tool_calls` is empty,
    so an AIMessage with neither content nor tool_calls terminates the
    agent loop silently — callers need to detect and recover from this.
    """
    if not result.generations:
        return True
    msg = result.generations[0].message
    content = msg.content
    if isinstance(content, str):
        has_content = bool(content.strip())
    else:
        has_content = bool(content)
    return not has_content and not getattr(msg, "tool_calls", None)


def _completion_to_chat_result(completion) -> ChatResult:
    """Convert an OpenAI ChatCompletion to a LangChain ChatResult."""
    choice = completion.choices[0]
    message = choice.message

    tool_calls = []
    if message.tool_calls:
        tool_calls = _parse_tool_calls(message.tool_calls)

    ai_message = AIMessage(
        content=message.content or "",
        tool_calls=tool_calls,
        response_metadata={
            "finish_reason": choice.finish_reason,
            "model": completion.model,
        },
    )

    generation = ChatGeneration(message=ai_message)
    return ChatResult(
        generations=[generation],
        llm_output={
            "model": completion.model,
            "usage": dict(completion.usage) if completion.usage else {},
        },
    )


class ChatDoublewordBatch(BaseChatModel):
    """LangChain chat model wrapping doubleword.ai's autobatcher.

    All requests go through BatchOpenAI, which transparently collects
    them into batch API calls for 50-75% cost savings.
    """

    model_name: str = Field(default="")
    api_key: str = Field(default="")
    base_url: str = Field(default="")
    batch_window_seconds: float = Field(default=10.0)
    batch_size: int = Field(default=1000)
    poll_interval_seconds: float = Field(default=5.0)
    completion_window: str = Field(default="24h")

    _client: BatchOpenAI | None = PrivateAttr(default=None)
    _tools: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    def __init__(self, *, client: BatchOpenAI | None = None, **kwargs):
        # Apply defaults from settings for any missing values
        kwargs.setdefault("model_name", settings.model)
        kwargs.setdefault("api_key", settings.api_key)
        kwargs.setdefault("base_url", settings.base_url)
        kwargs.setdefault("batch_window_seconds", settings.batch_window_seconds)
        kwargs.setdefault("batch_size", settings.batch_size)
        kwargs.setdefault("poll_interval_seconds", settings.poll_interval_seconds)
        kwargs.setdefault("completion_window", settings.completion_window)
        super().__init__(**kwargs)
        if client is not None:
            self._client = client

    def _get_client(self) -> BatchOpenAI:
        if self._client is None:
            self._client = BatchOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                batch_size=self.batch_size,
                batch_window_seconds=self.batch_window_seconds,
                poll_interval_seconds=self.poll_interval_seconds,
                completion_window=self.completion_window,
            )
        return self._client

    @property
    def _llm_type(self) -> str:
        return "doubleword-batch"

    def bind_tools(self, tools: list, **kwargs) -> "ChatDoublewordBatch":
        """Bind tool schemas for inclusion in requests."""
        from langchain_core.utils.function_calling import convert_to_openai_tool

        bound = self.model_copy()
        bound._tools = [convert_to_openai_tool(t) for t in tools]
        return bound

    def _build_request_kwargs(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        openai_messages = _messages_to_openai(messages)
        request: dict[str, Any] = {
            "model": self.model_name,
            "messages": openai_messages,
        }
        if stop:
            request["stop"] = stop
        if self._tools:
            request["tools"] = self._tools
        request.update(kwargs)
        return request

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        """Sync wrapper — runs the async path in a new event loop.

        If called from within a running event loop (e.g., LangGraph inside
        an async context), runs the coroutine in a separate thread to avoid
        nested event loop issues.
        """
        coro = self._agenerate(messages, stop=stop, **kwargs)

        try:
            asyncio.get_running_loop()
            # We're inside a running loop — run in a new thread with its own loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            # No running loop — safe to use asyncio.run directly
            return asyncio.run(coro)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        """Primary async path — sends request through BatchOpenAI.

        Retries once if the backend returns an empty choice (no content,
        no tool_calls). Without this, LangGraph's agent loop would treat
        the empty AIMessage as a terminal state and silently exit.
        """
        client = self._get_client()
        request_kwargs = self._build_request_kwargs(messages, stop, **kwargs)

        completion = await client.chat.completions.create(**request_kwargs)
        result = _completion_to_chat_result(completion)
        if not _is_empty_generation(result):
            return result

        if run_manager is not None:
            await run_manager.on_text(
                "dwagents: empty LLM response, retrying once", verbose=True
            )
        completion = await client.chat.completions.create(**request_kwargs)
        result = _completion_to_chat_result(completion)
        if not _is_empty_generation(result):
            return result

        raise EmptyLLMResponseError(
            "Batch LLM returned empty content and no tool calls on two "
            f"consecutive attempts (model={self.model_name}, "
            f"messages={len(messages)}). This usually indicates a "
            "transient backend issue; re-running the prompt may succeed."
        )

    async def aclose(self) -> None:
        """Close the underlying BatchOpenAI client."""
        if self._client is not None:
            await self._client.close()
            self._client = None
