"""Tests for the model wrappers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from dwagents.models.batch import (
    ChatDoublewordBatch,
    EmptyLLMResponseError,
    _completion_to_chat_result,
    _is_empty_generation,
    _messages_to_openai,
    _parse_tool_calls,
)
from dwagents.models.realtime import create_realtime_model


class TestMessagesToOpenAI:
    def test_system_message(self):
        msgs = [SystemMessage(content="You are helpful.")]
        result = _messages_to_openai(msgs)
        assert result == [{"role": "system", "content": "You are helpful."}]

    def test_human_message(self):
        msgs = [HumanMessage(content="Hello")]
        result = _messages_to_openai(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_ai_message_plain(self):
        msgs = [AIMessage(content="Hi there")]
        result = _messages_to_openai(msgs)
        assert result == [{"role": "assistant", "content": "Hi there"}]

    def test_ai_message_with_tool_calls(self):
        msgs = [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "search", "args": {"query": "test"}}
                ],
            )
        ]
        result = _messages_to_openai(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert len(result[0]["tool_calls"]) == 1
        tc = result[0]["tool_calls"][0]
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "search"

    def test_tool_message(self):
        msgs = [ToolMessage(content="result", tool_call_id="call_1")]
        result = _messages_to_openai(msgs)
        assert result == [
            {"role": "tool", "tool_call_id": "call_1", "content": "result"}
        ]

    def test_mixed_conversation(self):
        msgs = [
            SystemMessage(content="system"),
            HumanMessage(content="user"),
            AIMessage(content="assistant"),
        ]
        result = _messages_to_openai(msgs)
        assert len(result) == 3
        assert [m["role"] for m in result] == ["system", "user", "assistant"]


class TestParseToolCalls:
    def test_single_tool_call(self):
        raw = [
            MagicMock(
                id="call_1",
                function=MagicMock(name="search", arguments='{"query": "test"}'),
            )
        ]
        # MagicMock.name is special, set it explicitly
        raw[0].function.name = "search"
        result = _parse_tool_calls(raw)
        assert len(result) == 1
        assert result[0]["id"] == "call_1"
        assert result[0]["name"] == "search"
        assert result[0]["args"] == {"query": "test"}


class TestCompletionToChatResult:
    def test_plain_response(self):
        completion = MagicMock()
        completion.choices = [
            MagicMock(
                message=MagicMock(content="Hello!", tool_calls=None),
                finish_reason="stop",
            )
        ]
        completion.model = "gpt-4o"
        completion.usage = MagicMock()
        completion.usage.__iter__ = MagicMock(
            return_value=iter([("prompt_tokens", 5), ("completion_tokens", 3)])
        )

        result = _completion_to_chat_result(completion)
        assert len(result.generations) == 1
        assert result.generations[0].message.content == "Hello!"
        assert result.generations[0].message.tool_calls == []

    def test_tool_call_response(self):
        func_mock = MagicMock()
        func_mock.name = "calculator"
        func_mock.arguments = '{"expression": "2+2"}'

        tc_mock = MagicMock(id="call_1", function=func_mock)

        completion = MagicMock()
        completion.choices = [
            MagicMock(
                message=MagicMock(content="", tool_calls=[tc_mock]),
                finish_reason="tool_calls",
            )
        ]
        completion.model = "gpt-4o"
        completion.usage = None

        result = _completion_to_chat_result(completion)
        msg = result.generations[0].message
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "calculator"
        assert msg.tool_calls[0]["args"] == {"expression": "2+2"}


class TestOpenAICompatPatch:
    """The doubleword batch API returns finish_reason=None on some tool-calling
    responses. Without the patch, autobatcher's parse loop raises and orphans
    every sibling request in the batch as 'No result for request X'."""

    def test_none_finish_reason_coerced_to_tool_calls_when_tool_call_present(self):
        from openai.types.chat import ChatCompletion

        payload = {
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                }
            ],
        }
        parsed = ChatCompletion.model_validate(payload)
        assert parsed.choices[0].finish_reason == "tool_calls"

    def test_none_finish_reason_coerced_to_stop_without_tool_calls(self):
        from openai.types.chat import ChatCompletion

        payload = {
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": None,
                    "message": {"role": "assistant", "content": "hi"},
                }
            ],
        }
        parsed = ChatCompletion.model_validate(payload)
        assert parsed.choices[0].finish_reason == "stop"

    def test_valid_finish_reason_untouched(self):
        from openai.types.chat import ChatCompletion

        payload = {
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "message": {"role": "assistant", "content": "hi"},
                }
            ],
        }
        parsed = ChatCompletion.model_validate(payload)
        assert parsed.choices[0].finish_reason == "length"


class TestRealtimeModel:
    def test_creates_chat_openai(self):
        with patch.dict("os.environ", {"DOUBLEWORD_API_KEY": "test-key"}):
            model = create_realtime_model(model="gpt-4o")
            assert model.model_name == "gpt-4o"


class TestChatDoublewordBatch:
    def test_llm_type(self):
        model = ChatDoublewordBatch(
            model_name="gpt-4o", api_key="test", base_url="http://test/v1/"
        )
        assert model._llm_type == "doubleword-batch"

    def test_bind_tools(self):
        from langchain_core.tools import tool

        @tool
        def dummy(x: str) -> str:
            """A dummy tool."""
            return x

        model = ChatDoublewordBatch(
            model_name="gpt-4o", api_key="test", base_url="http://test/v1/"
        )
        bound = model.bind_tools([dummy])
        assert len(bound._tools) == 1
        assert bound._tools[0]["function"]["name"] == "dummy"

    def test_build_request_kwargs(self):
        model = ChatDoublewordBatch(
            model_name="gpt-4o", api_key="test", base_url="http://test/v1/"
        )
        msgs = [HumanMessage(content="Hello")]
        kwargs = model._build_request_kwargs(msgs, stop=["END"])
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["messages"] == [{"role": "user", "content": "Hello"}]
        assert kwargs["stop"] == ["END"]


class TestEmptyResponseRetry:
    """An empty AIMessage (no content, no tool_calls) silently terminates
    LangGraph's agent loop. _agenerate must retry once and raise on persistent
    emptiness so sibling agents can keep running and the failure surfaces."""

    @staticmethod
    def _empty_completion():
        c = MagicMock()
        c.choices = [
            MagicMock(
                message=MagicMock(content=None, tool_calls=None),
                finish_reason="stop",
            )
        ]
        c.model = "m"
        c.usage = None
        return c

    @staticmethod
    def _good_completion():
        c = MagicMock()
        c.choices = [
            MagicMock(
                message=MagicMock(content="hello", tool_calls=None),
                finish_reason="stop",
            )
        ]
        c.model = "m"
        c.usage = None
        return c

    @pytest.mark.asyncio
    async def test_empty_then_success_retries_and_returns(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create = AsyncMock(
            side_effect=[self._empty_completion(), self._good_completion()]
        )
        model = ChatDoublewordBatch(
            model_name="m",
            api_key="k",
            base_url="http://x/",
            client=fake_client,
        )

        result = await model._agenerate([HumanMessage(content="hi")])
        assert result.generations[0].message.content == "hello"
        assert fake_client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_two_empty_responses_raise(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create = AsyncMock(
            side_effect=[self._empty_completion(), self._empty_completion()]
        )
        model = ChatDoublewordBatch(
            model_name="m",
            api_key="k",
            base_url="http://x/",
            client=fake_client,
        )

        with pytest.raises(EmptyLLMResponseError):
            await model._agenerate([HumanMessage(content="hi")])
        assert fake_client.chat.completions.create.await_count == 2

    @pytest.mark.asyncio
    async def test_first_response_non_empty_no_retry(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create = AsyncMock(
            return_value=self._good_completion()
        )
        model = ChatDoublewordBatch(
            model_name="m",
            api_key="k",
            base_url="http://x/",
            client=fake_client,
        )

        await model._agenerate([HumanMessage(content="hi")])
        assert fake_client.chat.completions.create.await_count == 1


class TestIsEmptyGeneration:
    def test_detects_both_missing(self):
        from langchain_core.outputs import ChatGeneration, ChatResult

        result = ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[]))]
        )
        assert _is_empty_generation(result)

    def test_whitespace_only_is_empty(self):
        from langchain_core.outputs import ChatGeneration, ChatResult

        result = ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content="   \n\t", tool_calls=[]))
            ]
        )
        assert _is_empty_generation(result)

    def test_false_with_content(self):
        from langchain_core.outputs import ChatGeneration, ChatResult

        result = ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="hi", tool_calls=[]))]
        )
        assert not _is_empty_generation(result)

    def test_false_with_tool_calls(self):
        from langchain_core.outputs import ChatGeneration, ChatResult

        result = ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="",
                        tool_calls=[{"id": "1", "name": "f", "args": {}}],
                    )
                )
            ]
        )
        assert not _is_empty_generation(result)

    def test_false_with_multimodal_content_list(self):
        from langchain_core.outputs import ChatGeneration, ChatResult

        result = ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=[{"type": "text", "text": "hi"}],
                        tool_calls=[],
                    )
                )
            ]
        )
        assert not _is_empty_generation(result)


