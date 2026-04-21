"""Tests for dwagents.tools.commands — local-command agent tools."""

from __future__ import annotations

import asyncio

import pytest

from dwagents.tools.commands import (
    DEFAULT_TIMEOUT_SECONDS,
    _collect_tool_command_metadata,
    _parse_tool_command_specs,
    build_command_tools,
)


class TestParseToolCommandSpecs:
    def test_parses_single(self):
        assert _parse_tool_command_specs(["echo=echo hello"]) == {"echo": "echo hello"}

    def test_parses_multiple(self):
        result = _parse_tool_command_specs(["a=true", "b=false"])
        assert result == {"a": "true", "b": "false"}

    def test_command_may_contain_equals(self):
        # Only the first '=' splits name from command.
        assert _parse_tool_command_specs(["set=env X=Y"]) == {"set": "env X=Y"}

    def test_rejects_missing_equals(self):
        with pytest.raises(SystemExit):
            _parse_tool_command_specs(["nope"])

    def test_rejects_empty_name(self):
        with pytest.raises(SystemExit):
            _parse_tool_command_specs(["=echo hi"])

    def test_rejects_empty_command(self):
        with pytest.raises(SystemExit):
            _parse_tool_command_specs(["echo="])

    def test_rejects_non_identifier_name(self):
        with pytest.raises(SystemExit) as exc:
            _parse_tool_command_specs(["bad-name=echo hi"])
        assert "identifier" in str(exc.value)

    def test_rejects_name_starting_with_digit(self):
        with pytest.raises(SystemExit):
            _parse_tool_command_specs(["1bad=echo hi"])

    def test_rejects_duplicate_name(self):
        with pytest.raises(SystemExit) as exc:
            _parse_tool_command_specs(["echo=echo a", "echo=echo b"])
        assert "more than once" in str(exc.value)


class TestCollectToolCommandMetadata:
    def test_defaults_when_no_metadata(self):
        desc, timeout = _collect_tool_command_metadata(
            allowed_names={"echo"}, descriptions=[], timeouts=[]
        )
        assert desc == {}
        assert timeout == {}

    def test_description_and_timeout_parsed(self):
        desc, timeout = _collect_tool_command_metadata(
            allowed_names={"echo"},
            descriptions=["echo=say hello"],
            timeouts=["echo=5.0"],
        )
        assert desc == {"echo": "say hello"}
        assert timeout == {"echo": 5.0}

    def test_description_for_unknown_tool_raises(self):
        with pytest.raises(SystemExit) as exc:
            _collect_tool_command_metadata(
                allowed_names={"echo"},
                descriptions=["nope=hi"],
                timeouts=[],
            )
        assert "nope" in str(exc.value)

    def test_timeout_for_unknown_tool_raises(self):
        with pytest.raises(SystemExit) as exc:
            _collect_tool_command_metadata(
                allowed_names={"echo"},
                descriptions=[],
                timeouts=["nope=5"],
            )
        assert "nope" in str(exc.value)

    def test_timeout_non_numeric_raises(self):
        with pytest.raises(SystemExit):
            _collect_tool_command_metadata(
                allowed_names={"echo"},
                descriptions=[],
                timeouts=["echo=abc"],
            )

    def test_timeout_zero_or_negative_raises(self):
        with pytest.raises(SystemExit):
            _collect_tool_command_metadata(
                allowed_names={"echo"},
                descriptions=[],
                timeouts=["echo=0"],
            )
        with pytest.raises(SystemExit):
            _collect_tool_command_metadata(
                allowed_names={"echo"},
                descriptions=[],
                timeouts=["echo=-1"],
            )


class TestBuildCommandTools:
    def test_tool_has_name_and_default_description(self):
        tools = build_command_tools(["echo=echo hello"])
        assert len(tools) == 1
        assert tools[0].name == "echo"
        assert "echo hello" in tools[0].description

    def test_custom_description_used(self):
        tools = build_command_tools(
            ["echo=echo hi"], descriptions=["echo=Say hi out loud."]
        )
        assert "Say hi out loud." in tools[0].description

    @pytest.mark.asyncio
    async def test_echo_runs_and_captures_stdout(self):
        tools = build_command_tools(["echo=echo hello"])
        result = await tools[0].ainvoke({"args": ""})
        assert "hello" in result
        assert "exit=0" in result

    @pytest.mark.asyncio
    async def test_extra_args_are_shlex_split_and_appended(self):
        # 'printf %s foo bar' should print 'foo' only — proving the agent's
        # args are tokenised, not interpreted as a shell string.
        tools = build_command_tools(["p=printf %s"])
        result = await tools[0].ainvoke({"args": "hello"})
        assert "hello" in result
        assert "exit=0" in result

    @pytest.mark.asyncio
    async def test_missing_binary_returns_error_string(self):
        tools = build_command_tools(["bad=nonexistent-binary-xyzzy"])
        result = await tools[0].ainvoke({"args": ""})
        assert "command not found" in result
        assert "nonexistent-binary-xyzzy" in result

    @pytest.mark.asyncio
    async def test_timeout_kills_slow_command(self):
        tools = build_command_tools(
            ["slow=sleep 5"], timeouts=["slow=0.2"]
        )
        result = await tools[0].ainvoke({"args": ""})
        assert "timed out" in result

    @pytest.mark.asyncio
    async def test_shell_metacharacters_are_not_interpreted(self):
        # 'echo' receives '|' and 'wc' as literal args — no pipe.
        tools = build_command_tools(["e=echo"])
        result = await tools[0].ainvoke({"args": "hello | wc -l"})
        assert "hello | wc -l" in result
        assert "exit=0" in result

    @pytest.mark.asyncio
    async def test_malformed_args_returns_error_not_exception(self):
        # Unbalanced quote — shlex.split raises ValueError; we surface it.
        tools = build_command_tools(["e=echo"])
        result = await tools[0].ainvoke({"args": "'unclosed"})
        assert result.startswith("Error:")

    def test_default_timeout_constant(self):
        assert DEFAULT_TIMEOUT_SECONDS == 30.0

    @pytest.mark.asyncio
    async def test_multiple_tools_are_independent(self):
        tools = build_command_tools(["t=true", "f=false"])
        assert {t.name for t in tools} == {"t", "f"}
        # `true` exits 0, `false` exits 1.
        results = await asyncio.gather(
            tools[0].ainvoke({"args": ""}),
            tools[1].ainvoke({"args": ""}),
        )
        codes = sorted(["exit=0" in r for r in results])
        assert codes == [False, True]
