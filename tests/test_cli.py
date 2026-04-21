"""Tests for dwagents.cli — argparse routing and CLI dispatch."""

from __future__ import annotations

import argparse
import io
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dwagents.cli import (
    DEFAULT_SYSTEM_PROMPT,
    _build_model_kwargs,
    _build_parser,
    _collect_mcp_headers,
    _parse_mcp_server_specs,
    _resolve_system_prompt,
    main,
)


class TestParseMcpServerSpecs:
    def test_parses_single_server(self):
        result = _parse_mcp_server_specs(["files=https://example/mcp"])
        assert result == {
            "files": {"transport": "streamable_http", "url": "https://example/mcp"}
        }

    def test_parses_multiple_servers(self):
        result = _parse_mcp_server_specs(
            ["a=http://x/", "b=http://y/"]
        )
        assert set(result) == {"a", "b"}
        assert result["a"]["url"] == "http://x/"
        assert result["b"]["url"] == "http://y/"

    def test_rejects_missing_equals(self):
        with pytest.raises(SystemExit):
            _parse_mcp_server_specs(["nope"])

    def test_rejects_empty_name(self):
        with pytest.raises(SystemExit):
            _parse_mcp_server_specs(["=http://x/"])


class TestParseMcpServerSpecsWithHeaders:
    def test_merges_headers_into_server_config(self):
        result = _parse_mcp_server_specs(
            ["files=http://x/"],
            headers_by_name={"files": {"Authorization": "Bearer xxx", "X-Extra": "y"}},
        )
        assert result == {
            "files": {
                "transport": "streamable_http",
                "url": "http://x/",
                "headers": {"Authorization": "Bearer xxx", "X-Extra": "y"},
            }
        }

    def test_no_headers_key_when_none_supplied(self):
        result = _parse_mcp_server_specs(["files=http://x/"])
        assert "headers" not in result["files"]

    def test_empty_headers_dict_also_omitted(self):
        result = _parse_mcp_server_specs(
            ["files=http://x/"], headers_by_name={"files": {}}
        )
        assert "headers" not in result["files"]


class TestCollectMcpHeaders:
    def test_env_bearer_only(self):
        result = _collect_mcp_headers(
            allowed_names={"files"},
            bearer_tokens=[],
            headers=[],
            env={"DWAGENTS_MCP_BEARER_FILES": "tok"},
        )
        assert result == {"files": {"Authorization": "Bearer tok"}}

    def test_env_bearer_name_non_alphanumeric_chars_translated(self):
        result = _collect_mcp_headers(
            allowed_names={"files-server"},
            bearer_tokens=[],
            headers=[],
            env={"DWAGENTS_MCP_BEARER_FILES_SERVER": "tok"},
        )
        assert result == {"files-server": {"Authorization": "Bearer tok"}}

    def test_cli_bearer_overrides_env_bearer(self):
        result = _collect_mcp_headers(
            allowed_names={"files"},
            bearer_tokens=["files=cli-tok"],
            headers=[],
            env={"DWAGENTS_MCP_BEARER_FILES": "env-tok"},
        )
        assert result == {"files": {"Authorization": "Bearer cli-tok"}}

    def test_cli_header_overrides_cli_bearer_for_same_key(self):
        result = _collect_mcp_headers(
            allowed_names={"files"},
            bearer_tokens=["files=bearer-tok"],
            headers=["files=Authorization:Custom xyz"],
            env={},
        )
        assert result == {"files": {"Authorization": "Custom xyz"}}

    def test_custom_header(self):
        result = _collect_mcp_headers(
            allowed_names={"crm"},
            bearer_tokens=[],
            headers=["crm=X-API-Key:abc"],
            env={},
        )
        assert result == {"crm": {"X-API-Key": "abc"}}

    def test_header_value_may_contain_colons(self):
        result = _collect_mcp_headers(
            allowed_names={"s"},
            bearer_tokens=[],
            headers=["s=X-Ref:Bearer abc:def"],
            env={},
        )
        assert result == {"s": {"X-Ref": "Bearer abc:def"}}

    def test_no_empty_dict_for_servers_without_auth(self):
        result = _collect_mcp_headers(
            allowed_names={"a", "b"},
            bearer_tokens=["a=tok"],
            headers=[],
            env={},
        )
        assert "b" not in result
        assert result["a"] == {"Authorization": "Bearer tok"}

    def test_bearer_for_unknown_server_raises(self):
        with pytest.raises(SystemExit) as exc:
            _collect_mcp_headers(
                allowed_names={"files"},
                bearer_tokens=["nope=tok"],
                headers=[],
                env={},
            )
        assert "nope" in str(exc.value)

    def test_header_for_unknown_server_raises(self):
        with pytest.raises(SystemExit) as exc:
            _collect_mcp_headers(
                allowed_names={"files"},
                bearer_tokens=[],
                headers=["nope=X:y"],
                env={},
            )
        assert "nope" in str(exc.value)

    def test_bearer_bad_format_raises(self):
        with pytest.raises(SystemExit):
            _collect_mcp_headers(
                allowed_names={"files"},
                bearer_tokens=["files"],
                headers=[],
                env={},
            )

    def test_header_missing_colon_raises(self):
        with pytest.raises(SystemExit):
            _collect_mcp_headers(
                allowed_names={"files"},
                bearer_tokens=[],
                headers=["files=noseparator"],
                env={},
            )


class TestResolveSystemPrompt:
    def test_defaults_when_nothing_passed(self):
        args = MagicMock(system_prompt=None, system_prompt_file=None)
        assert _resolve_system_prompt(args) == DEFAULT_SYSTEM_PROMPT

    def test_inline_text_wins(self):
        args = MagicMock(system_prompt="hello", system_prompt_file=None)
        assert _resolve_system_prompt(args) == "hello"

    def test_reads_from_file(self, tmp_path):
        f = tmp_path / "prompt.txt"
        f.write_text("from file")
        args = MagicMock(system_prompt=None, system_prompt_file=str(f))
        assert _resolve_system_prompt(args) == "from file"


class TestBuildModelKwargs:
    """CLI flags for DOUBLEWORD_* settings get mapped to ChatDoublewordBatch
    kwargs. Unset flags are excluded so env-var defaults still apply."""

    @staticmethod
    def _parse(argv: list[str]):
        # Always include the required --prompts-dir sentinel so argparse
        # doesn't reject the command.
        return _build_parser().parse_args(["run", "--prompts-dir", "/tmp"] + argv)

    def test_empty_when_no_flags_passed(self):
        args = self._parse([])
        assert _build_model_kwargs(args) == {}

    def test_all_flags_mapped_with_correct_types(self):
        args = self._parse([
            "--api-key", "sk-123",
            "--model", "my-model",
            "--base-url", "http://x/",
            "--batch-window-seconds", "2.5",
            "--batch-size", "42",
            "--poll-interval-seconds", "3.5",
            "--completion-window", "1h",
        ])
        kwargs = _build_model_kwargs(args)
        assert kwargs == {
            "api_key": "sk-123",
            "model_name": "my-model",
            "base_url": "http://x/",
            "batch_window_seconds": 2.5,
            "batch_size": 42,
            "poll_interval_seconds": 3.5,
            "completion_window": "1h",
        }
        # argparse type coercion, not strings.
        assert isinstance(kwargs["batch_window_seconds"], float)
        assert isinstance(kwargs["batch_size"], int)
        assert isinstance(kwargs["poll_interval_seconds"], float)

    def test_model_flag_renamed_to_model_name(self):
        # --model maps to model_name to match ChatDoublewordBatch's kwarg.
        args = self._parse(["--model", "foo"])
        assert _build_model_kwargs(args) == {"model_name": "foo"}

    def test_only_supplied_flags_present(self):
        args = self._parse(["--batch-size", "5", "--completion-window", "24h"])
        assert _build_model_kwargs(args) == {"batch_size": 5, "completion_window": "24h"}


class TestMainRoutesToRun:
    def test_run_without_mcp_uses_example_tools_and_invokes_runner(self, tmp_path):
        """Smoke test: argparse, prompt loading, tool fallback, and the
        runner all wire together. We mock run_agents_parallel so we don't
        actually hit the network."""
        (tmp_path / "a.md").write_text("prompt one")
        (tmp_path / "b.txt").write_text("prompt two")

        fake_results = {"a": {"messages": []}, "b": {"messages": []}}

        async def _fake_runner(prompts, **kwargs):
            # Assert the CLI wired things correctly.
            assert set(prompts) == {"a", "b"}
            assert kwargs["system_prompt"] == DEFAULT_SYSTEM_PROMPT
            assert callable(kwargs["callbacks_factory"])
            tool_names = [t.name for t in kwargs["tools"]]
            assert "web_search" in tool_names and "calculator" in tool_names
            return fake_results

        with patch(
            "dwagents.cli.run_agents_parallel", new=AsyncMock(side_effect=_fake_runner)
        ), patch("dwagents.cli.FilesystemBackend", create=True):
            # FilesystemBackend is imported inside _run; patch the module path
            # where it's actually resolved.
            with patch(
                "deepagents.backends.filesystem.FilesystemBackend"
            ) as MockBackend:
                MockBackend.return_value = MagicMock()
                rc = main(["run", "--prompts-dir", str(tmp_path)])

        assert rc == 0

    def test_run_with_mcp_calls_connect_mcp(self, tmp_path):
        (tmp_path / "a.md").write_text("prompt one")

        fake_tool = MagicMock()
        fake_tool.name = "remote_tool"
        # wrap_with_retry in the CLI path is the real one; bypass it so we
        # can use a plain MagicMock tool without a real pydantic args_schema.
        wrapped = MagicMock()
        wrapped.name = "remote_tool"

        async def _fake_connect(servers, **kwargs):
            assert servers == {
                "s": {"transport": "streamable_http", "url": "http://x/"}
            }
            return [fake_tool]

        async def _fake_runner(prompts, **kwargs):
            tool_names = [t.name for t in kwargs["tools"]]
            assert tool_names == ["remote_tool"]
            return {"a": {"messages": []}}

        with patch(
            "dwagents.cli.connect_mcp", new=AsyncMock(side_effect=_fake_connect)
        ), patch(
            "dwagents.cli.wrap_with_retry", return_value=wrapped
        ) as mock_wrap, patch(
            "dwagents.cli.run_agents_parallel", new=AsyncMock(side_effect=_fake_runner)
        ), patch("deepagents.backends.filesystem.FilesystemBackend") as MockBackend:
            MockBackend.return_value = MagicMock()
            rc = main(
                [
                    "run",
                    "--prompts-dir",
                    str(tmp_path),
                    "--mcp-server",
                    "s=http://x/",
                ]
            )

        assert rc == 0
        mock_wrap.assert_called_once_with(fake_tool)

    def test_bearer_token_flag_reaches_connect_mcp_as_header(self, tmp_path):
        (tmp_path / "a.md").write_text("prompt one")

        captured: dict = {}
        fake_tool = MagicMock()
        fake_tool.name = "remote_tool"
        wrapped = MagicMock()
        wrapped.name = "remote_tool"

        async def _fake_connect(servers, **kwargs):
            captured["servers"] = servers
            return [fake_tool]

        async def _fake_runner(prompts, **kwargs):
            return {"a": {"messages": []}}

        with patch(
            "dwagents.cli.connect_mcp", new=AsyncMock(side_effect=_fake_connect)
        ), patch(
            "dwagents.cli.wrap_with_retry", return_value=wrapped
        ), patch(
            "dwagents.cli.run_agents_parallel", new=AsyncMock(side_effect=_fake_runner)
        ), patch("deepagents.backends.filesystem.FilesystemBackend") as MockBackend:
            MockBackend.return_value = MagicMock()
            rc = main([
                "run",
                "--prompts-dir", str(tmp_path),
                "--mcp-server", "files=http://x/",
                "--mcp-bearer-token", "files=secret",
            ])

        assert rc == 0
        assert captured["servers"] == {
            "files": {
                "transport": "streamable_http",
                "url": "http://x/",
                "headers": {"Authorization": "Bearer secret"},
            }
        }

    def test_cli_flags_reach_runner_as_model_kwargs(self, tmp_path):
        (tmp_path / "a.md").write_text("prompt one")

        captured: dict = {}

        async def _fake_runner(prompts, **kwargs):
            captured["model_kwargs"] = kwargs.get("model_kwargs")
            return {"a": {"messages": []}}

        with patch(
            "dwagents.cli.run_agents_parallel", new=AsyncMock(side_effect=_fake_runner)
        ), patch("deepagents.backends.filesystem.FilesystemBackend") as MockBackend:
            MockBackend.return_value = MagicMock()
            rc = main([
                "run",
                "--prompts-dir", str(tmp_path),
                "--model", "custom-model",
                "--batch-size", "7",
                "--completion-window", "1h",
            ])

        assert rc == 0
        assert captured["model_kwargs"] == {
            "model_name": "custom-model",
            "batch_size": 7,
            "completion_window": "1h",
        }

    def test_tool_command_flag_reaches_runner_as_tool(self, tmp_path):
        (tmp_path / "a.md").write_text("prompt one")

        captured: dict = {}

        async def _fake_runner(prompts, **kwargs):
            captured["tool_names"] = [t.name for t in kwargs["tools"]]
            return {"a": {"messages": []}}

        with patch(
            "dwagents.cli.run_agents_parallel", new=AsyncMock(side_effect=_fake_runner)
        ), patch("deepagents.backends.filesystem.FilesystemBackend") as MockBackend:
            MockBackend.return_value = MagicMock()
            rc = main([
                "run",
                "--prompts-dir", str(tmp_path),
                "--tool-command", "echo=echo hi",
                "--tool-command-description", "echo=Say hi.",
            ])

        assert rc == 0
        assert captured["tool_names"] == ["echo"]

    def test_tool_command_composes_with_mcp(self, tmp_path):
        (tmp_path / "a.md").write_text("prompt one")

        fake_tool = MagicMock()
        fake_tool.name = "remote_tool"
        wrapped = MagicMock()
        wrapped.name = "remote_tool"

        captured: dict = {}

        async def _fake_connect(servers, **kwargs):
            return [fake_tool]

        async def _fake_runner(prompts, **kwargs):
            captured["tool_names"] = [t.name for t in kwargs["tools"]]
            return {"a": {"messages": []}}

        with patch(
            "dwagents.cli.connect_mcp", new=AsyncMock(side_effect=_fake_connect)
        ), patch(
            "dwagents.cli.wrap_with_retry", return_value=wrapped
        ), patch(
            "dwagents.cli.run_agents_parallel", new=AsyncMock(side_effect=_fake_runner)
        ), patch("deepagents.backends.filesystem.FilesystemBackend") as MockBackend:
            MockBackend.return_value = MagicMock()
            rc = main([
                "run",
                "--prompts-dir", str(tmp_path),
                "--mcp-server", "s=http://x/",
                "--tool-command", "echo=echo hi",
            ])

        assert rc == 0
        # MCP tool comes first (MCP branch runs first in _resolve_tools);
        # command tool follows. Both must be present.
        assert captured["tool_names"] == ["remote_tool", "echo"]

    def test_run_returns_nonzero_on_agent_failure(self, tmp_path):
        (tmp_path / "a.md").write_text("prompt one")

        async def _fake_runner(prompts, **kwargs):
            return {"a": RuntimeError("blew up")}

        with patch(
            "dwagents.cli.run_agents_parallel", new=AsyncMock(side_effect=_fake_runner)
        ), patch("deepagents.backends.filesystem.FilesystemBackend") as MockBackend:
            MockBackend.return_value = MagicMock()
            rc = main(["run", "--prompts-dir", str(tmp_path)])

        assert rc == 1


class TestHelpOutput:
    """`--help` and `--version` output. Guards against regressions in the
    top-level description, epilog examples, argument-group layout, and the
    version flag."""

    @staticmethod
    def _run_subparser_help() -> str:
        parser = _build_parser()
        # Reach into subparsers to format the `run` subcommand's help
        # without actually invoking argparse's SystemExit path.
        subparsers_action = next(
            a for a in parser._actions if isinstance(a, argparse._SubParsersAction)  # type: ignore[attr-defined]
        )
        return subparsers_action.choices["run"].format_help()

    def test_top_level_help_lists_run_and_version_and_points_to_examples(self):
        help_text = _build_parser().format_help()
        assert "run" in help_text
        assert "--version" in help_text
        assert "examples/parallel_agents.py" in help_text
        assert "quick start" in help_text

    def test_run_help_has_argument_groups(self):
        help_text = self._run_subparser_help()
        assert "input" in help_text
        assert "tools (MCP)" in help_text
        assert "tools (local commands)" in help_text
        assert "agent runtime" in help_text
        assert "model & batching" in help_text

    def test_run_help_has_local_commands_flags(self):
        help_text = self._run_subparser_help()
        assert "--tool-command NAME=COMMAND" in help_text
        assert "--tool-command-description NAME=TEXT" in help_text
        assert "--tool-command-timeout NAME=SECONDS" in help_text
        # Safety note from the group description.
        assert "shlex.split" in help_text

    def test_run_help_has_worked_examples(self):
        help_text = self._run_subparser_help()
        assert "examples:" in help_text
        # The README-derived worked example with --mcp-server NAME=URL.
        assert "--mcp-server files=" in help_text
        assert "DWAGENTS_MCP_BEARER_FILES" in help_text

    def test_run_help_mentions_builtin_defaults(self):
        help_text = self._run_subparser_help()
        # Default system prompt + fallback tools should be discoverable here.
        assert "built-in assistant prompt" in help_text
        assert "web_search" in help_text and "calculator" in help_text

    def test_run_help_has_metavars(self):
        help_text = self._run_subparser_help()
        # Spot-check a few metavars we added.
        assert "--prompts-dir PATH" in help_text
        assert "--api-key KEY" in help_text
        assert "--completion-window WINDOW" in help_text

    def test_version_flag_prints_version_and_exits_zero(self):
        buf = io.StringIO()
        with pytest.raises(SystemExit) as exc, redirect_stdout(buf), redirect_stderr(buf):
            main(["--version"])
        assert exc.value.code == 0
        assert buf.getvalue().startswith("dwagents ")

    def test_help_flag_exits_zero(self):
        buf = io.StringIO()
        with pytest.raises(SystemExit) as exc, redirect_stdout(buf), redirect_stderr(buf):
            main(["--help"])
        assert exc.value.code == 0
        assert "dwagents" in buf.getvalue()
