"""Expose local command-line programs as agent tools.

Each declared tool runs via :func:`asyncio.create_subprocess_exec` with
``shell=False``. Both the declared command template and the agent-supplied
``args`` string are tokenised with :func:`shlex.split` — shell metacharacters
(``|``, ``>``, ``*`` …) are **not** interpreted. Users who need shell
behaviour declare it explicitly, e.g.
``--tool-command pipeline="sh -c 'git log | head'"``.
"""

from __future__ import annotations

import asyncio
import shlex

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

DEFAULT_TIMEOUT_SECONDS = 30.0
OUTPUT_BYTE_LIMIT = 8 * 1024


class _CommandToolInput(BaseModel):
    """Schema for a local-command tool's single input."""

    args: str = Field(
        default="",
        description=(
            "Extra arguments appended to the declared command, tokenised "
            "with shlex.split. Shell metacharacters are NOT interpreted. "
            "Pass an empty string to run the command as declared."
        ),
    )


def _parse_tool_command_specs(specs: list[str]) -> dict[str, str]:
    """Parse ``--tool-command NAME=COMMAND`` specs.

    Raises ``SystemExit`` on malformed input or names that aren't valid
    Python identifiers (required because LangChain's ``@tool`` decorator
    binds the tool name to the function's ``__name__``).
    """
    parsed: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(
                f"--tool-command expects 'name=command', got: {spec!r}"
            )
        name, command = spec.split("=", 1)
        name, command = name.strip(), command.strip()
        if not name or not command:
            raise SystemExit(
                f"--tool-command expects non-empty name and command, got: {spec!r}"
            )
        if not name.isidentifier():
            raise SystemExit(
                f"--tool-command name must be a valid Python identifier, got: {name!r}"
            )
        if name in parsed:
            raise SystemExit(
                f"--tool-command name {name!r} declared more than once."
            )
        parsed[name] = command
    return parsed


def _collect_tool_command_metadata(
    *,
    allowed_names: set[str],
    descriptions: list[str],
    timeouts: list[str],
) -> tuple[dict[str, str], dict[str, float]]:
    """Parse ``--tool-command-description NAME=TEXT`` and
    ``--tool-command-timeout NAME=SECONDS`` into per-name maps.

    Raises ``SystemExit`` if a flag names a tool not declared via
    ``--tool-command`` — catches typos early instead of silently dropping
    config.
    """
    desc_map: dict[str, str] = {}
    for spec in descriptions:
        if "=" not in spec:
            raise SystemExit(
                f"--tool-command-description expects 'name=text', got: {spec!r}"
            )
        name, text = spec.split("=", 1)
        name = name.strip()
        if not name or not text:
            raise SystemExit(
                f"--tool-command-description expects non-empty name and text, got: {spec!r}"
            )
        if name not in allowed_names:
            raise SystemExit(
                f"--tool-command-description references tool {name!r} but no "
                f"--tool-command with that name was declared."
            )
        desc_map[name] = text

    timeout_map: dict[str, float] = {}
    for spec in timeouts:
        if "=" not in spec:
            raise SystemExit(
                f"--tool-command-timeout expects 'name=seconds', got: {spec!r}"
            )
        name, raw = spec.split("=", 1)
        name, raw = name.strip(), raw.strip()
        if not name or not raw:
            raise SystemExit(
                f"--tool-command-timeout expects non-empty name and seconds, got: {spec!r}"
            )
        if name not in allowed_names:
            raise SystemExit(
                f"--tool-command-timeout references tool {name!r} but no "
                f"--tool-command with that name was declared."
            )
        try:
            seconds = float(raw)
        except ValueError:
            raise SystemExit(
                f"--tool-command-timeout seconds must be a number, got: {raw!r}"
            ) from None
        if seconds <= 0:
            raise SystemExit(
                f"--tool-command-timeout seconds must be > 0, got: {seconds}"
            )
        timeout_map[name] = seconds

    return desc_map, timeout_map


def _truncate(stream: bytes, label: str) -> str:
    if len(stream) <= OUTPUT_BYTE_LIMIT:
        return stream.decode("utf-8", errors="replace")
    head = stream[:OUTPUT_BYTE_LIMIT].decode("utf-8", errors="replace")
    return f"{head}\n[...{label} truncated, {len(stream) - OUTPUT_BYTE_LIMIT} bytes dropped]"


def _format_result(argv: list[str], returncode: int, stdout: bytes, stderr: bytes) -> str:
    return (
        f"$ {shlex.join(argv)}\n"
        f"exit={returncode}\n"
        f"stdout:\n{_truncate(stdout, 'stdout')}\n"
        f"stderr:\n{_truncate(stderr, 'stderr')}"
    )


def _make_tool(name: str, command: str, description: str, timeout: float) -> BaseTool:
    base_argv = shlex.split(command)
    if not base_argv:
        raise SystemExit(
            f"--tool-command {name!r} parsed to an empty argv; "
            f"check quoting of {command!r}."
        )

    async def _invoke(args: str = "") -> str:
        try:
            extra = shlex.split(args) if args else []
        except ValueError as e:
            return f"Error: could not parse args {args!r}: {e}"
        argv = base_argv + extra
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return f"Error: command not found: {argv[0]}"
        except Exception as e:  # noqa: BLE001
            return f"Error: {type(e).__name__}: {e}"
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            try:
                await proc.wait()
            except Exception:  # noqa: BLE001
                pass
            return f"Error: timed out after {timeout}s: $ {shlex.join(argv)}"
        return _format_result(argv, proc.returncode or 0, stdout, stderr)

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=name,
        description=description,
        args_schema=_CommandToolInput,
    )


def build_command_tools(
    specs: list[str],
    *,
    descriptions: list[str] | None = None,
    timeouts: list[str] | None = None,
) -> list[BaseTool]:
    """Turn ``--tool-command*`` CLI specs into a list of LangChain tools."""
    commands = _parse_tool_command_specs(specs)
    desc_map, timeout_map = _collect_tool_command_metadata(
        allowed_names=set(commands),
        descriptions=descriptions or [],
        timeouts=timeouts or [],
    )
    return [
        _make_tool(
            name=name,
            command=command,
            description=desc_map.get(
                name,
                f"Run the local command `{command}`. Pass extra arguments via "
                f"the `args` string (shlex-split, appended to the command).",
            ),
            timeout=timeout_map.get(name, DEFAULT_TIMEOUT_SECONDS),
        )
        for name, command in commands.items()
    ]
