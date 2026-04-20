"""Console-script entry point for dwagents.

Exposes ``dwagents run`` as a ready-made wrapper around
:func:`run_agents_parallel`, :func:`connect_mcp`, :func:`wrap_with_retry`, and
:class:`ToolCallLogger`. It's the "batteries included" path — hand it a
directory of prompt files and optionally one or more MCP server URLs, and it
runs them in parallel through a single batch window with per-agent logging
and a post-run message trail.

For anything beyond the shape of this CLI, copy ``examples/parallel_agents.py``
and edit in your own wiring.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool

from dwagents.mcp import connect_mcp, wrap_with_retry
from dwagents.observability import ToolCallLogger, print_message_trail
from dwagents.parallel import load_prompts_from_dir, run_agents_parallel

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the tools available to you to complete "
    "the user's task. When you're done, summarise what you did."
)


def _parse_mcp_server_specs(
    specs: list[str],
    *,
    headers_by_name: dict[str, dict[str, str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Parse ``--mcp-server name=url`` specs into MultiServerMCPClient config.

    If ``headers_by_name`` is supplied, each matching server's config also
    carries a ``headers`` dict (for bearer tokens or custom auth headers).
    """
    servers: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(
                f"--mcp-server expects 'name=url', got: {spec!r}"
            )
        name, url = spec.split("=", 1)
        name, url = name.strip(), url.strip()
        if not name or not url:
            raise SystemExit(
                f"--mcp-server expects non-empty name and url, got: {spec!r}"
            )
        config: dict[str, Any] = {"transport": "streamable_http", "url": url}
        if headers_by_name and name in headers_by_name and headers_by_name[name]:
            config["headers"] = dict(headers_by_name[name])
        servers[name] = config
    return servers


def _collect_mcp_headers(
    *,
    allowed_names: set[str],
    bearer_tokens: list[str],
    headers: list[str],
    env: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    """Merge env bearer tokens + ``--mcp-bearer-token`` + ``--mcp-header`` into
    a per-server headers dict.

    Precedence (low → high): ``DWAGENTS_MCP_BEARER_<NAME>`` env var,
    ``--mcp-bearer-token NAME=TOKEN``, ``--mcp-header NAME=KEY:VALUE``. The
    most-explicit source wins when sources collide on the same header key.

    Raises ``SystemExit`` if any auth flag names a server not in
    ``allowed_names`` — catches typos early instead of silently dropping auth.
    """
    result: dict[str, dict[str, str]] = {name: {} for name in allowed_names}

    # 1. Env bearer tokens (lowest precedence).
    for name in allowed_names:
        env_key = "DWAGENTS_MCP_BEARER_" + re.sub(r"\W", "_", name).upper()
        token = env.get(env_key)
        if token:
            result[name]["Authorization"] = f"Bearer {token}"

    # 2. --mcp-bearer-token NAME=TOKEN (overrides env).
    for spec in bearer_tokens:
        if "=" not in spec:
            raise SystemExit(
                f"--mcp-bearer-token expects 'name=token', got: {spec!r}"
            )
        name, token = spec.split("=", 1)
        name, token = name.strip(), token.strip()
        if not name or not token:
            raise SystemExit(
                f"--mcp-bearer-token expects non-empty name and token, got: {spec!r}"
            )
        if name not in result:
            raise SystemExit(
                f"--mcp-bearer-token references server {name!r} but no "
                f"--mcp-server with that name was declared."
            )
        result[name]["Authorization"] = f"Bearer {token}"

    # 3. --mcp-header NAME=KEY:VALUE (most explicit; overrides bearer if keys collide).
    for spec in headers:
        if "=" not in spec:
            raise SystemExit(
                f"--mcp-header expects 'name=key:value', got: {spec!r}"
            )
        name, keyval = spec.split("=", 1)
        name = name.strip()
        if ":" not in keyval:
            raise SystemExit(
                f"--mcp-header expects 'name=key:value', got: {spec!r}"
            )
        header_key, header_val = keyval.split(":", 1)
        header_key, header_val = header_key.strip(), header_val.strip()
        if not name or not header_key:
            raise SystemExit(
                f"--mcp-header expects non-empty name and header key, got: {spec!r}"
            )
        if name not in result:
            raise SystemExit(
                f"--mcp-header references server {name!r} but no "
                f"--mcp-server with that name was declared."
            )
        result[name][header_key] = header_val

    # Drop empty dicts so server specs without auth don't carry an empty headers key.
    return {name: h for name, h in result.items() if h}


# CLI-flag name → ChatDoublewordBatch kwarg name. Listed in priority order
# (precedence is handled by argparse + ChatDoublewordBatch itself). Kept as a
# module constant so test_cli can exercise the mapping directly.
_MODEL_KWARG_MAPPING: list[tuple[str, str]] = [
    ("api_key", "api_key"),
    ("model", "model_name"),
    ("base_url", "base_url"),
    ("batch_window_seconds", "batch_window_seconds"),
    ("batch_size", "batch_size"),
    ("poll_interval_seconds", "poll_interval_seconds"),
    ("completion_window", "completion_window"),
]


def _build_model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Return only the keys whose flag was actually supplied.

    Unset flags stay out of the dict so they don't shadow env-var defaults that
    ``ChatDoublewordBatch.__init__`` applies via ``kwargs.setdefault(...)``.
    """
    return {
        dest: getattr(args, attr)
        for attr, dest in _MODEL_KWARG_MAPPING
        if getattr(args, attr, None) is not None
    }


def _resolve_system_prompt(args: argparse.Namespace) -> str:
    if args.system_prompt is not None:
        return args.system_prompt
    if args.system_prompt_file is not None:
        return Path(args.system_prompt_file).read_text(encoding="utf-8")
    return DEFAULT_SYSTEM_PROMPT


async def _resolve_tools(args: argparse.Namespace) -> list[BaseTool]:
    if args.mcp_server:
        server_names = {
            spec.split("=", 1)[0].strip()
            for spec in args.mcp_server
            if "=" in spec
        }
        headers_by_name = _collect_mcp_headers(
            allowed_names=server_names,
            bearer_tokens=args.mcp_bearer_token,
            headers=args.mcp_header,
            env=os.environ,
        )
        servers = _parse_mcp_server_specs(
            args.mcp_server, headers_by_name=headers_by_name
        )
        raw = await connect_mcp(servers)
        return [wrap_with_retry(t) for t in raw]
    # No MCP configured — fall back to the built-in example tools so the CLI
    # still does *something* useful out of the box.
    from dwagents.tools.example_tools import calculator, web_search
    return [web_search, calculator]


async def _run(args: argparse.Namespace) -> int:
    prompts = load_prompts_from_dir(args.prompts_dir)
    print(f"Loaded {len(prompts)} prompt(s): {list(prompts)}", flush=True)

    tools = await _resolve_tools(args)
    print(f"Loaded {len(tools)} tool(s): {[t.name for t in tools]}", flush=True)

    system_prompt = _resolve_system_prompt(args)
    model_kwargs = _build_model_kwargs(args)

    backend = None
    if not args.no_filesystem_backend:
        from deepagents.backends.filesystem import FilesystemBackend
        backend = FilesystemBackend()

    results = await run_agents_parallel(
        prompts,
        tools=tools,
        system_prompt=system_prompt,
        model_kwargs=model_kwargs or None,
        backend=backend,
        callbacks_factory=lambda name: [ToolCallLogger(name)],
    )

    exit_code = 0
    for name, result in results.items():
        print(f"\n=== Agent: {name} ===")
        if isinstance(result, Exception):
            print(f"FAILED: {type(result).__name__}: {result}")
            exit_code = 1
            continue
        print_message_trail(name, result["messages"])

    return exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dwagents",
        description=(
            "Run parallel batched LangChain deep agents on doubleword.ai "
            "inference, with optional MCP tools."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="Run one agent per prompt file in a directory, in parallel.",
    )
    run.add_argument(
        "--prompts-dir",
        required=True,
        help="Directory containing one .txt or .md prompt file per agent.",
    )
    prompt_group = run.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--system-prompt",
        default=None,
        help="System prompt text shared by every agent.",
    )
    prompt_group.add_argument(
        "--system-prompt-file",
        default=None,
        help="Path to a file whose contents become the shared system prompt.",
    )
    run.add_argument(
        "--mcp-server",
        action="append",
        default=[],
        metavar="NAME=URL",
        help=(
            "MCP server to connect to, as NAME=URL. Repeat for multiple "
            "servers. Uses streamable_http transport. If omitted, the CLI "
            "falls back to the built-in example tools."
        ),
    )
    run.add_argument(
        "--mcp-bearer-token",
        action="append",
        default=[],
        metavar="NAME=TOKEN",
        help=(
            "Bearer token for a named --mcp-server (adds Authorization: Bearer "
            "<token>). Repeatable. Prefer the DWAGENTS_MCP_BEARER_<NAME> env "
            "var — CLI flags can leak into shell history and `ps` output."
        ),
    )
    run.add_argument(
        "--mcp-header",
        action="append",
        default=[],
        metavar="NAME=KEY:VALUE",
        help=(
            "Arbitrary HTTP header for a named --mcp-server, as "
            "NAME=KEY:VALUE. Repeatable. Overrides --mcp-bearer-token if it "
            "also sets Authorization."
        ),
    )
    run.add_argument(
        "--no-filesystem-backend",
        action="store_true",
        help=(
            "Disable the deepagents FilesystemBackend (read_file/write_file "
            "will hit an in-memory virtual filesystem instead of real disk)."
        ),
    )

    # Model / batch settings. Each flag overrides the matching DOUBLEWORD_*
    # env var; anything left unset falls through to the env var or built-in
    # default via ChatDoublewordBatch.__init__.
    model_group = run.add_argument_group(
        "model settings",
        "Override DOUBLEWORD_* env vars. CLI flag > env var > built-in default.",
    )
    model_group.add_argument(
        "--api-key",
        default=None,
        help=(
            "doubleword.ai API key. Prefer the DOUBLEWORD_API_KEY env var — "
            "CLI flags can leak into shell history and `ps` output."
        ),
    )
    model_group.add_argument(
        "--model",
        default=None,
        help="Model name (overrides DOUBLEWORD_MODEL).",
    )
    model_group.add_argument(
        "--base-url",
        default=None,
        help="API base URL (overrides DOUBLEWORD_BASE_URL).",
    )
    model_group.add_argument(
        "--batch-window-seconds",
        type=float,
        default=None,
        help="Seconds to accumulate requests before flushing a batch (overrides DOUBLEWORD_BATCH_WINDOW_SECONDS).",
    )
    model_group.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Max requests per batch (overrides DOUBLEWORD_BATCH_SIZE).",
    )
    model_group.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=None,
        help="Seconds between batch-status polls (overrides DOUBLEWORD_POLL_INTERVAL_SECONDS).",
    )
    model_group.add_argument(
        "--completion-window",
        default=None,
        help="Batch completion window, e.g. '1h' or '24h' (overrides DOUBLEWORD_COMPLETION_WINDOW).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return asyncio.run(_run(args))
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
