"""Generic parallel-agents example.

This is the template for "spin up N agents in parallel, all sharing one
doubleword.ai batch window, each solving a different prompt." Copy the file,
swap in your own system prompt and MCP server(s), and you're running.

Setup:

    export DOUBLEWORD_API_KEY="your-key"
    # Optional: point at one or more MCP servers. Without this the example
    # runs with the built-in example_tools (web_search, calculator) so you can
    # smoke-test the pipeline without any external dependency.
    export DWAGENTS_MCP_URL="https://your.mcp.server/mcp"
    # Optional: bearer token for the above server, if it requires auth.
    export DWAGENTS_MCP_BEARER="your-token"

Run:

    python examples/parallel_agents.py --prompts-dir examples/prompts

Each prompt file in ``--prompts-dir`` (``.txt`` or ``.md``) becomes one agent;
the file stem is its name. All agents share the same system prompt, tool set,
and batch client, so their LLM calls collate into one batch window for the
50-75% doubleword.ai batch discount.

The CLI flags for model/batch settings override matching DOUBLEWORD_* env
vars. Precedence: CLI flag > env var > built-in default.
"""

import argparse
import asyncio
import os
from typing import Any

from deepagents.backends.filesystem import FilesystemBackend

from dwagents import (
    ToolCallLogger,
    connect_mcp,
    load_prompts_from_dir,
    print_message_trail,
    run_agents_parallel,
    wrap_with_retry,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant. Read files with read_file, write files with "
    "write_file, and use any other tools available to complete the user's "
    "task. When you're done, briefly summarise what you did."
)

# CLI-flag attr name → ChatDoublewordBatch kwarg name. Kept in sync with the
# dwagents CLI so users can swap between the two entry points freely.
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
    """Collect only the flags the user actually supplied, so unset flags don't
    shadow env-var defaults."""
    return {
        dest: getattr(args, attr)
        for attr, dest in _MODEL_KWARG_MAPPING
        if getattr(args, attr, None) is not None
    }


async def _build_tools():
    mcp_url = os.environ.get("DWAGENTS_MCP_URL")
    if mcp_url:
        server_cfg: dict[str, Any] = {"transport": "streamable_http", "url": mcp_url}
        bearer = os.environ.get("DWAGENTS_MCP_BEARER")
        if bearer:
            server_cfg["headers"] = {"Authorization": f"Bearer {bearer}"}
        raw = await connect_mcp({"server": server_cfg})
        print(f"Loaded {len(raw)} MCP tools from {mcp_url}: {[t.name for t in raw]}")
        return [wrap_with_retry(t) for t in raw]
    # No MCP configured — fall back to the bundled example tools.
    from dwagents.tools.example_tools import calculator, web_search
    print("No DWAGENTS_MCP_URL set; using bundled example tools.")
    return [web_search, calculator]


async def main(prompts_dir: str, model_kwargs: dict[str, Any] | None = None) -> None:
    prompts = load_prompts_from_dir(prompts_dir)
    print(f"Loaded {len(prompts)} prompt(s): {list(prompts)}")

    tools = await _build_tools()

    results = await run_agents_parallel(
        prompts,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        model_kwargs=model_kwargs or None,
        backend=FilesystemBackend(),
        callbacks_factory=lambda name: [ToolCallLogger(name)],
    )

    for name, result in results.items():
        print(f"\n=== Agent: {name} ===")
        if isinstance(result, Exception):
            print(f"FAILED: {type(result).__name__}: {result}")
            continue
        print_message_trail(name, result["messages"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel-agents example")
    parser.add_argument(
        "--prompts-dir",
        required=True,
        help="Directory containing one .txt or .md prompt file per agent.",
    )

    model_group = parser.add_argument_group(
        "model settings",
        "Override DOUBLEWORD_* env vars. CLI flag > env var > built-in default.",
    )
    model_group.add_argument("--api-key", default=None,
        help="Prefer DOUBLEWORD_API_KEY env var — CLI flags can leak via shell history.")
    model_group.add_argument("--model", default=None,
        help="Model name (overrides DOUBLEWORD_MODEL).")
    model_group.add_argument("--base-url", default=None,
        help="API base URL (overrides DOUBLEWORD_BASE_URL).")
    model_group.add_argument("--batch-window-seconds", type=float, default=None,
        help="Seconds to accumulate requests before flushing (overrides DOUBLEWORD_BATCH_WINDOW_SECONDS).")
    model_group.add_argument("--batch-size", type=int, default=None,
        help="Max requests per batch (overrides DOUBLEWORD_BATCH_SIZE).")
    model_group.add_argument("--poll-interval-seconds", type=float, default=None,
        help="Seconds between batch-status polls (overrides DOUBLEWORD_POLL_INTERVAL_SECONDS).")
    model_group.add_argument("--completion-window", default=None,
        help="Batch completion window, e.g. '1h' or '24h' (overrides DOUBLEWORD_COMPLETION_WINDOW).")

    args = parser.parse_args()
    asyncio.run(
        main(
            prompts_dir=args.prompts_dir,
            model_kwargs=_build_model_kwargs(args),
        )
    )
