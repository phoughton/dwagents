"""Run multiple supervisor agents in parallel over a single shared batch client."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from deepagents import SubAgent
from deepagents.backends.protocol import BackendProtocol
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import BaseTool

from dwagents.agent import create_supervisor
from dwagents.models.batch import ChatDoublewordBatch


def load_prompts_from_dir(
    prompts_dir: str | Path,
    suffixes: tuple[str, ...] = (".txt", ".md"),
) -> dict[str, str]:
    """Load one prompt per file from a directory.

    Args:
        prompts_dir: Directory containing prompt files.
        suffixes: File suffixes to include. Other files are ignored.

    Returns:
        Mapping of agent name (file stem) to prompt text.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If the directory contains no files with the given suffixes.
    """
    path = Path(prompts_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Prompts directory not found: {path}")

    prompts: dict[str, str] = {}
    for entry in sorted(path.iterdir()):
        if entry.is_file() and entry.suffix.lower() in suffixes:
            prompts[entry.stem] = entry.read_text(encoding="utf-8").strip()

    if not prompts:
        raise ValueError(
            f"No prompt files ({', '.join(suffixes)}) found in {path}"
        )
    return prompts


async def run_agents_parallel(
    prompts: dict[str, str],
    *,
    tools: list[BaseTool | callable | dict[str, Any]] | None = None,
    system_prompt: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    subagents: list[SubAgent] | None = None,
    backend: BackendProtocol | None = None,
    callbacks_factory: Callable[[str], list[BaseCallbackHandler]] | None = None,
) -> dict[str, Any]:
    """Run one supervisor per prompt concurrently, sharing one BatchOpenAI client.

    Each agent gets its own message history, but all LLM calls flow through a
    single shared ``BatchOpenAI`` so concurrent requests collate into the same
    batch window (``batch_window_seconds``).

    Args:
        prompts: Mapping of agent name to initial user prompt.
        tools: Tools passed to every agent.
        system_prompt: System prompt shared by every agent.
        model_kwargs: kwargs for the shared ``ChatDoublewordBatch``.
        subagents: SubAgent definitions shared by every agent.
        callbacks_factory: Optional factory called once per agent with the
            agent's name; returns a list of LangChain callback handlers passed
            via ``config={"callbacks": ...}`` to that agent's invocation. Use
            for per-agent logging/tracing.

    Returns:
        Mapping of agent name to the invocation result, or the Exception raised.
    """
    if not prompts:
        return {}

    shared_model = ChatDoublewordBatch(**(model_kwargs or {}))
    shared_model._get_client()  # prime so model_copy() in bind_tools shares the client

    async def _one(name: str, prompt: str) -> Any:
        agent = create_supervisor(
            tools=tools,
            system_prompt=system_prompt,
            model=shared_model,
            subagents=subagents,
            backend=backend,
        )
        config: dict[str, Any] = {}
        if callbacks_factory is not None:
            config["callbacks"] = callbacks_factory(name)
        return await agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config=config or None,
        )

    names = list(prompts.keys())
    try:
        results = await asyncio.gather(
            *(_one(n, prompts[n]) for n in names),
            return_exceptions=True,
        )
    finally:
        await shared_model.aclose()
    return dict(zip(names, results))
