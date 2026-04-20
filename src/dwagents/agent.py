"""Agent creation and wiring."""

from __future__ import annotations

from typing import Any

from deepagents import SubAgent, create_deep_agent
from deepagents.backends.protocol import BackendProtocol
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from dwagents.models.batch import ChatDoublewordBatch


def create_supervisor(
    tools: list[BaseTool | callable | dict[str, Any]] | None = None,
    subagents: list[SubAgent] | None = None,
    system_prompt: str | None = None,
    model: ChatDoublewordBatch | None = None,
    model_kwargs: dict[str, Any] | None = None,
    backend: BackendProtocol | None = None,
    **kwargs,
) -> CompiledStateGraph:
    """Create a deep agent supervisor with doubleword.ai batch inference.

    All LLM calls go through autobatcher for 50-75% cost savings.

    Args:
        tools: Custom tools the agent can use.
        subagents: SubAgent definitions for multi-agent delegation. Each subagent
            that doesn't specify a model reuses the supervisor's model, so their
            LLM calls collate into the same batch window.
        system_prompt: System prompt for the supervisor.
        model: Pre-built ChatDoublewordBatch to use. When multiple supervisors
            share one model instance (primed via _get_client), their LLM calls
            fall into the same BatchOpenAI batch window.
        model_kwargs: Override kwargs for ChatDoublewordBatch (e.g., model_name,
            batch_window_seconds). Ignored when `model` is supplied.
        backend: deepagents backend for the built-in file tools. Defaults to
            deepagents' StateBackend (in-memory virtual FS). Pass
            `deepagents.backends.filesystem.FilesystemBackend()` to have
            read_file/write_file hit the real disk.
        **kwargs: Additional kwargs passed to create_deep_agent.

    Returns:
        A compiled LangGraph state graph ready to invoke.
    """
    if model is None:
        model = ChatDoublewordBatch(**(model_kwargs or {}))

    if subagents:
        resolved_subagents = []
        for sa in subagents:
            if "model" not in sa:
                sa = {**sa, "model": model}
            resolved_subagents.append(sa)
        subagents = resolved_subagents

    if backend is not None:
        kwargs["backend"] = backend

    return create_deep_agent(
        model=model,
        tools=tools or [],
        system_prompt=system_prompt,
        subagents=subagents,
        **kwargs,
    )
