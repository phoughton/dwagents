"""Agent creation and wiring."""

from __future__ import annotations

from typing import Any

from deepagents import SubAgent, create_deep_agent
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from dwagents.models.batch import ChatDoublewordBatch


def create_supervisor(
    tools: list[BaseTool | callable | dict[str, Any]] | None = None,
    subagents: list[SubAgent] | None = None,
    system_prompt: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> CompiledStateGraph:
    """Create a deep agent supervisor with doubleword.ai batch inference.

    All LLM calls go through autobatcher for 50-75% cost savings.

    Args:
        tools: Custom tools the agent can use.
        subagents: SubAgent definitions for multi-agent delegation. Each subagent
            that doesn't specify a model will get its own ChatDoublewordBatch instance.
        system_prompt: System prompt for the supervisor.
        model_kwargs: Override kwargs for ChatDoublewordBatch (e.g., model_name, batch_window_seconds).
        **kwargs: Additional kwargs passed to create_deep_agent.

    Returns:
        A compiled LangGraph state graph ready to invoke.
    """
    model = ChatDoublewordBatch(**(model_kwargs or {}))

    # Ensure subagents without explicit models get their own batch instance
    if subagents:
        resolved_subagents = []
        for sa in subagents:
            if "model" not in sa:
                sa = {**sa, "model": ChatDoublewordBatch(**(model_kwargs or {}))}
            resolved_subagents.append(sa)
        subagents = resolved_subagents

    return create_deep_agent(
        model=model,
        tools=tools or [],
        system_prompt=system_prompt,
        subagents=subagents,
        **kwargs,
    )
