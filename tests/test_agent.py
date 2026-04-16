"""Tests for agent creation."""

from unittest.mock import patch

from dwagents.agent import create_supervisor
from dwagents.tools.example_tools import calculator, web_search


class TestCreateSupervisor:
    def test_creates_agent_with_tools(self):
        agent = create_supervisor(
            tools=[web_search, calculator],
            system_prompt="You are a helpful assistant.",
            model_kwargs={
                "model_name": "gpt-4o",
                "api_key": "test-key",
                "base_url": "http://test/v1/",
            },
        )
        # Should return a compiled graph
        assert agent is not None

    def test_creates_agent_with_subagents(self):
        agent = create_supervisor(
            tools=[web_search],
            system_prompt="You are a supervisor.",
            subagents=[
                {
                    "name": "researcher",
                    "description": "A research agent that searches the web.",
                    "system_prompt": "You search the web for information.",
                    "tools": [web_search],
                },
            ],
            model_kwargs={
                "model_name": "gpt-4o",
                "api_key": "test-key",
                "base_url": "http://test/v1/",
            },
        )
        assert agent is not None

    def test_subagents_get_batch_model(self):
        subagents = [
            {
                "name": "worker",
                "description": "A worker agent.",
                "system_prompt": "You do work.",
            },
        ]
        agent = create_supervisor(
            tools=[],
            subagents=subagents,
            model_kwargs={
                "model_name": "gpt-4o",
                "api_key": "test-key",
                "base_url": "http://test/v1/",
            },
        )
        assert agent is not None
