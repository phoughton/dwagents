"""Tests for dwagents.parallel — prompt loading and parallel agent runner."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dwagents import load_prompts_from_dir, run_agents_parallel


class TestLoadPromptsFromDir:
    def test_reads_txt_and_md_and_uses_stem_as_name(self, tmp_path):
        (tmp_path / "alpha.md").write_text("first prompt\n")
        (tmp_path / "beta.txt").write_text("  second prompt  \n")
        (tmp_path / "skip.yaml").write_text("ignored")

        prompts = load_prompts_from_dir(tmp_path)

        assert prompts == {"alpha": "first prompt", "beta": "second prompt"}

    def test_raises_when_dir_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_prompts_from_dir(tmp_path / "nope")

    def test_raises_when_no_matching_files(self, tmp_path):
        (tmp_path / "unrelated.yaml").write_text("x")
        with pytest.raises(ValueError):
            load_prompts_from_dir(tmp_path)


def _fake_agent_factory(observed: list, delay: float = 0.0, fail_on: str | None = None):
    """Build a fake create_supervisor that returns an agent recording the model
    (and its _client at build time) and the prompt it was invoked with."""

    def _fake_create_supervisor(*, tools=None, system_prompt=None, model=None, subagents=None, **_):
        # Record (model, client) now — aclose() in the runner will null _client
        # after gather completes.
        observed.append((model, getattr(model, "_client", None)))

        class _FakeAgent:
            async def ainvoke(self, state, config=None):
                prompt = state["messages"][0]["content"]
                if delay:
                    await asyncio.sleep(delay)
                if fail_on is not None and prompt == fail_on:
                    raise RuntimeError(f"boom on {prompt}")
                return {"messages": [{"role": "assistant", "content": f"ok:{prompt}"}]}

        return _FakeAgent()

    return _fake_create_supervisor


def _patch_batch_client():
    """Patch BatchOpenAI so instances are MagicMocks with an async close()."""
    mock_cls = MagicMock()
    mock_cls.return_value.close = AsyncMock()
    return patch("dwagents.models.batch.BatchOpenAI", mock_cls)


class TestRunAgentsParallel:
    def test_all_agents_share_the_same_batch_client(self):
        observed_models: list = []
        with patch("dwagents.parallel.create_supervisor", _fake_agent_factory(observed_models)), \
             _patch_batch_client() as MockBatchOpenAI:
            prompts = {"one": "a", "two": "b", "three": "c"}
            asyncio.run(run_agents_parallel(
                prompts,
                tools=[],
                model_kwargs={"api_key": "k", "base_url": "http://x/", "model_name": "m"},
            ))

        # BatchOpenAI instantiated exactly once — all agents share it.
        assert MockBatchOpenAI.call_count == 1

        # Every supervisor got the same ChatDoublewordBatch instance.
        assert len(observed_models) == 3
        first_model, first_client = observed_models[0]
        assert all(m is first_model for m, _ in observed_models)

        # And that model's client is the single mocked BatchOpenAI.
        shared_client = MockBatchOpenAI.return_value
        assert all(c is shared_client for _, c in observed_models)
        assert first_client is shared_client

    def test_runs_concurrently_not_serially(self):
        observed: list = []
        with patch("dwagents.parallel.create_supervisor", _fake_agent_factory(observed, delay=0.3)), \
             _patch_batch_client():
            prompts = {f"p{i}": f"content-{i}" for i in range(3)}
            start = time.perf_counter()
            results = asyncio.run(run_agents_parallel(prompts, tools=[]))
            elapsed = time.perf_counter() - start

        # 3 prompts x 0.3s each = 0.9s serial; concurrent should be well under.
        assert elapsed < 0.7, f"ran serially ({elapsed:.2f}s)"
        assert set(results) == set(prompts)

    def test_partial_failure_captured_without_killing_others(self):
        observed: list = []
        with patch("dwagents.parallel.create_supervisor", _fake_agent_factory(observed, fail_on="bad")), \
             _patch_batch_client():
            prompts = {"good": "ok", "explode": "bad"}
            results = asyncio.run(run_agents_parallel(prompts, tools=[]))

        assert isinstance(results["explode"], RuntimeError)
        assert results["good"]["messages"][0]["content"] == "ok:ok"

    def test_empty_prompts_short_circuits(self):
        with patch("dwagents.parallel.create_supervisor") as mock_create, \
             _patch_batch_client() as MockBatchOpenAI:
            results = asyncio.run(run_agents_parallel({}, tools=[]))
        assert results == {}
        # Short-circuit: no model or supervisor built.
        mock_create.assert_not_called()
        MockBatchOpenAI.assert_not_called()
