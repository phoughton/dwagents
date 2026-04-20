"""Compatibility shims for the OpenAI SDK pydantic models.

The doubleword.ai batch endpoint returns ``finish_reason: null`` on some
tool-calling responses. OpenAI's ``Choice`` model declares ``finish_reason`` as
a required ``Literal``, so ``ChatCompletion.model_validate`` raises a
``ValidationError`` — which ``autobatcher`` catches at the outer JSONL-parse
boundary, aborting the loop and orphaning every remaining request in the batch
as ``"No result for request <id>"``.

This module installs a one-time patch on ``ChatCompletion.model_validate`` that
coerces ``finish_reason=None`` to a valid literal before delegating to the
original validator. Import this module once (it's imported by
``dwagents.models.batch``); ``install_patch()`` is idempotent.
"""

from __future__ import annotations

from typing import Any

_PATCHED = False


def install_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    from openai.types.chat import ChatCompletion

    orig_validate = ChatCompletion.model_validate

    def _coerce_and_validate(obj: Any, *args: Any, **kwargs: Any):
        if isinstance(obj, dict):
            for choice in obj.get("choices") or []:
                if isinstance(choice, dict) and choice.get("finish_reason") is None:
                    message = choice.get("message") or {}
                    has_tool_calls = isinstance(message, dict) and message.get("tool_calls")
                    choice["finish_reason"] = "tool_calls" if has_tool_calls else "stop"
        return orig_validate(obj, *args, **kwargs)

    ChatCompletion.model_validate = _coerce_and_validate  # type: ignore[method-assign]
    _PATCHED = True
