from langchain_openai import ChatOpenAI

from dwagents.config import settings


def create_realtime_model(
    model: str | None = None,
    **kwargs,
) -> ChatOpenAI:
    """Create a real-time ChatOpenAI pointed at doubleword.ai.

    Kwargs override settings defaults (e.g., api_key, base_url).
    """
    kwargs.setdefault("base_url", settings.base_url)
    kwargs.setdefault("api_key", settings.api_key)
    return ChatOpenAI(
        model=model or settings.model,
        **kwargs,
    )
