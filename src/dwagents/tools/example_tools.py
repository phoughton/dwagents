"""Example tools for demonstrating agent capabilities."""

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query.
    """
    return f"[Mock result for: {query}] — replace with a real search implementation."


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A mathematical expression to evaluate (e.g., '2 + 2').
    """
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"
