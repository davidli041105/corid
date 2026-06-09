"""
Astron LLM client wrapper.

iFLYTEK's Astron Token Plan exposes an OpenAI-compatible chat completions
endpoint. We use the official `openai` SDK pointed at Astron's base URL.

This module is a thin wrapper that:
  - Reads config (key, base URL, default model) from `config.py`
  - Exposes a `chat()` function with sensible defaults
  - Supports OpenAI-style tool calling (needed for M3 Agent 1)

Why a wrapper rather than calling `openai` directly everywhere?
  - One place to swap providers if we ever change away from Astron
  - Easier to add logging / retry / token-tracking later
  - Tests can mock this one module
  - Centralised handling of provider-specific quirks if they emerge
"""

from typing import Optional, Any
from openai import OpenAI

from config import (
    ASTRON_API_KEY,
    ASTRON_BASE_URL,
    MODEL_ASTRON,
    M3_TEMPERATURE,
    MAX_OUTPUT_TOKENS,
)


def _client() -> OpenAI:
    """Construct the OpenAI client pointed at Astron.

    Built lazily (on each call) rather than as a module-level singleton so
    that import-time errors don't fire just from importing this module —
    e.g., during testing or in environments where the key isn't set yet.
    The OpenAI SDK itself is cheap to construct.
    """
    if not ASTRON_API_KEY:
        raise RuntimeError(
            "ASTRON_API_KEY is not set. "
            "Copy .env.example to .env and fill in your key."
        )
    return OpenAI(api_key=ASTRON_API_KEY, base_url=ASTRON_BASE_URL)


def chat(
    messages: list[dict],
    model: str = MODEL_ASTRON,
    temperature: float = M3_TEMPERATURE,
    max_tokens: Optional[int] = MAX_OUTPUT_TOKENS,
    tools: Optional[list[dict]] = None,
    tool_choice: Optional[str] = None,
) -> Any:
    """Send a chat completion request.

    Args:
        messages: list of {"role": "...", "content": "..."} dicts (or
            assistant messages with tool_calls). OpenAI format.
        model: model identifier (defaults to MODEL_ASTRON from config).
        temperature: sampling temperature.
        max_tokens: optional cap on output length.
        tools: optional list of OpenAI-format tool descriptors. When
            provided, the model can emit tool_calls in its response.
            Used by M3 Agent 1 (Information Extractor).
        tool_choice: optional control over tool selection ("auto", "none",
            or {"type": "function", "function": {"name": "..."}}).

    Returns:
        The full response message object (not just the content string).
        Callers should access `.content` for text output and `.tool_calls`
        for tool invocations. We return the message rather than just the
        text because Agent 1 needs to inspect both.

        For text-only callers (e.g., Agent 2), use `chat_text()` instead.

    Raises:
        RuntimeError if the API key is missing.
        openai.APIError (or subclass) on API-level failures — we let these
        bubble up so callers can decide retry policy.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if tools is not None:
        kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

    response = _client().chat.completions.create(**kwargs)
    return response.choices[0].message


def chat_text(
    messages: list[dict],
    model: str = MODEL_ASTRON,
    temperature: float = M3_TEMPERATURE,
    max_tokens: Optional[int] = MAX_OUTPUT_TOKENS,
) -> str:
    """Convenience wrapper for text-only chat calls.

    Use this when you don't need tool calling — most callers (Agent 2,
    GEPA reflect-LM, translation utilities) only want the assistant's
    text response.
    """
    message = chat(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return message.content or ""
