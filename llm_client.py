"""
DeepSeek LLM client wrapper.

DeepSeek's API is OpenAI-compatible — we use the official `openai` SDK
pointed at DeepSeek's base URL. This module is a thin wrapper that:
  - Reads config (key, base URL, default model) from `config.py`
  - Exposes a `chat()` function with sensible defaults
  - Centralises error handling so caller code stays clean

Why a wrapper rather than calling `openai` directly everywhere?
  - One place to swap providers if we ever change away from DeepSeek
  - Easier to add logging / retry / token-tracking later
  - Tests can mock this one module
"""

from typing import Optional
from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    MODEL_M3,
    M3_TEMPERATURE,
)


def _client() -> OpenAI:
    """Construct the OpenAI client pointed at DeepSeek.

    We build it lazily (on each call) rather than as a module-level singleton
    so that import-time errors don't fire just from importing this module —
    e.g., during testing or in environments where the key isn't set yet.
    The OpenAI SDK itself is cheap to construct.
    """
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set. "
            "Copy .env.example to .env and fill in your key."
        )
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def chat(
    messages: list[dict],
    model: str = MODEL_M3,
    temperature: float = M3_TEMPERATURE,
    max_tokens: Optional[int] = None,
) -> str:
    """Send a chat completion request and return the assistant's reply text.

    Args:
        messages: list of {"role": "...", "content": "..."} dicts in OpenAI format.
        model: model identifier (defaults to MODEL_M3 from config).
        temperature: sampling temperature.
        max_tokens: optional cap on output length.

    Returns:
        The assistant message content as a string.

    Raises:
        RuntimeError if the API key is missing.
        openai.APIError (or subclass) on API-level failures — we let these
        bubble up rather than swallow, so caller can decide retry policy.
    """
    response = _client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
