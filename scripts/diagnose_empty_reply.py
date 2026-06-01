"""
Diagnostic for the empty-response issue.

Goal: see the *full* response object from DeepSeek, not just the .content
field. This tells us:
  - Did the API actually return something?
  - Is the content in a different field (e.g., reasoning_content)?
  - Did the request hit a length or finish-reason limit?
  - What is finish_reason — "stop", "length", "tool_calls", something else?

Run after the empty-reply smoke test to identify the cause.

Usage:
    python scripts/diagnose_empty_reply.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, MODEL_M3


def main():
    if not DEEPSEEK_API_KEY:
        print("DEEPSEEK_API_KEY not set.")
        sys.exit(1)

    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

    print(f"Diagnostic call to model: {MODEL_M3}")
    print("Asking for a longer reply with more tokens, to rule out budget issues.")
    print("=" * 70)

    response = client.chat.completions.create(
        model=MODEL_M3,
        messages=[
            {
                "role": "user",
                "content": "Reply with one short sentence confirming you are reachable.",
            }
        ],
        temperature=0.2,
        max_tokens=500,  # plenty of budget — rules out length cutoff
    )

    # Print the raw structure as JSON so we see everything the API returned.
    # .model_dump() is a Pydantic method that converts the response object
    # into a plain dict, then we json.dumps it for readability.
    print("FULL RESPONSE OBJECT:")
    print(json.dumps(response.model_dump(), indent=2, default=str))

    print("=" * 70)
    print("Key fields to inspect:")
    print(f"  finish_reason   : {response.choices[0].finish_reason}")
    print(f"  content         : {response.choices[0].message.content!r}")

    # DeepSeek's thinking-mode reply puts the chain-of-thought in
    # `reasoning_content`. If that field is populated and `content` is empty,
    # we've found the cause.
    message = response.choices[0].message
    if hasattr(message, "reasoning_content"):
        rc = getattr(message, "reasoning_content", None)
        print(f"  reasoning_content (first 200 chars): {(rc or '')[:200]!r}")
    else:
        print("  reasoning_content : (not present on this message)")

    print(f"  usage           : {response.usage}")


if __name__ == "__main__":
    main()
