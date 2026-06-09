"""
Smoke test for the Astron LLM client.

Two checks:
  1. Plain text chat — confirms basic connectivity, model ID, auth.
  2. Tool-calling chat — confirms the API supports OpenAI-style tool
     calls. This matters because M3 Agent 1 depends on it. If tool
     calling doesn't work on this endpoint, we have to fall back to
     a structured-JSON-output mechanism.

Run from the project root:
    python scripts/test_llm_client.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import chat, chat_text


def test_plain_text():
    print("=" * 70)
    print("TEST 1: plain text chat")
    print("=" * 70)
    reply = chat_text(
        messages=[
            {"role": "user", "content": "Reply with exactly one short sentence confirming you are reachable."}
        ],
        max_tokens=80,
    )
    print(f"Reply: {reply!r}")
    if reply.strip():
        print("✓ Plain text works.")
    else:
        print("✗ Empty reply — connection or model issue.")
    print()


def test_tool_calling():
    print("=" * 70)
    print("TEST 2: tool-calling chat")
    print("=" * 70)

    # A trivial fake tool. We're testing whether the API accepts the tools
    # parameter and whether the model emits a tool_call when asked to.
    fake_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Returns the current server time in ISO 8601 format.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "IANA timezone name, e.g. 'UTC' or 'Asia/Shanghai'."
                        }
                    },
                    "required": ["timezone"]
                }
            }
        }
    ]

    message = chat(
        messages=[
            {"role": "user", "content": "What time is it right now in UTC? Use the available tool."}
        ],
        tools=fake_tools,
        tool_choice="auto",
        max_tokens=200,
    )

    print(f"Response content : {message.content!r}")
    print(f"Tool calls       : {message.tool_calls}")

    if message.tool_calls:
        print(f"✓ Tool calling works. Got {len(message.tool_calls)} tool call(s).")
        for tc in message.tool_calls:
            print(f"  - {tc.function.name}({tc.function.arguments})")
    else:
        print("✗ No tool calls emitted. Either the model declined, or the API")
        print("  doesn't pass through the tools parameter. Inspect content above.")
    print()


def main():
    test_plain_text()
    test_tool_calling()


if __name__ == "__main__":
    main()
