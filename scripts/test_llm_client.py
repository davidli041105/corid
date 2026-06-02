"""
Smoke test for the DeepSeek API connection.

Run this once after setting up `.env` to confirm:
  1. The OpenAI SDK is installed
  2. The DEEPSEEK_API_KEY environment variable is readable
  3. We can reach DeepSeek's API
  4. We get a sensible response from the model

If this script prints a model response without errors, step 1 is done
and we can move on to building Bundle Normalization.

Usage:
    python scripts/test_llm_client.py
"""

import sys
from pathlib import Path

# Add project root to path so we can import from `corid/` or top-level modules.
# (Convention: scripts/ sits one level below the project root.)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_client import chat
from config import MODEL_M3


def main():
    print(f"Testing DeepSeek API connection with model: {MODEL_M3}")
    print("-" * 60)

    test_messages = [
        {
            "role": "user",
            "content": "Reply with exactly one short sentence confirming you are reachable.",
        }
    ]

    try:
        reply = chat(test_messages, max_tokens=50)
    except Exception as e:
        print(f"FAILED. Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        sys.exit(1)

    print("Response received:")
    print(f"  {reply}")
    print("-" * 60)
    print("OK — API connection works. Proceed to step 2.")


if __name__ == "__main__":
    main()
