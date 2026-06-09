"""
web_search tool for M3 Agent 1.

A thin wrapper around the `ddgs` library, which queries DuckDuckGo's
public search endpoints. No API key required.

Snippets-only: each result has a title, URL, and short body snippet.
We deliberately do NOT fetch full page content — Agent 1 reads what
the snippet provides. If snippet quality turns out to be insufficient
during calibration, we can add a URL-fetch branch later.

Exposes:
  - web_search(query, max_results) : python entry point
  - WEB_SEARCH_TOOL : OpenAI function descriptor for Agent 1's LLM
"""

from typing import Any

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException


DEFAULT_MAX_RESULTS = 5


def web_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> dict[str, Any]:
    """Run a DuckDuckGo text search and return structured results.

    Returns:
        {"results": [...]} on success, each result has title, url, snippet.
        {"error": <code>, "message": <text>} on failure. Error codes:
          - "ratelimit": DDGS reports too many requests; back off.
          - "network":   timeout or other transport error.
          - "empty":     no results matched the query.
          - "library":   unexpected DDGS library error.

    Why return a dict either way instead of raising:
        Agent 1 reads tool output as data. Giving it a stable shape means
        the agent's prompt can describe "when result has 'error', do X;
        otherwise read 'results'" without us baking exception handling
        into agent control flow.
    """
    if not isinstance(query, str) or not query.strip():
        return {"error": "library", "message": "query must be a non-empty string"}

    try:
        # ddgs is not thread-safe across calls on a shared instance; constructing
        # a fresh DDGS() per call is cheap and safer.
        ddgs = DDGS()
        raw = ddgs.text(query, max_results=max_results)
    except RatelimitException as e:
        return {"error": "ratelimit", "message": str(e)}
    except TimeoutException as e:
        return {"error": "network", "message": f"timeout: {e}"}
    except DDGSException as e:
        return {"error": "library", "message": str(e)}
    except Exception as e:
        return {"error": "library",
                "message": f"{type(e).__name__}: {e}"}

    if not raw:
        return {"error": "empty", "message": "no results"}

    return {
        "results": [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in raw
        ]
    }


# --- OpenAI function descriptor ---

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web via DuckDuckGo. Returns up to N result snippets, "
            "each with title, URL, and a short body preview. Use this to "
            "interpret probe-side tokens (e.g., look up what a model number "
            "means) or confirm vendor/category candidates. Results are "
            "snippets only — there is no full-page fetching. "
            "If the API returns an error (rate limit, network failure, no "
            "results), the response will include an 'error' field instead "
            "of 'results'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Plain text search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of result snippets to return (default 5).",
                },
            },
            "required": ["query"],
        },
    },
}


if __name__ == "__main__":
    import json
    print("Searching for: 'Telesquare TLR-2005KSH'")
    out = web_search("Telesquare TLR-2005KSH", max_results=3)
    print(json.dumps(out, ensure_ascii=False, indent=2))
