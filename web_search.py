"""
web_search tool for M3 Agent 1.

A thin wrapper around the `ddgs` library, which queries DuckDuckGo's
public search endpoints. No API key required.

Snippets-only: each result has a title, URL, and short body snippet.
We deliberately do NOT fetch full page content — Agent 1 reads what
the snippet provides. If snippet quality turns out to be insufficient
during calibration, we can add a URL-fetch branch later.

Output format (a list of result dicts):
    [
      {
        "title": "Page title",
        "url": "https://example.com/...",
        "snippet": "Short body preview..."
      },
      ...
    ]

On failure (rate limit, network error, no results), returns a structured
error dict that Agent 1 can read and react to:
    {"error": "ratelimit", "message": "..."}
    {"error": "network",   "message": "..."}
    {"error": "empty",     "message": "no results"}

This means Agent 1 always gets a parseable response — no exceptions to
handle in the agent code.
"""

from typing import Any

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException


# Default cap on result count. M3 doesn't typically need more than a
# handful — too many results cost tokens without adding signal.
DEFAULT_MAX_RESULTS = 5


def web_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> dict[str, Any]:
    """Run a DuckDuckGo text search and return structured results.

    Args:
        query: a search query string. Plain text; no special syntax required.
        max_results: cap on number of result entries to return.

    Returns:
        {"results": [...]} on success, where each result has title, url, snippet.
        {"error": <code>, "message": <text>} on failure. Error codes:
          - "ratelimit": DDGS reports too many requests; back off.
          - "network":   timeout or other transport error.
          - "empty":     no results matched the query.
          - "library":   unexpected DDGS library error.

    Why return a dict either way instead of raising:
        Agent 1 reads its tool output as JSON-ish data. Giving it a stable
        shape (results or error) means the agent prompt can describe
        "when the result has an `error` key, do X; otherwise read from
        `results`" without us having to bake exception handling into
        agent control flow.
    """
    if not isinstance(query, str) or not query.strip():
        return {"error": "library", "message": "query must be a non-empty string"}

    try:
        # ddgs is not thread-safe across calls on a shared instance, but
        # constructing a fresh DDGS() per call is cheap (no real session
        # initialization). The Codexity dev journal article specifically
        # advises this pattern.
        ddgs = DDGS()
        raw = ddgs.text(query, max_results=max_results)
    except RatelimitException as e:
        return {"error": "ratelimit", "message": str(e)}
    except TimeoutException as e:
        return {"error": "network", "message": f"timeout: {e}"}
    except DDGSException as e:
        return {"error": "library", "message": str(e)}
    except Exception as e:
        # Catch-all so unexpected errors don't kill Agent 1 mid-turn.
        # Logged as "library" with the exception type for visibility.
        return {"error": "library",
                "message": f"{type(e).__name__}: {e}"}

    if not raw:
        return {"error": "empty", "message": "no results"}

    # Normalize the ddgs field names to our canonical (title, url, snippet).
    # ddgs returns dicts with keys "title", "href", "body".
    results = [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        }
        for r in raw
    ]
    return {"results": results}


# --- Smoke test ---

if __name__ == "__main__":
    # Quick sanity check: search for something unambiguous and print the
    # structured result. Use this to confirm ddgs is installed and reachable
    # before wiring this into Agent 1.
    import json
    print("Searching for: 'Telesquare TLR-2005KSH'")
    out = web_search("Telesquare TLR-2005KSH", max_results=3)
    print(json.dumps(out, ensure_ascii=False, indent=2))
