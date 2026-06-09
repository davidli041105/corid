"""
Internal tools for M3 Agent 1.

"Internal" means these tools operate only on data already inside CoRID
(the bundle, the vocabularies, the workspace). They cannot reach outside
to bring in new information — only `web_search` can. The grounding
contract permits only external tools to bridge probe tokens to labels
via web evidence; internal tools support reasoning hygiene.

Four tools:
  - lookup_device_category() : returns the device category vocabulary
  - lookup_vendor()          : returns the vendor vocabulary
  - response_regularization(field_paths, bundle) : returns a trimmed
                                                    view of the bundle
  - query_refinement(reason) : flag-only signal, returns acknowledgment

Each tool is exposed both as a Python function (for the agent loop to
call when handling tool_calls) and as an OpenAI function descriptor
(for the LLM to invoke).

The agent loop is responsible for binding the bundle to
`response_regularization`, since the tool's input arrives from the LLM
without bundle context.
"""

import copy
from typing import Any

from vocabularies import device_category_vocab, vendor_vocab


# --- Tool implementations ---

def lookup_device_category() -> dict[str, Any]:
    """Return the full device category vocabulary list.

    Returned as a dict so the result shape is consistent with other tools.
    """
    return {"vocabulary": device_category_vocab().list_all()}


def lookup_vendor() -> dict[str, Any]:
    """Return the full vendor vocabulary list."""
    return {"vocabulary": vendor_vocab().list_all()}


def response_regularization(field_paths: list[str], bundle: dict) -> dict[str, Any]:
    """Return a copy of the bundle with the specified field paths removed.

    field_paths uses dotted notation matching the bundle structure:
      "cert_evidence"          -> drops the whole cert_evidence sub-dict
      "location.region_code"   -> drops a single nested field

    Operates on `bundle["evidence"]` (since that's what Agent 1 reads).
    The labels portion is left untouched (Agent 1 never sees labels anyway,
    but defensively we never strip from them either).

    Args:
      field_paths: list of dotted paths to remove
      bundle: the normalized bundle to trim (passed in by the agent loop)

    Returns:
      {"trimmed_evidence": {...}} — the trimmed evidence dict for Agent 1
      to use in subsequent turns.

    Note: this does NOT modify the bundle in place. The agent loop is free
    to decide whether to swap in the trimmed version for subsequent turns
    or keep both copies. (Current plan: the loop replaces the workspace's
    bundle reference with the trimmed version, so subsequent calls see
    the trimmed form.)
    """
    if not isinstance(field_paths, list):
        return {"error": "field_paths must be a list of dotted-path strings"}

    evidence = copy.deepcopy(bundle.get("evidence", {}))
    for path in field_paths:
        if not isinstance(path, str):
            continue
        _drop_path(evidence, path.split("."))
    return {"trimmed_evidence": evidence}


def _drop_path(obj: Any, parts: list[str]) -> None:
    """Recursively walk `parts` into `obj` and delete the final key
    if it exists. No-op if the path doesn't match the structure."""
    if not parts or not isinstance(obj, dict):
        return
    head, *rest = parts
    if not rest:
        # Final segment — delete if present
        obj.pop(head, None)
    else:
        # Descend
        child = obj.get(head)
        if isinstance(child, dict):
            _drop_path(child, rest)


def query_refinement(reason: str) -> dict[str, Any]:
    """Acknowledge that Agent 1 wants to refine its search query.

    This tool has no side effects. Its purpose is to make the intent
    explicit in the conversation trace — when GEPA later analyzes a run,
    a query_refinement call signals "the previous search was insufficient
    and Agent 1 chose to retry." The next turn's web_search call carries
    the refined query Agent 1 came up with.

    Args:
      reason: free-text rationale for refining

    Returns:
      Confirmation. Always succeeds.
    """
    return {
        "acknowledged": True,
        "reason": reason,
        "guidance": "Issue a new web_search call with your refined query.",
    }


# --- OpenAI function descriptors ---
#
# These descriptors are passed to the chat completion as `tools=[...]`.
# Each declares the function's name, purpose, and argument schema in
# OpenAI's standard format. Agent 1's LLM reads these and decides when
# to call each.

LOOKUP_DEVICE_CATEGORY_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_device_category",
        "description": (
            "Return the current Device Category vocabulary list — the "
            "set of canonical category names previously committed in this "
            "run or earlier. Call this before committing a device_category "
            "label, so you can reuse a canonical form if one is semantically "
            "equivalent to what you'd write, or commit a new value if none fits."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

LOOKUP_VENDOR_TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_vendor",
        "description": (
            "Return the current Vendor vocabulary list — the set of canonical "
            "vendor names previously committed. Call this before committing a "
            "vendor label, so you can reuse a canonical form if one matches, "
            "or commit a new value if none fits."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

RESPONSE_REGULARIZATION_TOOL = {
    "type": "function",
    "function": {
        "name": "response_regularization",
        "description": (
            "Trim the normalized bundle by removing specified field paths "
            "(in dotted notation, e.g. 'cert_evidence' or 'location.region_code'). "
            "Use this when parts of the bundle are clearly irrelevant to the "
            "current identification task and you want to focus subsequent "
            "reasoning on what matters. The trimmed evidence will be visible "
            "in subsequent turns of this run."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "field_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of dotted field paths to remove.",
                },
            },
            "required": ["field_paths"],
        },
    },
}

QUERY_REFINEMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "query_refinement",
        "description": (
            "Signal that the latest web_search results were unsatisfactory "
            "(off-topic, low relevance, ambiguous) and that you intend to "
            "issue a refined search query in the next turn. This tool has "
            "no side effects — it makes your refinement intent explicit in "
            "the reasoning trace. The actual refined search happens in your "
            "next web_search call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why the previous search results were insufficient.",
                },
            },
            "required": ["reason"],
        },
    },
}


# Convenience: a registry mapping tool name -> (python function, descriptor).
# The agent loop iterates over this to know how to dispatch tool_calls.
INTERNAL_TOOLS = {
    "lookup_device_category": (lookup_device_category, LOOKUP_DEVICE_CATEGORY_TOOL),
    "lookup_vendor": (lookup_vendor, LOOKUP_VENDOR_TOOL),
    "response_regularization": (response_regularization, RESPONSE_REGULARIZATION_TOOL),
    "query_refinement": (query_refinement, QUERY_REFINEMENT_TOOL),
}


if __name__ == "__main__":
    # Smoke test each tool with trivial inputs
    print("lookup_device_category:", lookup_device_category())
    print("lookup_vendor:", lookup_vendor())

    fake_bundle = {
        "evidence": {
            "data": "HTTP/1.1 200 OK",
            "cert_evidence": {"issuer": "x", "subject": "y"},
            "location": {"city": "Rome", "region_code": "IT"},
        },
        "labels": {"vendor": "ACME"},
    }
    print("response_regularization (drop cert + region_code):")
    print(response_regularization(
        ["cert_evidence", "location.region_code"], fake_bundle
    ))

    print("query_refinement:")
    print(query_refinement("Top results were CVE pages, not vendor specs."))
