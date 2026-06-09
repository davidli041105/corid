"""
M3 Agent 1 — Information Extractor.

Reads a normalized bundle, uses tools (web_search + 4 internal), maintains
a three-bucket workspace (Bundle Analysis, Tool Findings, Current Hypothesis),
and emits a structured output that Agent 2 (Rule Inducer) consumes.

Critical contract: every observation in `bundle_analysis` MUST carry
`anchored_to` information — the probe-side (token, field) pairs that
justify the observation. This is what makes the grounding contract
survive the agent split (see spec §6 implementation note).

Output shape (canonical):
    {
      "bundle_analysis": [
        {
          "observation": "...",
          "anchored_to": [{"token": "...", "field": "..."}, ...],
          "via": "direct" | "web_evidence",
          "web_source": "https://..."  // present only when via == "web_evidence"
        },
        ...
      ],
      "tool_findings": [
        {"tool": "web_search", "query": "...", "summary": "..."},
        ...
      ],
      "current_hypothesis": {
        "device_category": "..." | null,
        "vendor": "..." | null,
        "model": "..." | null,
        "confidence": "high" | "medium" | "low"
      }
    }

The agent loop is a standard OpenAI tool-calling loop. The LLM decides
when to call which tool. The loop terminates when the assistant emits
a final message with no tool_calls; that message body should contain
the JSON payload above.
"""

import json
from typing import Any

from llm_client import chat
from config import MODEL_M3_AGENT_1, M3_TEMPERATURE, MAX_OUTPUT_TOKENS
from web_search import web_search, WEB_SEARCH_TOOL
from internal_tools import (
    INTERNAL_TOOLS,
    response_regularization,
)


# --- Tool registry ---
#
# Combine web_search + the 4 internal tools into the tool list the LLM
# sees. The agent loop dispatches tool_calls by name using this registry.

ALL_TOOLS_DESCRIPTORS = [
    WEB_SEARCH_TOOL,
    INTERNAL_TOOLS["lookup_device_category"][1],
    INTERNAL_TOOLS["lookup_vendor"][1],
    INTERNAL_TOOLS["response_regularization"][1],
    INTERNAL_TOOLS["query_refinement"][1],
]


# Maximum tool-call turns before we give up. Without a cap, a misbehaving
# LLM could loop indefinitely. We're not enforcing a token budget per the
# spec, but we do want a runaway-loop safeguard.
MAX_TURNS = 20


# --- System prompt ---
#
# The initial system prompt is intentionally moderate in length. GEPA will
# evolve this over calibration rounds. For now, the prompt has to:
#   - Explain the role (information extraction, not rule induction)
#   - Describe the workspace + output schema
#   - Explain each tool
#   - Reinforce the anchor requirement
#   - Forbid Agent 1 from emitting a draft rule (that's Agent 2's job)

SYSTEM_PROMPT = """You are the Information Extractor for CoRID, a device-identification framework.

ROLE
----
Given a single device's probe-derived evidence bundle (banner text, HTTP
fields, hostnames, etc.), you must extract structured observations about
what the bundle reveals — vendor candidates, device category candidates,
model candidates, and supporting evidence — using your reasoning plus
the available tools.

You DO NOT produce the final identification rule. Another component
(Agent 2) reads your structured output and induces the rule. Your job
is information extraction, not rule induction.

WORKSPACE
---------
Maintain three buckets of state across your turns:

  1. Bundle Analysis: observations about what's in the bundle.
  2. Tool Findings: summaries of what each tool call produced.
  3. Current Hypothesis: your best current guess at (device_category,
     vendor, model) — may include nulls for fields you cannot ground.

When you call tools, use the results to enrich these buckets. When you
finish, emit them as your final JSON output.

GROUNDING CONTRACT (HARD CONSTRAINTS — violations make the output invalid)
-------------------------------------------------------------------------
1. Every observation in Bundle Analysis MUST include `anchored_to`:
   a list of (token, field) pairs from the probe-side evidence that
   justified the observation. Observations without anchors are forbidden.

2. There are exactly two source types:
   - via="direct": the observation comes straight from probe tokens.
     Example: cpe23 contains "telesquare" -> vendor candidate "Telesquare".
     Anchor: [{"token": "telesquare", "field": "cpe23"}].

   - via="web_evidence": you used web_search to interpret a probe token,
     and the web result informed the observation. The anchor MUST STILL
     reference the probe-side token that was looked up, plus the URL
     of the supporting web source.
     Anchor: [{"token": "tlr-2005ksh", "field": "data"}],
     via="web_evidence", web_source="<url>".

3. If you cannot ground an observation in a probe token, do not emit it.
4. If you cannot ground a hypothesis field, set it to null. Do not guess.

MANDATORY VOCABULARY LOOKUP (HARD CONSTRAINT)
---------------------------------------------
Before you may emit a non-null value for `current_hypothesis.vendor`,
you MUST have called `lookup_vendor` at least once during this run.

Before you may emit a non-null value for `current_hypothesis.device_category`,
you MUST have called `lookup_device_category` at least once during this run.

This holds even if the vocabulary list is empty when you check —
the lookup is mandatory for audit-trail purposes. The result of the
lookup tells you whether a canonical form already exists you should
reuse, or whether you are committing a new canonical value.

Failure to perform the mandatory lookup before committing makes the
output invalid. There are NO exceptions to this rule, including when
the evidence seems unambiguous.

WHEN TO USE web_search
----------------------
Call web_search when:
  - A probe-side token (model number, banner string, server identifier)
    is unfamiliar and you need to interpret it.
  - You see a vendor or category candidate but want corroboration before
    committing.
  - The bundle is ambiguous and external context could disambiguate.

You are NOT required to call web_search if direct evidence is already
sufficient to support a confident commit. But if you DO call it and
the results return an error ({"error": ...}), do not pretend it
succeeded — either retry with a refined query (after query_refinement),
or proceed without web evidence for that observation and set
confidence accordingly.

TOOLS
-----
  - web_search(query, max_results): DuckDuckGo search; returns snippets.
    No full-page fetching. Use to interpret probe tokens.
  - lookup_device_category(): returns canonical category vocabulary.
    MANDATORY before non-null device_category commit.
  - lookup_vendor(): returns canonical vendor vocabulary.
    MANDATORY before non-null vendor commit.
  - response_regularization(field_paths): trim the bundle by removing
    irrelevant fields. Subsequent turns see the trimmed bundle.
  - query_refinement(reason): signal that the last web_search was
    unsatisfactory; explain why. Issue a refined search in the next turn.

FINAL OUTPUT
------------
When you have completed all required lookups and have nothing more to
investigate, emit exactly one final assistant message containing only
a JSON object with this structure:

{
  "bundle_analysis": [
    {
      "observation": "<what you noticed>",
      "anchored_to": [{"token": "<probe token>", "field": "<bundle field path>"}, ...],
      "via": "direct" | "web_evidence",
      "web_source": "<url if via=web_evidence, else omit>"
    },
    ...
  ],
  "tool_findings": [
    {"tool": "<tool name>", "query": "<input>", "summary": "<what was learned>"},
    ...
  ],
  "current_hypothesis": {
    "device_category": "<canonical category>" | null,
    "vendor": "<canonical vendor>" | null,
    "model": "<canonical model>" | null,
    "confidence": "high" | "medium" | "low"
  }
}

No prose around the JSON. No code fences. Just the JSON object.

ABOUT `tool_findings`
---------------------
`tool_findings` is a SEMANTIC SUMMARY of what you learned from your tool
use — not an audit log of invocations. The rules:

  - One entry per tool that contributed information you used in your
    reasoning. If you called the same tool multiple times but the result
    informed a single insight, one entry is enough. Counts and timing
    don't matter; what was learned matters.

  - Every entry you write MUST correspond to a tool you actually invoked
    via the OpenAI tool-calling mechanism in this run. Do NOT include
    a tool in `tool_findings` if you never called it. The runtime
    validates this — fabricating a finding makes the output invalid
    and the run fails.

  - It is acceptable for a tool you invoked to not appear in
    `tool_findings` if its result didn't actually inform your reasoning
    (e.g., a lookup that returned empty and led nowhere). But if you
    used a tool's output to support an observation or hypothesis, that
    tool's finding should be summarized here.

In short: do not invent findings, but also do not feel obligated to log
every call. Summarize what you learned, anchored to tools that actually
ran."""


# Number of attempts before giving up. The first attempt is included
# in the count, so MAX_ATTEMPTS=3 means "initial try + up to 2 retries."
# After exhausting all attempts we return the last result regardless —
# success if any attempt succeeded, otherwise the final failure.
MAX_ATTEMPTS = 3


# --- Agent loop ---

def run_agent_1(bundle: dict, verbose: bool = False) -> dict[str, Any]:
    """Run Agent 1 on one normalized bundle, with bounded retries.

    Tries up to MAX_ATTEMPTS times. Returns as soon as any attempt
    produces a result without an `_error` field. If all attempts fail,
    returns the last failed attempt's output (which still carries the
    `_error` field, plus `_trace` for inspection).

    The returned dict always includes:
      - `_attempts`: how many attempts were made (1..MAX_ATTEMPTS)
    """
    last_result: dict[str, Any] = {}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if verbose:
            print(f"=== Attempt {attempt}/{MAX_ATTEMPTS} ===")
        last_result = _run_agent_1_once(bundle, verbose=verbose)
        if "_error" not in last_result:
            last_result["_attempts"] = attempt
            return last_result
        if verbose:
            print(f"  attempt {attempt} failed: {last_result.get('_error')}")

    last_result["_attempts"] = MAX_ATTEMPTS
    return last_result


def _run_agent_1_once(bundle: dict, verbose: bool = False) -> dict[str, Any]:
    """Single attempt of the Agent 1 tool-calling loop.

    See run_agent_1 for the retry wrapper. This function performs one
    full tool-calling loop and returns either a parsed output or an
    error dict.
    """
    # The bundle we expose to the agent is a working copy; if it calls
    # response_regularization, we replace this reference. The original
    # bundle stays intact.
    current_bundle = {"evidence": dict(bundle.get("evidence", {}))}

    # Initial user message: hand over the bundle and ask the agent to begin.
    initial_user_message = (
        "Here is the normalized evidence bundle for one device. Extract "
        "structured observations and a current hypothesis per the rules in "
        "your system prompt.\n\n"
        f"BUNDLE EVIDENCE:\n{json.dumps(current_bundle['evidence'], ensure_ascii=False, indent=2)}"
    )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_user_message},
    ]

    for turn in range(MAX_TURNS):
        if verbose:
            print(f"--- Turn {turn + 1} ---")

        # Call the model with all tools available.
        try:
            assistant_msg = chat(
                messages=messages,
                model=MODEL_M3_AGENT_1,
                temperature=M3_TEMPERATURE,
                max_tokens=MAX_OUTPUT_TOKENS,
                tools=ALL_TOOLS_DESCRIPTORS,
                tool_choice="auto",
            )
        except Exception as e:
            return {
                "_error": f"LLM call failed: {type(e).__name__}: {e}",
                "_trace": messages,
            }

        # Append the assistant message to the history (whether or not it has
        # tool_calls — OpenAI requires the assistant turn precede the tool
        # responses).
        messages.append(_assistant_message_to_dict(assistant_msg))

        # If the model emitted tool_calls, dispatch each and append the
        # tool responses; then loop.
        if assistant_msg.tool_calls:
            for tc in assistant_msg.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    tool_result = {"error": "invalid_arguments",
                                   "message": f"JSON parse error: {e}"}
                else:
                    tool_result = _dispatch_tool(
                        tool_name, args, current_bundle
                    )
                    # response_regularization can replace the working bundle
                    if tool_name == "response_regularization" and "trimmed_evidence" in tool_result:
                        current_bundle = {"evidence": tool_result["trimmed_evidence"]}

                if verbose:
                    print(f"  tool_call: {tool_name}({tc.function.arguments})")
                    print(f"  -> {json.dumps(tool_result, ensure_ascii=False)[:200]}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tool_name,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

            # Continue the loop — model gets to see tool results and respond.
            continue

        # No tool_calls — this is the final assistant message. Parse JSON.
        content = (assistant_msg.content or "").strip()
        parsed = _parse_final_output(content)
        if parsed is None:
            return {
                "_error": "final assistant message did not contain valid JSON",
                "_raw_content": content,
                "_trace": messages,
            }

        # Truthfulness check (asymmetric):
        #   - Hallucinated findings (claimed but not invoked) are FATAL.
        #   - Missing summaries (invoked but not summarized) are NOT fatal;
        #     we augment tool_findings with stub entries and continue.
        # The first preserves the "no hallucination" principle; the second
        # accepts that a tool whose result didn't inform reasoning may
        # legitimately not need a semantic summary.
        violation, warnings = _validate_and_augment_tool_findings(parsed, messages)
        if violation is not None:
            return {
                "_error": f"tool_findings hallucination: {violation}",
                "_raw_output": parsed,
                "_trace": messages,
            }

        if warnings:
            parsed["_warnings"] = warnings

        parsed["_trace"] = messages
        return parsed

    # Loop fell through — too many turns
    return {
        "_error": f"agent did not terminate within {MAX_TURNS} turns",
        "_trace": messages,
    }


# --- Helpers ---

def _dispatch_tool(name: str, args: dict, bundle: dict) -> dict:
    """Route a tool_call to its implementation."""
    if name == "web_search":
        return web_search(
            query=args.get("query", ""),
            max_results=args.get("max_results", 5),
        )
    if name == "response_regularization":
        return response_regularization(
            field_paths=args.get("field_paths", []),
            bundle=bundle,
        )
    if name in INTERNAL_TOOLS:
        fn = INTERNAL_TOOLS[name][0]
        # lookup_device_category and lookup_vendor take no args;
        # query_refinement takes reason. Filter args accordingly.
        if name in ("lookup_device_category", "lookup_vendor"):
            return fn()
        if name == "query_refinement":
            return fn(reason=args.get("reason", ""))
    return {"error": "unknown_tool", "message": f"no tool named {name!r}"}


def _assistant_message_to_dict(msg: Any) -> dict:
    """Convert an OpenAI message object into the dict shape that the API
    expects when echoed back in the message history."""
    out: dict[str, Any] = {"role": "assistant"}
    if msg.content is not None:
        out["content"] = msg.content
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return out


def _parse_final_output(content: str) -> dict | None:
    """Try to parse the assistant's final content as the expected JSON
    output schema. Returns the parsed dict on success, None on failure.

    Tolerant of:
      - Markdown code fences (```json ... ```)
      - Preamble text before the JSON (e.g., "Here is the result: { ... }")
      - Trailing text after the JSON

    Strategy: find the first '{' in the content, then walk forward
    tracking brace depth (skipping over braces inside string literals)
    until depth returns to zero. The substring between those positions
    is the candidate JSON object. Try to parse it.

    This is more robust than relying on the model to obey the
    "no prose around the JSON" instruction, which Astron has been
    observed to ignore in practice.
    """
    if not content:
        return None

    candidate = _extract_json_object(content)
    if candidate is None:
        return None

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _extract_json_object(s: str) -> str | None:
    """Extract the first balanced JSON object substring from `s`.

    Walks character by character, tracking:
      - brace depth (only at the top level of strings)
      - whether we're inside a string literal (so braces inside strings
        don't affect depth)
      - escape characters inside strings

    Returns the substring from the first unescaped '{' to its matching '}',
    or None if no balanced object is found.
    """
    start = s.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        c = s[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def _validate_and_augment_tool_findings(parsed_output: dict, trace: list[dict]) -> tuple[str | None, list[str]]:
    """Reconcile `tool_findings` with the actual trace.

    Two classes of mismatch, both treated non-fatally:

    1. Hallucinated findings — `tool_findings` mentions a tool that was
       NEVER invoked in the trace. We STRIP these entries from the
       output silently (with a warning logged) rather than failing the
       run. Rationale: on bundles with overwhelming direct evidence the
       model sometimes shortcuts and skips mandatory lookups while still
       trying to look compliant by writing them into tool_findings. We
       don't want to fail the whole run for this — the bundle_analysis
       observations are still anchored to real probe tokens, and the
       commit is still grounded in evidence. We just clean up the
       fabricated bookkeeping.

    2. Missing summaries — a tool WAS invoked but no `tool_findings`
       entry summarizes it. We augment with a stub entry so downstream
       consumers see all invoked tools.

    Returns:
      (violation, warnings) — violation is always None (this validator
      no longer produces fatal violations). warnings is a list of
      human-readable notes about anything that was changed.

    Mutates `parsed_output["tool_findings"]` to strip hallucinated entries
    and add stub entries for missing ones.
    """
    # Collect actually-invoked tool names from the trace
    actually_called: set[str] = set()
    for msg in trace:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            name = fn.get("name")
            if name:
                actually_called.add(name)

    findings = parsed_output.get("tool_findings", []) or []
    if not isinstance(findings, list):
        findings = []
    parsed_output["tool_findings"] = findings

    # Strip hallucinated entries — keep only those for tools actually called.
    original_count = len(findings)
    cleaned: list[dict] = []
    stripped_tools: set[str] = set()
    for tf in findings:
        if isinstance(tf, dict):
            name = tf.get("tool")
            if name in actually_called:
                cleaned.append(tf)
            elif name:
                stripped_tools.add(name)
        # Non-dict entries are also dropped (they can't be valid)

    # Replace the list contents in place so any external reference
    # (and the parsed_output dict) sees the cleaned version.
    findings.clear()
    findings.extend(cleaned)

    warnings: list[str] = []
    if stripped_tools:
        warnings.append(
            f"stripped fabricated tool_findings entries for tools that "
            f"were never invoked: {sorted(stripped_tools)}"
        )

    # Augment with stubs for tools that ran but weren't summarized.
    claimed_after_strip: set[str] = set()
    for tf in cleaned:
        name = tf.get("tool")
        if name:
            claimed_after_strip.add(name)

    missing = actually_called - claimed_after_strip
    if missing:
        warnings.append(
            f"tools invoked but not summarized in tool_findings: "
            f"{sorted(missing)} (stub entries added)"
        )
        for tool_name in sorted(missing):
            findings.append({
                "tool": tool_name,
                "query": "(not summarized by model)",
                "summary": "(model invoked this tool but did not summarize its result)",
            })

    return None, warnings


# --- Smoke test ---

if __name__ == "__main__":
    # Run Agent 1 on the first bundle from the calibration sample.
    from loader import load_clean_records_list
    from normalization import normalize_record

    records = load_clean_records_list()
    bundle = normalize_record(records[0])

    print(f"Running Agent 1 on first bundle...")
    print(f"True labels (held back from agent): {bundle['labels']}")
    print()

    output = run_agent_1(bundle, verbose=True)
    print()
    print("=" * 70)
    print("Final output:")
    print("=" * 70)
    # Don't print the full trace; it's long
    output_for_display = {k: v for k, v in output.items() if k != "_trace"}
    print(json.dumps(output_for_display, ensure_ascii=False, indent=2))
