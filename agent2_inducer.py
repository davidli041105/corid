"""
M3 Agent 2 — Rule Inducer.

Reads Agent 1's structured output (bundle_analysis, tool_findings,
current_hypothesis) and emits a draft rule that proceeds to M4 for
PMI filtering and library-consistency checking.

Agent 2 has NO tools. It is a pure reasoner over Agent 1's output.
Its job is to turn Agent 1's anchored observations into a draft rule
with per-keyword `generalizable_core` annotations.

Output shape (canonical):
    {
      "draft_rule": {
        "labels": {
          "device_category": "..." | null,
          "vendor": "..." | null,
          "model": "..." | null
        },
        "confidence": "high" | "medium" | "low",
        "candidate_keywords": [
          {
            "source_field": "http.server",
            "anchor_token": "KM-MFP-http/V0.0.1",
            "generalizable_core": "KM-MFP-http",
            "supports_labels": ["vendor", "device_category"]
          },
          ...
        ]
      }
    }

The agent loop is a single chat-completion call (no tools, no loop).
After parsing the model's JSON output, the loop enforces the
strict_anchor_binding contract: every non-null label in the draft
must be supported by at least one keyword whose supports_labels
includes that label. If a label is unsupported, the run is rejected
with an explicit error — so downstream sees the violation rather than
a silently-broken rule.

Note on robustness — Agent 1's recent runs showed Astron sometimes
fabricates content (the tool_findings hallucination pattern). We carry
forward the same defensive patterns here: JSON extraction tolerant of
preambles and code fences; retry loop with bounded attempts; clear
error reporting.
"""

import json
from typing import Any

from llm_client import chat_text
from config import MODEL_M3_AGENT_2, M3_TEMPERATURE, MAX_OUTPUT_TOKENS


# Retry budget. First attempt plus up to 2 retries. Agent 2 is simpler
# than Agent 1 (no tools, single LLM call), so failures should be rarer,
# but we keep retries for parity and resilience.
MAX_ATTEMPTS = 3


# Valid label fields, used by the validator.
LABEL_FIELDS = ("device_category", "vendor", "model")


# --- System prompt ---
#
# Concise, since Agent 2 has fewer responsibilities than Agent 1.
# GEPA will refine this during calibration. Key points:
#   - Define role: rule induction from anchored observations
#   - Define output schema
#   - Define generalizable_core: short, version-free, identifying substring
#   - Enforce strict_anchor_binding: every non-null label needs a keyword

SYSTEM_PROMPT = """You are the Rule Inducer for CoRID, a device-identification framework.

ROLE
----
You receive the structured output of Agent 1 (Information Extractor) —
anchored observations about a device's evidence bundle, tool findings,
and a current hypothesis. Your job is to turn this into a DRAFT RULE
that downstream PMI filtering and library-consistency checks (M4) will
process into a final, library-quality identification rule.

You have NO tools. You cannot call web_search, vocabulary lookups, or
anything else. You reason purely from Agent 1's output.

OUTPUT SCHEMA
-------------
Emit exactly one final assistant message containing only a JSON object
with this structure:

{
  "draft_rule": {
    "labels": {
      "device_category": "<canonical category>" | null,
      "vendor": "<canonical vendor>" | null,
      "model": "<canonical model>" | null
    },
    "confidence": "high" | "medium" | "low",
    "candidate_keywords": [
      {
        "source_field": "<field path from the bundle>",
        "anchor_token": "<the literal token from probe evidence>",
        "generalizable_core": "<the identifying substring of the anchor>",
        "supports_labels": ["<one or more of: device_category, vendor, model>"]
      },
      ...
    ]
  }
}

No prose around the JSON. No code fences. Just the JSON object.

WHAT EACH FIELD MEANS
---------------------

LABELS:
  Copy from Agent 1's `current_hypothesis`. You may abstain (set null)
  on any field where the supporting evidence is too weak, ambiguous, or
  unanchored. You may NOT change a non-null label to a different value;
  if you disagree with Agent 1's commit, your only options are to keep
  it or to null it out.

CONFIDENCE:
  Copy or lower Agent 1's confidence. Don't raise it.

CANDIDATE_KEYWORDS:
  For each label field you commit (non-null), include AT LEAST ONE
  keyword anchored to a probe-side token from Agent 1's bundle_analysis
  that supports it. Multiple keywords per label are fine — strong rules
  have multiple anchors.

  For each keyword:
    - `source_field`: the field path from the bundle where the anchor
      token appeared (e.g., "http.server", "cpe23", "data").
    - `anchor_token`: the literal token from Agent 1's observation.
      Use the same string Agent 1 cited in `anchored_to`.
    - `generalizable_core`: the substring of the anchor that's likely
      to generalize across instances of the same device. STRIP version
      numbers, dates, IPs, host-specific identifiers, and ports.
      Examples:
        "KM-MFP-http/V0.0.1"     -> "KM-MFP-http"
        "Apache/2.4.10 (Ubuntu)" -> "Apache" or "Apache (Ubuntu)" if Ubuntu is identifying
        "cpe:2.3:o:telesquare:tlr-2005ksh:..." -> "telesquare:tlr-2005ksh"
        "Westermo Lynx v4.28.4"  -> "Westermo Lynx"
      Keep the core SHORT — it should be the minimal identifying
      substring, not the whole anchor.
    - `supports_labels`: list of which labels this keyword anchors.
      Each entry must be one of: "device_category", "vendor", "model".

GROUNDING CONTRACT (HARD CONSTRAINT)
------------------------------------
Every non-null label MUST be supported by at least one keyword whose
supports_labels list includes that label. The validator checks this.
If a label cannot be anchored, set it to null instead.

This is the same grounding contract Agent 1 enforced. Your draft rule
inherits this requirement — labels cannot be made up; they must trace
back to probe-side tokens.

INPUT
-----
The user message will contain Agent 1's structured output as a JSON
object. Read its `bundle_analysis` to find anchored tokens, its
`current_hypothesis` for the label triple to consider committing,
and its `tool_findings` for additional context."""


def run_agent_2(agent1_output: dict, verbose: bool = False) -> dict[str, Any]:
    """Run Agent 2 on Agent 1's output, with bounded retries.

    Args:
      agent1_output: the parsed dict returned by run_agent_1. Should
        contain `bundle_analysis`, `current_hypothesis`, and
        `tool_findings`. May contain trace/warnings/error keys from
        Agent 1; those are ignored (we only consume the semantic parts).
      verbose: print progress per attempt.

    Returns:
      {
        "draft_rule": {...},
        "_attempts": <int>,
        "_trace": <list of messages from the last attempt>,
        "_warnings": [...]  // if any
      }

      On terminal failure (all attempts exhausted), returns the last
      attempt's output with an `_error` key.
    """
    # Edge case: Agent 1 produced no usable output (full abstention or error).
    # Don't waste an LLM call — emit an empty draft rule directly.
    if _is_empty_input(agent1_output):
        if verbose:
            print("Agent 1 output is empty or fully abstained. "
                  "Emitting empty draft rule without calling LLM.")
        return {
            "draft_rule": {
                "labels": {f: None for f in LABEL_FIELDS},
                "confidence": "low",
                "candidate_keywords": [],
            },
            "_attempts": 0,
            "_trace": [],
            "_warnings": ["agent 1 produced no usable output; draft rule empty"],
        }

    last_result: dict[str, Any] = {}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if verbose:
            print(f"=== Agent 2 attempt {attempt}/{MAX_ATTEMPTS} ===")
        last_result = _run_agent_2_once(agent1_output, verbose=verbose)
        if "_error" not in last_result:
            last_result["_attempts"] = attempt
            return last_result
        if verbose:
            print(f"  attempt {attempt} failed: {last_result.get('_error')}")

    last_result["_attempts"] = MAX_ATTEMPTS
    return last_result


def _is_empty_input(agent1_output: dict) -> bool:
    """Return True if Agent 1's output has nothing usable for rule induction.

    Two conditions count as "empty":
      - All three label fields in current_hypothesis are null.
      - bundle_analysis is empty (no observations to ground anything in).
    """
    hyp = agent1_output.get("current_hypothesis") or {}
    all_null = all(hyp.get(f) is None for f in LABEL_FIELDS)
    no_observations = not (agent1_output.get("bundle_analysis") or [])
    return all_null or no_observations


def _run_agent_2_once(agent1_output: dict, verbose: bool = False) -> dict[str, Any]:
    """Single LLM call: send Agent 1's output, parse the draft rule,
    validate against strict_anchor_binding."""
    # Build the user message: extract just the parts Agent 2 needs.
    # Strip Agent 1's bookkeeping fields (trace, warnings, attempts) so
    # they don't confuse the model.
    payload = {
        "bundle_analysis": agent1_output.get("bundle_analysis", []),
        "current_hypothesis": agent1_output.get("current_hypothesis", {}),
        "tool_findings": agent1_output.get("tool_findings", []),
    }

    user_message = (
        "Here is the structured output from Agent 1. Produce a draft rule "
        "per the rules in your system prompt.\n\n"
        f"AGENT 1 OUTPUT:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        reply_text = chat_text(
            messages=messages,
            model=MODEL_M3_AGENT_2,
            temperature=M3_TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
        )
    except Exception as e:
        return {
            "_error": f"LLM call failed: {type(e).__name__}: {e}",
            "_trace": messages,
        }

    if verbose:
        print(f"  reply ({len(reply_text)} chars): {reply_text[:200]}...")

    # Record the assistant turn in the trace
    messages.append({"role": "assistant", "content": reply_text})

    parsed = _parse_final_output(reply_text)
    if parsed is None:
        return {
            "_error": "agent 2 reply did not contain valid JSON",
            "_raw_content": reply_text,
            "_trace": messages,
        }

    # Enforce strict_anchor_binding on the draft rule.
    violation = _validate_anchor_binding(parsed)
    if violation is not None:
        return {
            "_error": f"strict_anchor_binding violation: {violation}",
            "_raw_output": parsed,
            "_trace": messages,
        }

    parsed["_trace"] = messages
    return parsed


# --- Validators ---

def _validate_anchor_binding(parsed_output: dict) -> str | None:
    """Enforce strict_anchor_binding on Agent 2's draft rule.

    Every non-null label in `draft_rule.labels` must have at least one
    entry in `draft_rule.candidate_keywords` whose `supports_labels`
    list includes that label name.

    Returns:
      A human-readable violation string if the contract is broken,
      or None if all non-null labels are anchored.
    """
    draft = parsed_output.get("draft_rule")
    if not isinstance(draft, dict):
        return "missing or malformed `draft_rule` object"

    labels = draft.get("labels") or {}
    if not isinstance(labels, dict):
        return "missing or malformed `draft_rule.labels` object"

    keywords = draft.get("candidate_keywords") or []
    if not isinstance(keywords, list):
        return "missing or malformed `draft_rule.candidate_keywords` list"

    # Build the set of labels that have at least one supporting keyword.
    supported: set[str] = set()
    for kw in keywords:
        if not isinstance(kw, dict):
            continue
        for lbl in kw.get("supports_labels") or []:
            if isinstance(lbl, str):
                supported.add(lbl)

    # Check each non-null label is supported.
    unanchored = []
    for field in LABEL_FIELDS:
        value = labels.get(field)
        if value is not None and field not in supported:
            unanchored.append(field)

    if unanchored:
        return (
            f"non-null label(s) {unanchored} have no supporting keyword "
            f"in candidate_keywords"
        )
    return None


# --- JSON parsing (same approach as Agent 1: balanced-brace extraction) ---

def _parse_final_output(content: str) -> dict | None:
    """Parse the assistant's reply as the expected JSON, tolerating
    preambles, code fences, and trailing text. Same approach as
    agent1_extractor._parse_final_output."""
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
    """Extract the first balanced JSON object substring from `s`."""
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


# --- Smoke test ---

if __name__ == "__main__":
    # Run Agent 1 then Agent 2 on the first calibration bundle.
    from loader import load_clean_records_list
    from normalization import normalize_record
    from agent1_extractor import run_agent_1

    records = load_clean_records_list()
    bundle = normalize_record(records[0])

    print(f"Running Agent 1 first on the bundle...")
    print(f"True labels (for reference): {bundle['labels']}")
    print()
    a1_output = run_agent_1(bundle, verbose=False)
    print(f"Agent 1 done. Hypothesis: {a1_output.get('current_hypothesis')}")
    print(f"  observations: {len(a1_output.get('bundle_analysis') or [])}")
    print(f"  warnings: {a1_output.get('_warnings', [])}")
    print()

    print("Running Agent 2 on Agent 1's output...")
    a2_output = run_agent_2(a1_output, verbose=True)
    print()
    print("=" * 70)
    print("AGENT 2 OUTPUT:")
    print("=" * 70)
    display = {k: v for k, v in a2_output.items() if k != "_trace"}
    print(json.dumps(display, ensure_ascii=False, indent=2))
