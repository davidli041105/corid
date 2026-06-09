"""
Diagnostic: Agent 1 on the GeoVision/GeoHttpServer bundle, with full visibility.

Goal: figure out why Agent 1 skipped lookup_vendor, lookup_device_category,
AND web_search yet still committed non-null labels (a mandatory-lookup-rule
violation).

Three things this script does:
  1. Print the full normalized bundle Agent 1 sees, so we know what
     evidence was on the table.
  2. Run Agent 1 with verbose=True, so every turn / tool call / response
     gets printed inline.
  3. Print the full message trace at the end so we can read exactly
     what was sent to and from the model.

Usage:
    python scripts/diagnose_geovision.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loader import load_clean_records_list
from normalization import normalize_record
from agent1_extractor import run_agent_1


TARGET_VENDOR = "GeoVision"
TARGET_MODEL = "GeoHttpServer for webcams"


def find_first(records, vendor, model):
    for r in records:
        if r.get("vendor") == vendor and r.get("model") == model:
            return r
    return None


def main():
    print("Loading clean records...")
    records = load_clean_records_list()
    print(f"  {len(records)} clean records.\n")

    record = find_first(records, TARGET_VENDOR, TARGET_MODEL)
    if record is None:
        print(f"No record found for {TARGET_VENDOR} / {TARGET_MODEL}.")
        return

    bundle = normalize_record(record)

    print("=" * 75)
    print("FULL NORMALIZED BUNDLE (evidence portion):")
    print("=" * 75)
    print(json.dumps(bundle["evidence"], ensure_ascii=False, indent=2))
    print()
    print("=" * 75)
    print("Ground truth labels (Agent 1 never sees these):")
    print("=" * 75)
    print(json.dumps(bundle["labels"], ensure_ascii=False, indent=2))
    print()

    print("=" * 75)
    print("RUNNING Agent 1 (verbose=True)")
    print("=" * 75)
    output = run_agent_1(bundle, verbose=True)

    print()
    print("=" * 75)
    print("FINAL OUTPUT")
    print("=" * 75)
    output_no_trace = {k: v for k, v in output.items() if k != "_trace"}
    print(json.dumps(output_no_trace, ensure_ascii=False, indent=2))

    print()
    print("=" * 75)
    print("FULL MESSAGE TRACE")
    print("=" * 75)
    trace = output.get("_trace", [])
    print(f"({len(trace)} messages in trace)")
    for i, msg in enumerate(trace):
        role = msg.get("role", "?")
        print()
        print(f"--- Message {i + 1} (role={role}) ---")
        if role == "system":
            content = msg.get("content", "")
            print(f"[system prompt, {len(content)} chars; not shown in full]")
        elif role == "user":
            content = msg.get("content", "")
            # User messages may be long (the bundle). Show first 400 chars.
            if len(content) > 800:
                print(content[:400])
                print(f"  ... [{len(content)-400} more chars]")
            else:
                print(content)
        elif role == "assistant":
            content = msg.get("content")
            tool_calls = msg.get("tool_calls", [])
            if content:
                print(f"content: {content[:1000]}")
                if len(content) > 1000:
                    print(f"  ... [{len(content)-1000} more chars]")
            if tool_calls:
                print(f"tool_calls ({len(tool_calls)}):")
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    print(f"  - {fn.get('name')}({fn.get('arguments')})")
        elif role == "tool":
            name = msg.get("name", "?")
            content = msg.get("content", "")
            if len(content) > 500:
                print(f"tool={name}, content (first 500 chars):")
                print(content[:500])
                print(f"  ... [{len(content)-500} more chars]")
            else:
                print(f"tool={name}, content:")
                print(content)


if __name__ == "__main__":
    main()
