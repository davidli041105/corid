"""
Multi-bundle test for Agent 1.

Runs Agent 1 on a curated set of representative bundles to see how it
behaves across the difficulty spectrum:

  - Kyocera printer:    abundant direct evidence (easy)
  - Telesquare router:  vendor in CPE, model in banner (clean direct)
  - TRENDnet webcam:    vendor name NOT in evidence — web_search needed or abstain
  - DrayTek router:     Vigor model, vendor also not in evidence
  - GeoVision webcam:   less-famous vendor with partial evidence

Reports per record:
  - True labels (ground truth)
  - Agent 1's hypothesis (committed labels + confidence)
  - Number of bundle_analysis observations
  - Tools called (with name and a short summary)
  - Match status: did agent's labels match ground truth on each field?
  - Errors/issues if any

Usage:
    python scripts/test_agent1_multi.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loader import load_clean_records_list
from normalization import normalize_record
from agent1_extractor import run_agent_1


# Bundles to test, identified by (vendor, model) label match.
# We look for the first record that matches each spec.
TEST_TARGETS = [
    ("Kyocera",     "Printer Panel",          "Easy — direct evidence"),
    ("Telesquare",  "TLR-2005KSH",            "Direct — vendor in CPE"),
    ("TRENDnet",    "TV-IP110W",              "Hard — vendor invisible"),
    ("DrayTek",     "Vigor ADSL router",      "Hard — vendor invisible"),
    ("GeoVision",   "GeoHttpServer for webcams", "Medium — partial evidence"),
]


def find_record(records, target_vendor, target_model):
    """Find the first record matching the target vendor+model."""
    for r in records:
        if (r.get("vendor") == target_vendor and
            r.get("model") == target_model):
            return r
    return None


def label_match(committed, truth):
    """Compare a committed label against ground truth, case-insensitive.
    Returns 'match', 'mismatch', or 'abstain' (when committed is None)."""
    if committed is None:
        return "abstain"
    if not truth:
        return "no_truth"
    return "match" if str(committed).lower() == str(truth).lower() else "mismatch"


def summarize_output(output: dict) -> dict:
    """Pull just the salient fields from Agent 1's output for the report."""
    if "_error" in output:
        return {"error": output["_error"]}

    hypothesis = output.get("current_hypothesis", {})
    bundle_analysis = output.get("bundle_analysis", [])
    tool_findings = output.get("tool_findings", [])

    return {
        "hypothesis": hypothesis,
        "n_observations": len(bundle_analysis),
        "tools_called": [
            f"{tf.get('tool')}: {tf.get('summary', '')[:80]}"
            for tf in tool_findings
        ],
    }


def print_report(idx, target, record, output):
    """Pretty-print one test result."""
    vendor, model, note = target
    print("=" * 75)
    print(f"TEST {idx + 1}: {vendor} / {model}")
    print(f"  ({note})")
    print("=" * 75)

    if record is None:
        print("  NO MATCHING RECORD FOUND IN DATASET.")
        return

    truth = {
        "device_category": record.get("devicetype"),
        "vendor": record.get("vendor"),
        "model": record.get("model"),
    }
    print(f"  True labels:")
    for k, v in truth.items():
        print(f"    {k:<18}: {v!r}")

    summary = summarize_output(output)
    if "error" in summary:
        print(f"  ERROR: {summary['error']}")
        return

    print()
    print(f"  Agent hypothesis:")
    hyp = summary["hypothesis"]
    for k in ["device_category", "vendor", "model", "confidence"]:
        print(f"    {k:<18}: {hyp.get(k)!r}")

    print()
    print(f"  Match status:")
    for k in ["device_category", "vendor", "model"]:
        status = label_match(hyp.get(k), truth.get(k))
        marker = {"match": "✓", "mismatch": "✗", "abstain": "○", "no_truth": "·"}[status]
        print(f"    {k:<18}: {marker} {status}")

    print()
    print(f"  Observations emitted : {summary['n_observations']}")
    print(f"  Tools called         : {len(summary['tools_called'])}")
    for t in summary["tools_called"]:
        print(f"    - {t}")
    print()


def main():
    print("Loading clean records...")
    records = load_clean_records_list()
    print(f"  {len(records)} records available.\n")

    results = []
    for idx, target in enumerate(TEST_TARGETS):
        vendor, model, note = target
        record = find_record(records, vendor, model)
        if record is None:
            print_report(idx, target, None, None)
            continue

        print(f"Running Agent 1 on {vendor} / {model}...", flush=True)
        bundle = normalize_record(record)
        output = run_agent_1(bundle, verbose=False)
        print_report(idx, target, record, output)
        results.append((target, record, output))

    # --- Aggregate summary ---
    print("=" * 75)
    print("AGGREGATE")
    print("=" * 75)
    counts = {"match": 0, "mismatch": 0, "abstain": 0, "no_truth": 0, "error": 0}
    total_fields = 0
    for target, record, output in results:
        if "_error" in output:
            counts["error"] += 1
            continue
        hyp = output.get("current_hypothesis", {})
        truth = {
            "device_category": record.get("devicetype"),
            "vendor": record.get("vendor"),
            "model": record.get("model"),
        }
        for k in truth:
            status = label_match(hyp.get(k), truth.get(k))
            counts[status] += 1
            total_fields += 1
    print(f"Total fields evaluated: {total_fields}")
    for k, v in counts.items():
        if v > 0:
            pct = 100 * v / total_fields if total_fields else 0
            print(f"  {k:<10}: {v}  ({pct:.0f}%)")


if __name__ == "__main__":
    main()
