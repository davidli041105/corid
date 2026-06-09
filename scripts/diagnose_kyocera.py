"""
Diagnostic: run Agent 1 on the Kyocera bundle with verbose=True to see
whether retries are actually firing on its failure mode.

If we see "=== Attempt 1/3 ===" through "=== Attempt 3/3 ===" in the
output, retries ARE firing but the model is failing deterministically.

If we only see "=== Attempt 1/3 ===" and nothing more, the retry loop
isn't running for some reason.

Usage:
    python scripts/diagnose_kyocera.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loader import load_clean_records_list
from normalization import normalize_record
from agent1_extractor import run_agent_1


def find_first(records, vendor, model):
    for r in records:
        if r.get("vendor") == vendor and r.get("model") == model:
            return r
    return None


def main():
    print("Loading clean records...")
    records = load_clean_records_list()

    record = find_first(records, "Kyocera", "Printer Panel")
    if record is None:
        print("No Kyocera/Printer Panel record found.")
        return

    bundle = normalize_record(record)

    print(f"Running Agent 1 with verbose=True...\n")
    output = run_agent_1(bundle, verbose=True)

    print()
    print("=" * 70)
    print("FINAL")
    print("=" * 70)
    out_no_trace = {k: v for k, v in output.items() if k != "_trace"}
    # Don't dump the full _raw_output either; just the headline
    if "_raw_output" in out_no_trace:
        out_no_trace["_raw_output"] = "(present, suppressed for brevity)"
    print(json.dumps(out_no_trace, ensure_ascii=False, indent=2)[:2000])

    print()
    print(f"_attempts field: {output.get('_attempts', 'NOT SET')}")
    print(f"has _error    : {'_error' in output}")


if __name__ == "__main__":
    main()
