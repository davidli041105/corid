"""
Check whether the 5 still-Chinese device values exist in the raw input
files (datasets/raw/*.json), and if so how exactly they're stored.

This will tell us whether the bug is in:
  - The translation: raw has different bytes than what we think
  - Or somewhere else in the pipeline

Usage:
    python scripts/check_raw_values.py
"""

import json
from pathlib import Path
from collections import Counter

RAW_DIR = Path(__file__).resolve().parent.parent / "datasets" / "raw"

TARGETS = {
    "深信服 Application Delivery Management System",
    "深信服 Firewall",
    "深信服 Full Traffic Threat Analysis System",
    "深信服 SD-WAN Secure Intelligent Router",
    "碧海云盒 Router",
}


def main():
    found = Counter()
    files_with_targets = {}

    files = sorted(RAW_DIR.glob("*.json"))
    print(f"Scanning {len(files)} raw files...\n")

    for path in files:
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                v = r.get("device")
                if v in TARGETS:
                    found[v] += 1
                    files_with_targets.setdefault(path.name, []).append((line_no, v))

    print("Results:")
    for t in TARGETS:
        print(f"  {found[t]:>5} records with device == {t!r}")

    print()
    print("Files containing target strings:")
    for fname, hits in files_with_targets.items():
        print(f"  {fname}: {len(hits)} hits")
        # show first 2 sample line numbers
        for line_no, v in hits[:2]:
            print(f"    line {line_no}: {v!r}")
        if len(hits) > 2:
            print(f"    ... and {len(hits)-2} more")


if __name__ == "__main__":
    main()
