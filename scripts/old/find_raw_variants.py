"""
Find every raw device value that contains '深信服' or '碧海云盒'.

The merged output has 5 specific strings still showing Chinese, but the
raw data has 0 records matching those exact strings — so the raw versions
must be slightly different (different separators, whitespace, etc.).

This script finds the actual raw forms so we can add them to the
translation table.

Usage:
    python scripts/find_raw_variants.py
"""

import json
from pathlib import Path
from collections import Counter

RAW_DIR = Path(__file__).resolve().parent.parent / "datasets" / "raw"
NEEDLES = ["深信服", "碧海云盒"]


def main():
    variants = Counter()

    files = sorted(RAW_DIR.glob("*.json"))
    for path in files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                v = r.get("device")
                if isinstance(v, str) and any(n in v for n in NEEDLES):
                    variants[v] += 1

    print(f"Found {len(variants)} unique raw `device` values containing 深信服 or 碧海云盒.")
    print(f"Total occurrences: {sum(variants.values())}\n")
    for v, c in variants.most_common():
        # Show repr so any weird whitespace is visible
        print(f"  {c:>5}  repr={v!r}")
        print(f"         hex bytes (first 40): {v.encode('utf-8')[:80].hex(' ')}")


if __name__ == "__main__":
    main()
