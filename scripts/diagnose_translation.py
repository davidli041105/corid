"""
Diagnose why the 5 manual translations didn't apply.

For each "still-Chinese" string in the merged dataset, check whether
the translation table has *exactly* that key, including bytewise
equality, Unicode normalization, and trailing-whitespace differences.

Usage:
    python scripts/diagnose_translation.py
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


MERGED_FILE = Path(__file__).resolve().parent.parent / "datasets" / "merged_translated.jsonl"
TABLE_FILE = Path(__file__).resolve().parent.parent / "datasets" / "translation_table.json"

CHINESE = re.compile(r"[\u4e00-\u9fff]")


def describe(s: str) -> str:
    """Human-readable summary of a string's bytes."""
    return f"len={len(s)}, repr={s!r}"


def main():
    # Collect still-Chinese device values from the merged data
    counter = Counter()
    with open(MERGED_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            v = r.get("device")
            if v and CHINESE.search(str(v)):
                counter[v] += 1

    # Load the current translation table
    with open(TABLE_FILE, encoding="utf-8") as f:
        table = json.load(f)
    table_keys = set(table.keys())

    print(f"Records with Chinese device values: {sum(counter.values())}")
    print(f"Unique still-Chinese device values: {len(counter)}")
    print(f"Translation table has {len(table_keys)} keys total.")
    print()

    for data_val, n_records in counter.most_common():
        print("=" * 75)
        print(f"DATA value (in {n_records} records):")
        print(f"  {describe(data_val)}")

        # Exact match?
        if data_val in table_keys:
            print(f"  -> EXACT MATCH in table; translates to: {table[data_val]!r}")
            print("     (so why didn't it apply?  re-running preprocess should have caught it)")
            continue

        print("  -> NOT in table as exact match.")
        print()

        # Look for near-matches: same NFC-normalized form
        normalized_data = unicodedata.normalize("NFC", data_val)
        normalized_matches = [
            k for k in table_keys
            if unicodedata.normalize("NFC", k) == normalized_data
        ]
        if normalized_matches:
            print("  But these table keys match after NFC normalization:")
            for k in normalized_matches:
                print(f"    table key: {describe(k)}")
                print(f"    diff in bytes: see hex dump below")
                _hex_dump_compare(data_val, k)
            continue

        # Look for keys that start the same way (prefix match) — to find
        # the typo or whitespace difference
        prefix_matches = [k for k in table_keys if k.startswith(data_val[:5])]
        if prefix_matches:
            print(f"  Table keys starting with the same first 5 chars ({data_val[:5]!r}):")
            for k in prefix_matches[:5]:
                print(f"    table key: {describe(k)}")
                # Show first ~10 chars hex of both for visual diff
                _hex_dump_compare(data_val, k)
        else:
            print("  No similar keys found. Probably never translated at all.")
        print()


def _hex_dump_compare(a: str, b: str):
    """Print bytewise hex side-by-side for the first 30 chars."""
    a_bytes = a.encode("utf-8")
    b_bytes = b.encode("utf-8")
    print(f"    data ({len(a_bytes)} bytes utf8): "
          f"{a_bytes[:60].hex(' ')}")
    print(f"    tbl  ({len(b_bytes)} bytes utf8): "
          f"{b_bytes[:60].hex(' ')}")


if __name__ == "__main__":
    main()
