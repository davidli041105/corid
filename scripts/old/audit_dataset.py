"""
Audit the merged dataset.

Walks every record in merged_translated.jsonl and reports:
  1. Population rates for the four label fields (device_type, device_type_sub,
     manufacturer, device)
  2. How often the labels are co-populated (e.g., manufacturer AND device_type_sub)
  3. The top values seen in each field — so we can spot OS strings masquerading
     as manufacturers, generic terms in `device`, etc.
  4. How often the banner / header / title fields are populated as evidence

Output is plain text, designed for reading and deciding what to filter.

Usage:
    python scripts/audit_dataset.py
"""

import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --- Paths ---

DATASET_FILE = Path(__file__).resolve().parent.parent / "datasets" / "merged_translated.jsonl"


# --- Fields of interest ---

LABEL_FIELDS = ["device_type", "device_type_sub", "manufacturer", "device", "product"]
EVIDENCE_FIELDS = ["banner", "header", "title", "server"]


def is_populated(val) -> bool:
    """A value counts as 'populated' if it's a non-empty, non-null string,
    OR a non-empty list. Treats empty strings and 'null' (literal string) as missing."""
    if val is None:
        return False
    if isinstance(val, str):
        s = val.strip()
        return bool(s) and s.lower() != "null"
    if isinstance(val, list):
        return len(val) > 0
    return True


def load_records() -> list[dict]:
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_FILE}")
    records = []
    with open(DATASET_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def report_population(records):
    """Section 1: how often each label field is populated."""
    print("=" * 75)
    print("SECTION 1 — Population rates")
    print("=" * 75)
    n = len(records)
    print(f"Total records: {n}\n")

    print("Label fields (the things we'd use as ground truth):")
    for field in LABEL_FIELDS:
        count = sum(1 for r in records if is_populated(r.get(field)))
        pct = 100 * count / n
        print(f"  {field:<18}: {count:>5} / {n}  ({pct:5.1f}%)")

    print("\nEvidence fields (sources M3 could read):")
    for field in EVIDENCE_FIELDS:
        count = sum(1 for r in records if is_populated(r.get(field)))
        pct = 100 * count / n
        print(f"  {field:<18}: {count:>5} / {n}  ({pct:5.1f}%)")


def report_cooccurrence(records):
    """Section 2: how labels co-occur. Reveals whether records that have one
    label tend to have all of them, or only a subset."""
    print("\n" + "=" * 75)
    print("SECTION 2 — Label co-occurrence")
    print("=" * 75)

    n = len(records)

    # Combinations we want to count
    combos = {
        "category + vendor (device_type_sub + manufacturer)":
            lambda r: is_populated(r.get("device_type_sub")) and is_populated(r.get("manufacturer")),
        "category + vendor + model (device_type_sub + manufacturer + device)":
            lambda r: is_populated(r.get("device_type_sub")) and is_populated(r.get("manufacturer")) and is_populated(r.get("device")),
        "vendor + model (manufacturer + device)":
            lambda r: is_populated(r.get("manufacturer")) and is_populated(r.get("device")),
        "all three populated (3-tier complete)":
            lambda r: is_populated(r.get("device_type_sub")) and is_populated(r.get("manufacturer")) and is_populated(r.get("device")),
        "fully unlabelled (no device, no manufacturer, no device_type_sub)":
            lambda r: not any([is_populated(r.get("device")), is_populated(r.get("manufacturer")), is_populated(r.get("device_type_sub"))]),
    }

    for name, predicate in combos.items():
        count = sum(1 for r in records if predicate(r))
        pct = 100 * count / n
        print(f"  {count:>5} / {n}  ({pct:5.1f}%)  {name}")


def report_top_values(records):
    """Section 3: top values in each label field. This is where OS-strings-as-
    manufacturer and generic terms in `device` will become visible."""
    print("\n" + "=" * 75)
    print("SECTION 3 — Top values per field")
    print("=" * 75)

    for field in LABEL_FIELDS:
        values = Counter()
        for r in records:
            v = r.get(field)
            if is_populated(v):
                values[v] += 1

        print(f"\n{field} — {len(values)} unique populated values, top 20:")
        for val, count in values.most_common(20):
            # Truncate long values for display
            display = val if len(val) <= 60 else val[:57] + "..."
            print(f"  {count:>5}  {display}")


def report_banner_lengths(records):
    """Section 4: how long are banner / header strings? Tells us how much
    evidence M3 has to work with on average."""
    print("\n" + "=" * 75)
    print("SECTION 4 — Evidence field lengths")
    print("=" * 75)

    for field in EVIDENCE_FIELDS:
        lengths = []
        for r in records:
            v = r.get(field)
            if is_populated(v) and isinstance(v, str):
                lengths.append(len(v))
        if not lengths:
            print(f"  {field}: no populated values")
            continue
        lengths.sort()
        n_pop = len(lengths)
        print(f"  {field}: {n_pop} populated, "
              f"min={lengths[0]}, "
              f"median={lengths[n_pop//2]}, "
              f"max={lengths[-1]}, "
              f"mean={sum(lengths)//n_pop}")


def main():
    print(f"Loading: {DATASET_FILE}")
    records = load_records()
    print(f"Loaded {len(records)} records.\n")

    report_population(records)
    report_cooccurrence(records)
    report_top_values(records)
    report_banner_lengths(records)


if __name__ == "__main__":
    main()
