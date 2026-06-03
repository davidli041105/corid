"""
Dataset loader for CoRID.

Reads the merged + translated dataset and filters to records suitable for
CoRID's pipeline: those with a complete 3-tier label (device_type_sub +
manufacturer + device) and excluding placeholder values (`Other`, `Other
Network Devices`) which are auto-fallback labels rather than real
categorizations.

What this module gives you:

  - load_clean_records(): generator yielding clean record dicts. Use this
    when you want to iterate without holding everything in memory.

  - load_clean_records_list(): same data as a list. Convenient for small
    sample sizes; uses more memory.

  - report_filtering(): walks the dataset once and prints how many records
    survive each filter step. Useful for understanding the data before
    sampling.

The loader does NOT do Bundle Normalization. Records come out of here as
raw dicts (same shape as the merged file). Normalization is a separate
downstream step.
"""

import json
from pathlib import Path
from typing import Iterator
from collections import Counter


DATASET_FILE = Path(__file__).resolve().parent / "datasets" / "merged_translated.jsonl"


# --- Filtering policy (locked in based on yesterday's audit) ---

# Required label fields — all must be populated for a record to be "clean".
REQUIRED_LABEL_FIELDS = ["device_type_sub", "manufacturer", "device"]

# Placeholder values to exclude. These are auto-fallback labels that
# don't represent real categorizations.
EXCLUDED_VENDORS = {"Other"}
EXCLUDED_CATEGORIES = {"Other Network Devices"}


def _is_populated(val) -> bool:
    """A value counts as populated if it's a non-empty non-null string,
    or a non-empty list. Treats the literal string 'null' as missing
    (this appears in the source data)."""
    if val is None:
        return False
    if isinstance(val, str):
        s = val.strip()
        return bool(s) and s.lower() != "null"
    if isinstance(val, list):
        return len(val) > 0
    return True


def _is_clean(record: dict) -> bool:
    """Apply the cleanliness filter to a single record."""
    # All three label fields must be populated
    for field in REQUIRED_LABEL_FIELDS:
        if not _is_populated(record.get(field)):
            return False
    # Vendor is not a placeholder
    if record.get("manufacturer") in EXCLUDED_VENDORS:
        return False
    # Category is not a placeholder
    if record.get("device_type_sub") in EXCLUDED_CATEGORIES:
        return False
    return True


def load_clean_records(path: Path = DATASET_FILE) -> Iterator[dict]:
    """Generator: yield each clean record from the merged dataset.

    Use this for streaming iteration when you don't want to hold the
    whole dataset in memory. ~471K records of mixed-size JSON would
    typically fit in memory comfortably, but a generator keeps options
    open for larger datasets later.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Run preprocess_data.py first to generate it."
        )

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _is_clean(record):
                yield record


def load_clean_records_list(path: Path = DATASET_FILE) -> list[dict]:
    """Same as load_clean_records() but returns a list. Convenient when
    you want random access or len()."""
    return list(load_clean_records(path))


def report_filtering(path: Path = DATASET_FILE) -> None:
    """Walk the merged dataset once and report how many records survive
    each filter step. Output is human-readable, designed for printing
    to stdout."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    total = 0
    fail_missing_category = 0
    fail_missing_vendor = 0
    fail_missing_device = 0
    fail_other_vendor = 0
    fail_other_category = 0
    clean = 0

    # Track category distribution among clean records
    category_counts = Counter()
    vendor_counts = Counter()

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1

            # Check each filter in turn — count records that fail at each step.
            # A single record can fail multiple checks; we count them in all
            # the buckets they trigger, so totals don't sum to `total`.
            if not _is_populated(r.get("device_type_sub")):
                fail_missing_category += 1
            elif r.get("device_type_sub") in EXCLUDED_CATEGORIES:
                fail_other_category += 1
            if not _is_populated(r.get("manufacturer")):
                fail_missing_vendor += 1
            elif r.get("manufacturer") in EXCLUDED_VENDORS:
                fail_other_vendor += 1
            if not _is_populated(r.get("device")):
                fail_missing_device += 1

            if _is_clean(r):
                clean += 1
                category_counts[r["device_type_sub"]] += 1
                vendor_counts[r["manufacturer"]] += 1

    print("=" * 75)
    print("LOADER — filtering report")
    print("=" * 75)
    print(f"Total records in merged dataset:  {total}")
    print(f"Clean records (pass all filters): {clean}  ({100*clean/total:.1f}%)")
    print()
    print("Filter rejections (a record may fail multiple — counts don't sum to total):")
    print(f"  Missing device_type_sub : {fail_missing_category}")
    print(f"  Missing manufacturer    : {fail_missing_vendor}")
    print(f"  Missing device          : {fail_missing_device}")
    print(f"  Manufacturer == 'Other' : {fail_other_vendor}")
    print(f"  Category == 'Other ...' : {fail_other_category}")
    print()
    print(f"Among clean records:")
    print(f"  Unique categories: {len(category_counts)}")
    print(f"  Unique vendors:    {len(vendor_counts)}")
    print()
    print("Top 15 categories by count:")
    for cat, n in category_counts.most_common(15):
        print(f"  {n:>6}  {cat}")


if __name__ == "__main__":
    # Convenience: running this module directly prints the filtering report.
    report_filtering()
