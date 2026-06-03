"""
Dataset loader for CoRID — Shodan schema.

Reads the merged Shodan dataset and filters to records suitable for
CoRID's pipeline: those with a complete 3-tier label
(`devicetype` + `vendor` + `model`).

The Shodan schema uses these field names for the labels:
  - `devicetype`  : device category (e.g., "router", "printer", "webcam")
  - `vendor`      : vendor name (e.g., "Telesquare", "Kyocera")
  - `model`       : exact model (e.g., "TLR-2005KSH")

No translation needed; Shodan data is already in English.

What this module gives you:
  - load_clean_records(): generator yielding clean record dicts
  - load_clean_records_list(): same data as a list
  - report_filtering(): walks the dataset and prints a filtering report

The loader does NOT do Bundle Normalization. Records come out as raw dicts
(same shape as the merged file). Normalization is a separate downstream step.
"""

import json
from pathlib import Path
from typing import Iterator
from collections import Counter


DATASET_FILE = Path(__file__).resolve().parent / "datasets" / "merged.jsonl"

# Required label fields — all must be populated for a record to be "clean".
REQUIRED_LABEL_FIELDS = ["devicetype", "vendor", "model"]


def _is_populated(val) -> bool:
    """A value counts as populated if it's a non-empty non-null string,
    or a non-empty list. Treats the literal string 'null' as missing."""
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
    for field in REQUIRED_LABEL_FIELDS:
        if not _is_populated(record.get(field)):
            return False
    return True


def load_clean_records(path: Path = DATASET_FILE) -> Iterator[dict]:
    """Generator: yield each clean record from the merged dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Run merge_data.py first."
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
    """Same as load_clean_records() but returns a list."""
    return list(load_clean_records(path))


def report_filtering(path: Path = DATASET_FILE) -> None:
    """Walk the merged dataset and report filter statistics."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    total = 0
    fail_missing_category = 0
    fail_missing_vendor = 0
    fail_missing_model = 0
    clean = 0
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

            if not _is_populated(r.get("devicetype")):
                fail_missing_category += 1
            if not _is_populated(r.get("vendor")):
                fail_missing_vendor += 1
            if not _is_populated(r.get("model")):
                fail_missing_model += 1

            if _is_clean(r):
                clean += 1
                category_counts[r["devicetype"]] += 1
                vendor_counts[r["vendor"]] += 1

    print("=" * 75)
    print("LOADER — filtering report")
    print("=" * 75)
    print(f"Total records in merged dataset:  {total}")
    if total:
        print(f"Clean records (pass all filters): {clean}  ({100*clean/total:.1f}%)")
    else:
        print("Clean records: 0  (dataset is empty)")
    print()
    print("Filter rejections (a record may fail multiple — counts don't sum):")
    print(f"  Missing devicetype : {fail_missing_category}")
    print(f"  Missing vendor     : {fail_missing_vendor}")
    print(f"  Missing model      : {fail_missing_model}")
    print()
    print(f"Among clean records:")
    print(f"  Unique device categories: {len(category_counts)}")
    print(f"  Unique vendors:           {len(vendor_counts)}")
    print()
    print("Category distribution:")
    for cat, n in category_counts.most_common():
        print(f"  {n:>5}  {cat}")
    print()
    print("Top 15 vendors:")
    for v, n in vendor_counts.most_common(15):
        print(f"  {n:>5}  {v}")


if __name__ == "__main__":
    report_filtering()
