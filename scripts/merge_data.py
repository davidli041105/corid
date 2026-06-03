"""
Merge per-category Shodan JSONL files into one dataset.

Reads every .json file in INPUT_DIR (each one a JSONL file representing
one device category) and concatenates them into a single output file.

No translation step — Shodan data is already English. This replaces the
previous preprocess_data.py which existed only for the now-discarded
DayDayMap dataset.

Usage:
    python scripts/merge_data.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


INPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "raw"
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "datasets" / "merged.jsonl"


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}. "
            "Place all Shodan .json files there (one file per category)."
        )

    files = sorted(INPUT_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files found in {INPUT_DIR}")

    print(f"Reading {len(files)} files from {INPUT_DIR}")
    total = 0
    per_file_counts = []
    for path in files:
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    # Validate JSON parses — but don't actually do anything with the
                    # parsed object yet, we'll write the raw line out for fidelity.
                    json.loads(line)
                    n += 1
                except json.JSONDecodeError as e:
                    print(f"  WARNING: bad JSON in {path.name}: {e}")
        per_file_counts.append((path.name, n))
        total += n
        print(f"  {path.name}: {n} records")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for path in files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    out.write(line + "\n")

    print()
    print(f"Wrote {total} records to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
