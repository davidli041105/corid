"""
One-shot sanity check: are the 5 manually-added Sangfor / BlueCloud
entries actually present in translation_table.json right now?

Usage:
    python scripts/check_table.py
"""

import json
from pathlib import Path

TABLE_FILE = Path(__file__).resolve().parent.parent / "datasets" / "translation_table.json"

KEYS_TO_CHECK = [
    "深信服 Application Delivery Management System",
    "深信服 Firewall",
    "深信服 Full Traffic Threat Analysis System",
    "深信服 SD-WAN Secure Intelligent Router",
    "碧海云盒 Router",
]


def main():
    with open(TABLE_FILE, encoding="utf-8") as f:
        table = json.load(f)

    print(f"Table has {len(table)} keys total.\n")
    for k in KEYS_TO_CHECK:
        v = table.get(k)
        status = "IN TABLE" if v is not None else "NOT FOUND"
        print(f"  [{status}]  {k!r}")
        if v is not None:
            print(f"             -> {v!r}")


if __name__ == "__main__":
    main()
