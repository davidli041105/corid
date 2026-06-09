"""
Diagnostic: check whether vendor names actually appear in the evidence
text of their corresponding bundles.

If a vendor name shows up in 0 bundles' evidence, the dataset's labels
are not derivable from the probe-side text — CoRID's grounding contract
would force abstention.

Usage:
    python scripts/check_vendor_visibility.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loader import load_clean_records_list
from normalization import normalize_record


VENDORS_TO_CHECK = [
    "TRENDnet",
    "DrayTek",
    "Vigor",      # often appears as a model line for DrayTek
    "Kyocera",
    "Telesquare",
    "Microsoft",
    "Linksys",
]


def main():
    records = load_clean_records_list()
    print(f"Loaded {len(records)} clean records.\n")

    for needle in VENDORS_TO_CHECK:
        needle_lc = needle.lower()
        bundles_with_label = 0
        bundles_with_evidence_mention = 0

        for r in records:
            # How many bundles have this as their vendor label?
            if (r.get("vendor") or "").lower() == needle_lc:
                bundles_with_label += 1

            # How many bundles mention this string anywhere in evidence?
            bundle = normalize_record(r)
            ev_str = json.dumps(bundle["evidence"], ensure_ascii=False).lower()
            if needle_lc in ev_str:
                bundles_with_evidence_mention += 1

        print(f"  {needle:<12} : label in {bundles_with_label:>3} bundles | "
              f"appears in evidence text of {bundles_with_evidence_mention:>3} bundles")


if __name__ == "__main__":
    main()
