"""
Diagnostic: do model names/numbers appear in the evidence text of their
corresponding bundles?

If models are visible in evidence, M3 can anchor model → use web_search to
bridge model → vendor + category. Framework is viable on this dataset.

If models are also not visible, the dataset doesn't support text-based
device identification and the framework needs deeper rethinking.

Two visibility checks per model:
  - exact-match (case-insensitive)
  - token-match (any whitespace-separated word of the model name that's
    >= 3 chars, case-insensitive). Catches partial appearances like
    "TLR-2005KSH" → looks for "tlr-2005ksh" plus its parts.

Usage:
    python scripts/check_model_visibility.py
"""

import sys
import json
import re
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loader import load_clean_records_list
from normalization import normalize_record


def _evidence_text(bundle: dict) -> str:
    """Flatten all evidence string content into a single lowercase blob."""
    return json.dumps(bundle["evidence"], ensure_ascii=False).lower()


def _model_tokens(model: str) -> list[str]:
    """Decompose a model string into searchable sub-tokens.

    For 'Vigor ADSL router' -> ['vigor', 'adsl', 'router']
    For 'TLR-2005KSH'       -> ['tlr-2005ksh', 'tlr', '2005ksh']
    For 'HomePortal'        -> ['homeportal']

    Includes:
      - the full model string
      - whitespace-separated parts (length >= 3)
      - hyphen-separated sub-parts (length >= 3)
    """
    out = [model.lower()]
    for word in re.split(r"\s+", model.lower()):
        if len(word) >= 3:
            out.append(word)
            # Also split on hyphen for compound model numbers
            for piece in word.split("-"):
                if len(piece) >= 3 and piece not in out:
                    out.append(piece)
    # Deduplicate while preserving order
    seen = set()
    return [x for x in out if not (x in seen or seen.add(x))]


def main():
    records = load_clean_records_list()
    print(f"Loaded {len(records)} clean records.\n")

    # Group records by their (vendor, model) label so we can report per-model.
    by_model = {}
    for r in records:
        key = (r.get("vendor"), r.get("model"))
        by_model.setdefault(key, []).append(r)

    print(f"Unique (vendor, model) combinations: {len(by_model)}\n")

    # Aggregate stats
    total_bundles = 0
    bundles_with_full_match = 0
    bundles_with_partial_match = 0
    bundles_with_no_match = 0

    print(f"{'Model':<40} {'Vendor':<25} {'count':>5} {'full':>5} {'partial':>8} {'none':>5}")
    print("-" * 95)

    # Sort by largest model groups first
    sorted_keys = sorted(by_model.keys(), key=lambda k: -len(by_model[k]))

    for (vendor, model) in sorted_keys:
        bundles = by_model[(vendor, model)]
        if not model:
            continue
        n = len(bundles)
        model_lc = model.lower()
        tokens = _model_tokens(model)

        full = 0
        partial = 0
        for r in bundles:
            bundle = normalize_record(r)
            ev = _evidence_text(bundle)
            if model_lc in ev:
                full += 1
            elif any(t in ev for t in tokens if t != model_lc):
                partial += 1

        none = n - full - partial
        total_bundles += n
        bundles_with_full_match += full
        bundles_with_partial_match += partial
        bundles_with_no_match += none

        # Show all rows so we get a complete picture
        model_disp = model if len(model) <= 38 else model[:35] + "..."
        vendor_disp = (vendor or "")[:23]
        print(f"{model_disp:<40} {vendor_disp:<25} {n:>5} {full:>5} {partial:>8} {none:>5}")

    print("-" * 95)
    print(f"{'TOTAL':<66} {total_bundles:>5} "
          f"{bundles_with_full_match:>5} "
          f"{bundles_with_partial_match:>8} "
          f"{bundles_with_no_match:>5}")
    print()
    print(f"Coverage summary:")
    pct = lambda x: f"{100*x/total_bundles:.1f}%" if total_bundles else "—"
    print(f"  Full model match in evidence    : "
          f"{bundles_with_full_match} / {total_bundles}  ({pct(bundles_with_full_match)})")
    print(f"  Partial model token in evidence : "
          f"{bundles_with_partial_match} / {total_bundles}  ({pct(bundles_with_partial_match)})")
    print(f"  No match at all                 : "
          f"{bundles_with_no_match} / {total_bundles}  ({pct(bundles_with_no_match)})")
    print()
    print(f"Bundles where M3 could plausibly anchor on a model token: "
          f"{bundles_with_full_match + bundles_with_partial_match} / {total_bundles}  "
          f"({pct(bundles_with_full_match + bundles_with_partial_match)})")


if __name__ == "__main__":
    main()
