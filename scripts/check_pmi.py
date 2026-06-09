"""
PMI sanity check — verify the store produces sensible PMI scores.

Compares a few tokens we expect to be discriminative (vendor names,
model fragments) against tokens we expect to be generic (HTTP scaffolding,
dates) across both PMI variants.

Usage:
    python scripts/check_pmi.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stats_store import StatsStore


def main():
    store_path = Path(__file__).resolve().parent.parent / "datasets" / "stats_store.json"
    store = StatsStore.load(store_path)
    print(f"Loaded store with N={store.N} bundles.\n")

    # --- Production-mode PMI (k, source_field) ---
    # We expect high PMI when a token is field-bound (appears mostly in one
    # specific field) and low PMI when it's promiscuous across many fields.
    print("=" * 70)
    print("PMI(k, source_field) — production-mode, label-free")
    print("=" * 70)

    # Tokens we expect to be FIELD-BOUND (high PMI):
    print("\nExpected HIGH PMI (field-bound discriminative tokens):")
    bound_probes = [
        ("kyocera", "http.html"),
        ("trendnet", "http.html"),
        ("lighttpd", "http.server"),
        ("draytek", "vendor"),  # if it appears in any evidence field
        ("vigor", "http.html"),
    ]
    for tok, field in bound_probes:
        pmi = store.pmi_kf(tok, field)
        ck = store.count_k.get(tok, 0)
        cf = store.count_f.get(field, 0)
        from stats_store import _key2
        ckf = store.count_kf.get(_key2(tok, field), 0)
        print(f"  PMI({tok!r}, {field!r}) = {pmi:+.3f}  "
              f"[count_k={ck}, count_f={cf}, count_kf={ckf}]")

    print("\nExpected LOW PMI (promiscuous generic tokens):")
    generic_probes = [
        ("tcp", "transport"),
        ("text/html", "http.html"),
        ("200", "data"),
        ("gmt", "data"),
        ("wed", "data"),
    ]
    for tok, field in generic_probes:
        pmi = store.pmi_kf(tok, field)
        ck = store.count_k.get(tok, 0)
        cf = store.count_f.get(field, 0)
        from stats_store import _key2
        ckf = store.count_kf.get(_key2(tok, field), 0)
        print(f"  PMI({tok!r}, {field!r}) = {pmi:+.3f}  "
              f"[count_k={ck}, count_f={cf}, count_kf={ckf}]")

    # --- Calibration-mode PMI (k, label) ---
    print()
    print("=" * 70)
    print("PMI(k, label) — calibration-mode, label-based")
    print("=" * 70)

    print("\nExpected HIGH PMI (vendor-discriminative tokens):")
    vendor_probes = [
        ("kyocera", "vendor", "Kyocera"),
        ("trendnet", "vendor", "TRENDnet"),
        ("draytek", "vendor", "DrayTek"),
    ]
    for tok, lf, lv in vendor_probes:
        pmi = store.pmi_klabel(tok, lf, lv)
        ck = store.count_k.get(tok, 0)
        from stats_store import _key2, _key3
        cl = store.count_label.get(_key2(lf, lv.lower()), 0)
        ckl = store.count_klabel.get(_key3(tok, lf, lv.lower()), 0)
        print(f"  PMI({tok!r}, {lf}={lv!r}) = {pmi:+.3f}  "
              f"[count_k={ck}, count_label={cl}, count_klabel={ckl}]")

    print("\nExpected LOW PMI (non-discriminative tokens):")
    nondisc_probes = [
        ("tcp", "vendor", "Kyocera"),
        ("200", "vendor", "TRENDnet"),
        ("text/html", "vendor", "Kyocera"),
    ]
    for tok, lf, lv in nondisc_probes:
        pmi = store.pmi_klabel(tok, lf, lv)
        ck = store.count_k.get(tok, 0)
        from stats_store import _key2, _key3
        cl = store.count_label.get(_key2(lf, lv.lower()), 0)
        ckl = store.count_klabel.get(_key3(tok, lf, lv.lower()), 0)
        print(f"  PMI({tok!r}, {lf}={lv!r}) = {pmi:+.3f}  "
              f"[count_k={ck}, count_label={cl}, count_klabel={ckl}]")


if __name__ == "__main__":
    main()
