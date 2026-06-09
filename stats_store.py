"""
Statistics Store — §7.3 of CoRID.

The store is a passive counter accumulator. It runs after Bundle
Normalization on every bundle (regardless of whether the bundle reaches
M3 or just hits M2). Its only job is to maintain sufficient statistics
so PMI can be computed downstream.

The store NEVER drops tokens. Filtering happens at rule-induction time
(M4), not here. Even very generic tokens (like "Apache") stay in the
store with their full counts — because we need their counts to compute
the PMI threshold over the full distribution.

What the store tracks:

  N                    : total number of bundles processed
  count(f)             : per-field-path, how many bundles populated f
  count(k)             : per-token, how many bundles contain k anywhere
  count(k, f)          : per (token, field-path), co-occurrence count
  count(k, lf, lv)     : per (token, label-field, label-value), where
                         lf is one of {device_category, vendor, model}
                         and lv is the label value for that field.
                         Calibration-only; populated when labels are available.

Tokenization rules (per the design discussion):
  - Light punctuation: split on whitespace, ,;=()[]{}, but keep dots,
    hyphens, slashes, colons intact (so version strings and model
    numbers survive).
  - Case-insensitive: lowercase everything before storing.
  - Drop tokens of length 1.

Field paths:
  - Dotted paths into the evidence sub-dict (e.g., "http.server",
    "location.city", "cloud.provider"). Top-level fields are just
    their key (e.g., "data", "port", "isp").

Persistence:
  - The store serializes to / loads from datasets/stats_store.json.
  - JSON is simple and human-inspectable. For 150 records the file is
    small; if we scale to much larger datasets, we can switch to a
    binary format or SQLite.
  - Counter dicts use a string key joining the tuple components with a
    delimiter that won't appear in tokens or field paths, since JSON
    doesn't natively support tuple keys.

This module does NOT call any LLMs. Pure deterministic.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


# --- Tokenization ---

# Split on: whitespace, comma, semicolon, equals, parens, square/curly brackets,
# pipe, ampersand, quote chars, less-than/greater-than.
# Keep intact: dots, hyphens, underscores, slashes, colons.
_SPLIT_PATTERN = re.compile(r"[\s,;=()\[\]{}|&\"'<>]+")

MIN_TOKEN_LENGTH = 2


def tokenize(text: str) -> list[str]:
    """Split a string into tokens using the light-punctuation policy.

    Returns lowercased tokens with length >= MIN_TOKEN_LENGTH.
    Order is preserved; duplicates are kept (deduplication is the caller's
    job — for per-bundle co-occurrence we want set semantics, but the
    tokenizer itself just produces the stream).
    """
    if not isinstance(text, str) or not text:
        return []
    raw = _SPLIT_PATTERN.split(text)
    return [
        t.lower() for t in raw
        if t and len(t) >= MIN_TOKEN_LENGTH
    ]


# --- Walking the evidence dict ---

def iter_string_leaves(obj: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    """Recursively walk a value, yielding (field_path, string_value) for
    every string leaf encountered.

    Field paths use dotted notation: "http.server", "location.city", etc.
    List elements are not given indices; their parent path is used (so all
    elements of a list share the same field path).
    """
    if isinstance(obj, str):
        if obj:  # skip empty strings
            yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            child_prefix = f"{prefix}.{k}" if prefix else k
            yield from iter_string_leaves(v, child_prefix)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_string_leaves(item, prefix)
    # Other types (int, float, bool, None): not yielded.


# --- The store ---

# A JSON-safe delimiter for compound keys. We pick a sentinel that won't
# appear in either tokens (which are lowercased and punctuation-stripped)
# or in field paths (which are alphanumeric + dots).
_DELIM = "\x1f"  # ASCII Unit Separator, never in normal text


def _key2(a: str, b: str) -> str:
    return f"{a}{_DELIM}{b}"


def _key3(a: str, b: str, c: str) -> str:
    return f"{a}{_DELIM}{b}{_DELIM}{c}"


class StatsStore:
    """The statistics store. Accumulates counts incrementally.

    Public API:
      - update(bundle): incorporate one normalized bundle's counts
      - save(path) / load(path): persistence
      - PMI computation methods (compute later, from the stored counts)
    """

    # The three label-fields we track for PMI(k, label).
    LABEL_FIELDS = ("device_category", "vendor", "model")

    def __init__(self):
        self.N = 0
        self.count_f: dict[str, int] = defaultdict(int)              # field -> bundles populating it
        self.count_k: dict[str, int] = defaultdict(int)              # token -> bundles containing it
        self.count_kf: dict[str, int] = defaultdict(int)             # (token, field) -> count
        # (token, label_field, label_value) -> count
        self.count_klabel: dict[str, int] = defaultdict(int)
        # (label_field, label_value) -> count of bundles with that label
        self.count_label: dict[str, int] = defaultdict(int)

    # --- Update ---

    def update(self, normalized_bundle: dict) -> None:
        """Incorporate one normalized bundle into the running counts.

        normalized_bundle is the output of normalization.normalize_record:
            { "evidence": {...}, "labels": {...} }

        For the (k, label) counter, we read labels from the 'labels' part.
        If labels are missing (production mode), we just skip the label
        counter updates.
        """
        evidence = normalized_bundle.get("evidence", {})
        labels = normalized_bundle.get("labels", {}) or {}

        self.N += 1

        # Walk all string leaves, collecting (field, set_of_tokens_in_field).
        # We use sets per field within a bundle so that a token appearing
        # multiple times in the same field counts ONCE for count_kf — we
        # care about presence in field, not raw occurrences.
        tokens_in_field: dict[str, set[str]] = defaultdict(set)
        for field_path, text in iter_string_leaves(evidence):
            for tok in tokenize(text):
                tokens_in_field[field_path].add(tok)

        # Bundles populating each field
        for field_path in tokens_in_field:
            self.count_f[field_path] += 1

        # All distinct tokens in this bundle, across all fields
        all_tokens_in_bundle: set[str] = set()
        for toks in tokens_in_field.values():
            all_tokens_in_bundle |= toks

        # Per-token counts: each distinct token in the bundle contributes 1
        for tok in all_tokens_in_bundle:
            self.count_k[tok] += 1

        # (k, f) counts: each (token, field) pair contributes 1
        for field_path, toks in tokens_in_field.items():
            for tok in toks:
                self.count_kf[_key2(tok, field_path)] += 1

        # (k, label_field, label_value) counts — only when labels present
        for lf in self.LABEL_FIELDS:
            lv = labels.get(lf)
            if not lv:
                continue
            # We don't tokenize the label value — it's used verbatim as the
            # label. Lowercasing for consistency with token storage.
            lv_norm = str(lv).lower()
            # Count bundles per (label_field, label_value)
            self.count_label[_key2(lf, lv_norm)] += 1
            for tok in all_tokens_in_bundle:
                self.count_klabel[_key3(tok, lf, lv_norm)] += 1

    # --- Persistence ---

    def save(self, path: Path) -> None:
        """Serialize the store to JSON.

        We convert defaultdicts to plain dicts and store the four counters
        plus N at the top level. The delimiter inside compound keys is
        an ASCII Unit Separator character — JSON-safe (it'll be escaped
        as \u001f), unambiguous, and won't appear in real tokens.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "N": self.N,
            "count_f": dict(self.count_f),
            "count_k": dict(self.count_k),
            "count_kf": dict(self.count_kf),
            "count_klabel": dict(self.count_klabel),
            "count_label": dict(self.count_label),
            "_meta": {
                "delim": "U+001F (ASCII Unit Separator)",
                "format_version": 1,
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "StatsStore":
        """Load a store from disk."""
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        store = cls()
        store.N = payload["N"]
        store.count_f = defaultdict(int, payload["count_f"])
        store.count_k = defaultdict(int, payload["count_k"])
        store.count_kf = defaultdict(int, payload["count_kf"])
        store.count_klabel = defaultdict(int, payload["count_klabel"])
        store.count_label = defaultdict(int, payload.get("count_label", {}))
        return store

    # --- Convenience lookups ---

    def pmi_kf(self, k: str, f: str) -> float:
        """PMI(k, f) — production-mode label-free filter.

        Returns log of P(k, f) / (P(k) * P(f)). Returns -inf for impossible
        combinations (count of 0 anywhere in the formula).
        """
        import math
        if self.N == 0:
            return float("-inf")
        ck = self.count_k.get(k, 0)
        cf = self.count_f.get(f, 0)
        ckf = self.count_kf.get(_key2(k, f), 0)
        if ck == 0 or cf == 0 or ckf == 0:
            return float("-inf")
        # PMI = log( count(k,f) * N / (count(k) * count(f)) )
        return math.log((ckf * self.N) / (ck * cf))

    def pmi_klabel(self, k: str, label_field: str, label_value: str) -> float:
        """PMI(k, label) — calibration-mode label-based filter.

        label_field is one of LABEL_FIELDS. label_value is the specific
        value (will be lowercased to match storage).

        PMI = log( P(k, label) / (P(k) * P(label)) )
            = log( count(k, label) * N / (count(k) * count(label)) )

        Returns -inf for impossible combinations (any count being 0).
        """
        import math
        if self.N == 0:
            return float("-inf")
        lv_norm = str(label_value).lower()
        ck = self.count_k.get(k, 0)
        clabel = self.count_label.get(_key2(label_field, lv_norm), 0)
        cklabel = self.count_klabel.get(_key3(k, label_field, lv_norm), 0)
        if ck == 0 or clabel == 0 or cklabel == 0:
            return float("-inf")
        return math.log((cklabel * self.N) / (ck * clabel))

    # No outstanding TODOs — both PMI variants are correctly computable
    # from the maintained counters.


# --- Convenience: build the store from the loader's output ---

def build_store_from_records(normalized_bundles: list[dict]) -> StatsStore:
    """Build a fresh store by feeding it every normalized bundle.

    Caller is expected to normalize records first (via
    normalization.normalize_all).
    """
    store = StatsStore()
    for bundle in normalized_bundles:
        store.update(bundle)
    return store


# --- Top-level / debug ---

if __name__ == "__main__":
    # Build the store from the full 150-record sample and save it.
    from loader import load_clean_records_list
    from normalization import normalize_all

    print("Loading clean records...")
    records = load_clean_records_list()
    print(f"  {len(records)} records.")

    print("Normalizing...")
    bundles = normalize_all(records)

    print("Building statistics store...")
    store = build_store_from_records(bundles)

    print()
    print(f"Store contents:")
    print(f"  N (bundles)       : {store.N}")
    print(f"  Unique fields     : {len(store.count_f)}")
    print(f"  Unique tokens     : {len(store.count_k)}")
    print(f"  (token, field)    : {len(store.count_kf)}")
    print(f"  (token, label)    : {len(store.count_klabel)}")
    print(f"  Unique labels     : {len(store.count_label)}")

    # Show top 20 tokens by raw frequency
    print()
    print("Top 20 tokens by occurrence:")
    top_tokens = sorted(store.count_k.items(), key=lambda x: -x[1])[:20]
    for tok, count in top_tokens:
        print(f"  {count:>4}  {tok}")

    print()
    print("Top 10 fields by population:")
    top_fields = sorted(store.count_f.items(), key=lambda x: -x[1])[:10]
    for field, count in top_fields:
        print(f"  {count:>4}  {field}")

    # Save to disk
    out_path = Path(__file__).resolve().parent / "datasets" / "stats_store.json"
    store.save(out_path)
    print()
    print(f"Saved store to: {out_path}")
