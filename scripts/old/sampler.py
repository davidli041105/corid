"""
Stratified sampler for CoRID's initial 500-record sample.

Strategy: top-N proportional (strategy 'a' from the design discussion).
  - Identify the top N categories by corpus frequency
  - Compute each category's share of SAMPLE_SIZE in proportion to its
    corpus frequency, using the largest-remainder method for deterministic
    rounding
  - Within each category, randomly select that many records with a fixed seed

Reproducibility:
  - Same seed -> exactly the same 500 records every run
  - Default seed is 42; can be overridden in build_sample()

What this module gives you:

  - build_sample(): returns the list of 500 sampled record dicts in a stable
    order (sorted by category then IP for tie-breaking)
  - report_sample(): prints the per-category breakdown of an existing sample
  - save_sample() / load_sample(): persist the sample to a JSONL file so
    downstream pipeline steps can reuse it without re-running the sampler

The sampler does NOT do Bundle Normalization. Records come out raw, same
shape as the loader's output.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from loader import load_clean_records_list


# --- Configuration ---

SAMPLE_SIZE = 500
TOP_N_CATEGORIES = 20
DEFAULT_SEED = 42

SAMPLE_FILE = Path(__file__).resolve().parent / "datasets" / "sample_500.jsonl"


# --- Quota computation (largest-remainder method) ---

def _compute_quotas(
    category_corpus_counts: Counter,
    top_n: int,
    sample_size: int,
) -> dict[str, int]:
    """Decide how many records each category gets in the sample.

    Procedure:
      1. Pick the top_n categories by corpus frequency.
      2. Compute each category's IDEAL share: (cat_count / total_top_n_count) * sample_size.
         This is a fractional number.
      3. Each category gets floor(ideal_share) records.
      4. Remaining slots (sample_size - sum_of_floors) go to the categories
         with the largest fractional remainders. This is the "largest
         remainder method" — produces consistent results with no bias.

    Why this method:
      - Each category gets at least its proportional share, rounded down.
      - The unallocated remainder is distributed to who "deserves" it most,
        not arbitrarily.
      - Deterministic given the input counts: no randomness in quota
        assignment, only in which records get picked within each quota.

    Returns:
      Dict mapping category name -> number of records to sample from it.
    """
    top_cats = [c for c, _ in category_corpus_counts.most_common(top_n)]
    top_total = sum(category_corpus_counts[c] for c in top_cats)

    # Ideal fractional shares
    shares = {c: (category_corpus_counts[c] / top_total) * sample_size for c in top_cats}

    # Floor each share — that's the guaranteed allocation
    quotas = {c: int(shares[c]) for c in top_cats}
    allocated = sum(quotas.values())
    remainder = sample_size - allocated

    # Distribute the remainder by largest fractional remainder
    fractions = [(c, shares[c] - quotas[c]) for c in top_cats]
    fractions.sort(key=lambda x: -x[1])
    for i in range(remainder):
        # Should always have enough categories to receive the remainder,
        # since we have top_n >> remainder.
        cat = fractions[i][0]
        quotas[cat] += 1

    return quotas


# --- Sampling ---

def build_sample(seed: int = DEFAULT_SEED) -> list[dict]:
    """Build the 500-record sample.

    Steps:
      1. Load all clean records from the merged dataset (via the loader).
      2. Count category frequencies in the corpus.
      3. Compute per-category quotas using largest-remainder method.
      4. Within each top-N category, randomly select `quota` records using a
         seeded RNG.
      5. Return the union, sorted by (category, ip) for stable output order.
    """
    rng = random.Random(seed)

    # 1. Load
    print("Loading clean records...")
    records = load_clean_records_list()
    print(f"  {len(records)} clean records available.")

    # 2. Count categories
    category_counts = Counter(r["device_type_sub"] for r in records)

    # 3. Quotas
    quotas = _compute_quotas(category_counts, TOP_N_CATEGORIES, SAMPLE_SIZE)
    sum_quotas = sum(quotas.values())
    assert sum_quotas == SAMPLE_SIZE, f"Quota sum {sum_quotas} != {SAMPLE_SIZE}"

    # 4. Group records by category, then sample within each quota'd category.
    #    Only categories in `quotas` are considered.
    by_category = defaultdict(list)
    target_cats = set(quotas.keys())
    for r in records:
        cat = r["device_type_sub"]
        if cat in target_cats:
            by_category[cat].append(r)

    sample = []
    for cat in sorted(quotas.keys()):  # sort cats for deterministic iteration
        candidates = by_category[cat]
        n_take = quotas[cat]
        if n_take > len(candidates):
            # Shouldn't happen given the top-N is the biggest categories,
            # but guard anyway.
            print(f"  WARNING: category {cat!r} wants {n_take} but only "
                  f"has {len(candidates)}; taking all.")
            n_take = len(candidates)
        # rng.sample is order-stable given the same seed and input
        # (which is itself ordered by the merged-file order, deterministic
        # as long as the upstream preprocessing produces a stable file).
        chosen = rng.sample(candidates, n_take)
        sample.extend(chosen)

    # 5. Sort by (category, ip) for stable downstream order.
    sample.sort(key=lambda r: (r["device_type_sub"], r.get("ip", "")))

    return sample


# --- Reporting ---

def report_sample(sample: list[dict]) -> None:
    """Print a per-category breakdown of the sample."""
    print("=" * 75)
    print(f"SAMPLE REPORT — {len(sample)} records")
    print("=" * 75)

    cat_counts = Counter(r["device_type_sub"] for r in sample)
    vendor_counts = Counter(r["manufacturer"] for r in sample)

    print(f"Categories represented: {len(cat_counts)}")
    print(f"Unique vendors:         {len(vendor_counts)}")
    print()
    print("Per-category counts:")
    for cat, n in cat_counts.most_common():
        print(f"  {n:>4}  {cat}")


# --- Persistence ---

def save_sample(sample: list[dict], path: Path = SAMPLE_FILE) -> None:
    """Write the sample to JSONL so downstream steps can reuse it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(sample)} records to: {path}")


def load_sample(path: Path = SAMPLE_FILE) -> list[dict]:
    """Read back a previously-saved sample."""
    if not path.exists():
        raise FileNotFoundError(
            f"Sample file not found: {path}. Run build_sample() first."
        )
    sample = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sample.append(json.loads(line))
    return sample


# --- Main: build + report + save in one go ---

if __name__ == "__main__":
    sample = build_sample()
    report_sample(sample)
    print()
    save_sample(sample)
