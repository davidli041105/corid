"""
DayDayMap dataset translator and merger.

What this script does, in one pass over the data:
  1. Reads all .json files in INPUT_DIR (one JSONL file per device category).
  2. Collects every unique Chinese string that appears in the categorical
     fields we care about (device_type, device_type_sub, manufacturer,
     device, product, etc.).
  3. Translates them to English via DeepSeek and saves the result to
     translation_table.json so future runs can reuse it (cache).
  4. Applies the translations to every record, then writes the merged,
     translated dataset to a single output file (one JSON object per line).

Modes:
  Default        : full run — collect, translate, merge, write output
  --dry-run      : only count and sample unique Chinese strings per field.
                   Makes NO API calls. Writes no output. For cost estimation.

Why a translation table file rather than translating on the fly?
  - One LLM call per unique string instead of per record (cheap, fast)
  - Auditable: you can open translation_table.json and see/correct any
    translation before it propagates through the pipeline
  - Reproducible: re-running the script gives identical results

Usage:
    python preprocess_data.py             # full run
    python preprocess_data.py --dry-run   # count only, no API calls
"""

import json
import re
import sys
import argparse
from pathlib import Path
from typing import Iterable
from collections import Counter

# This script imports from your project root, so make sure it's runnable
# from the project root (where llm_client.py and config.py live).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# We only import the LLM client in non-dry-run mode, so that --dry-run
# works without needing a DEEPSEEK_API_KEY set. Lazy import inside main().


# --- Paths (adjust if your layout differs) ---

INPUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "raw"
OUTPUT_FILE = Path(__file__).resolve().parent.parent / "datasets" / "merged_translated.jsonl"
TRANSLATION_TABLE_FILE = Path(__file__).resolve().parent.parent / "datasets" / "translation_table.json"


# --- Configuration ---

# The fields we want to translate. Other fields (banner, server, ip, etc.)
# stay as-is because they're either evidence (M3 reads them verbatim) or
# already English/numeric.
#
# `device` and `product` added 2025-06: the audit showed Chinese strings
# remaining in these fields (e.g., "海康威视视频监控", "PC终端", "国产化品牌"
# inside compound product strings).
TRANSLATABLE_FIELDS = [
    "device_type",
    "device_type_sub",
    "manufacturer",
    "company",
    "industry",
    "device",
]

# Pattern to detect "contains Chinese characters". Range U+4E00–U+9FFF covers
# CJK Unified Ideographs (the bulk of common Chinese).
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def contains_chinese(s: str) -> bool:
    """True if the string has any CJK characters in it."""
    if not isinstance(s, str):
        return False
    return bool(CHINESE_PATTERN.search(s))


def collect_chinese_strings(records: Iterable[dict]) -> dict[str, Counter]:
    """Walk all records and return, per field, a Counter mapping each unique
    Chinese-containing string to its occurrence count.

    Returns a dict keyed by field name; each value is a Counter.
    """
    per_field = {f: Counter() for f in TRANSLATABLE_FIELDS}
    for r in records:
        for field in TRANSLATABLE_FIELDS:
            val = r.get(field)
            if val and contains_chinese(val):
                per_field[field][val] += 1
    return per_field


def translate_one(chinese: str, chat_fn) -> str:
    """Translate a single Chinese string to a canonical English form via DeepSeek.

    We deliberately ask for a *canonical, concise* form rather than a literal
    word-for-word translation, because these strings will become vocabulary
    entries — e.g., "工业控制设备" should become "Industrial Control Equipment",
    not "industry control device" or some longer paraphrase.

    For compound strings (English mixed with Chinese, comma-separated lists),
    we ask the model to translate only the Chinese parts and leave English
    tokens unchanged.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You translate Chinese technical terms used in IoT/network "
                "device cataloging into concise canonical English. "
                "Rules:\n"
                "- Reply with ONLY the translation — no quotes, no explanation.\n"
                "- Use title case for noun phrases (e.g., 'Industrial Switch').\n"
                "- Keep it short.\n"
                "- If the input has English words or commas, preserve them "
                "exactly and only translate the Chinese parts. Preserve the "
                "comma structure of compound strings."
            ),
        },
        {
            "role": "user",
            "content": f"Translate: {chinese}",
        },
    ]
    reply = chat_fn(messages, max_tokens=200, temperature=0.0)
    # Strip whitespace and any stray quotes the model might add
    return reply.strip().strip('"\'`')


def load_existing_table() -> dict[str, str]:
    """Load translation_table.json if it exists, else return empty dict."""
    if TRANSLATION_TABLE_FILE.exists():
        with open(TRANSLATION_TABLE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_table(table: dict[str, str]) -> None:
    """Write the translation table back to disk in a stable, readable form."""
    TRANSLATION_TABLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRANSLATION_TABLE_FILE, "w", encoding="utf-8") as f:
        # sort_keys for stable diffs across runs;
        # ensure_ascii=False so Chinese stays readable in the file
        json.dump(table, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_all_records() -> list[dict]:
    """Read every .json file in INPUT_DIR as JSONL and concatenate."""
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}. "
            "Place all DayDayMap .json files there."
        )

    all_records = []
    files = sorted(INPUT_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files found in {INPUT_DIR}")

    for path in files:
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    all_records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"WARNING: bad JSON in {path.name} line {line_no}: {e}")
        print(f"  Loaded {path.name}")
    return all_records


def apply_translation(records: list[dict], table: dict[str, str]) -> list[dict]:
    """Return a new list of records with translatable fields replaced by
    their English forms (looked up in `table`). Non-Chinese fields and
    fields not in `table` pass through unchanged.

    We don't mutate the input records — return fresh dicts so the original
    data is preserved if anything goes wrong mid-pipeline.
    """
    out = []
    for r in records:
        new_r = dict(r)  # shallow copy
        for field in TRANSLATABLE_FIELDS:
            val = new_r.get(field)
            if val and val in table:
                new_r[field] = table[val]
        out.append(new_r)
    return out


# --- Dry-run reporting ---

def dry_run_report(per_field: dict[str, Counter], existing_table: dict[str, str]) -> None:
    """Print what a real run would do, without making any API calls."""
    print("=" * 75)
    print("DRY RUN — what a real run would translate")
    print("=" * 75)
    print(f"Translation table currently caches: {len(existing_table)} strings.")
    print()

    grand_total_unique = 0
    grand_total_uncached = 0

    for field in TRANSLATABLE_FIELDS:
        counter = per_field[field]
        n_unique = len(counter)
        n_occurrences = sum(counter.values())
        uncached = [s for s in counter if s not in existing_table]
        n_uncached = len(uncached)

        grand_total_unique += n_unique
        grand_total_uncached += n_uncached

        if n_unique == 0:
            print(f"{field:<18} | no Chinese strings found")
            continue

        print(f"{field:<18} | {n_unique:>6} unique  | "
              f"{n_occurrences:>8} occurrences  | "
              f"{n_uncached:>6} would need API call")

        # Show up to 5 sample uncached strings, with their counts
        if uncached:
            samples = sorted(uncached, key=lambda s: -counter[s])[:5]
            for s in samples:
                display = s if len(s) <= 65 else s[:62] + "..."
                print(f"    e.g. ({counter[s]:>5}×)  {display}")
        print()

    print("-" * 75)
    print(f"TOTAL unique Chinese strings  : {grand_total_unique}")
    print(f"TOTAL uncached (need API call): {grand_total_uncached}")
    print()
    # Note: per-field totals double-count strings that appear in multiple
    # fields. We could deduplicate across fields for the true call count, but
    # for now showing per-field is more useful — and the cache is per-string
    # not per-field, so duplicates get translated only once anyway.
    cross_field_unique = set()
    cross_field_uncached = set()
    for counter in per_field.values():
        for s in counter:
            cross_field_unique.add(s)
            if s not in existing_table:
                cross_field_uncached.add(s)
    print(f"After cross-field dedup       : {len(cross_field_unique)} unique, "
          f"{len(cross_field_uncached)} uncached")
    print()
    print("No API calls made. No output files written.")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Count unique Chinese strings per field without making API calls."
    )
    args = parser.parse_args()

    print(f"Reading data from: {INPUT_DIR}")
    records = load_all_records()
    print(f"Total records loaded: {len(records)}")
    print()

    # Step 1: figure out what needs translating, per field
    print("Scanning for Chinese strings in translatable fields...")
    per_field = collect_chinese_strings(records)
    table = load_existing_table()

    if args.dry_run:
        dry_run_report(per_field, table)
        return

    # --- Real run from here on ---
    # Import LLM client lazily so --dry-run doesn't require API key.
    from scripts.old.llm_client import chat

    # Flatten per-field counters into a single set of unique strings.
    all_chinese = set()
    for counter in per_field.values():
        all_chinese.update(counter.keys())
    untranslated = [s for s in all_chinese if s not in table]
    print(f"Unique Chinese strings: {len(all_chinese)}")
    print(f"Already in translation table: {len(all_chinese) - len(untranslated)}")
    print(f"Need to translate via API: {len(untranslated)}")

    # Step 2: translate the new ones
    if untranslated:
        print()
        print("Translating...")
        for i, src in enumerate(untranslated, start=1):
            try:
                dst = translate_one(src, chat)
                table[src] = dst
                print(f"  [{i}/{len(untranslated)}] {src[:60]}  ->  {dst[:60]}")
            except Exception as e:
                print(f"  [{i}/{len(untranslated)}] FAILED on {src[:60]!r}: {e}")
                # Keep going; we'll just leave this one untranslated.

            # Save periodically so progress isn't lost on interrupt.
            # Every 50 strings ≈ minor cost, major recovery benefit.
            if i % 50 == 0:
                save_table(table)

        # Final save
        save_table(table)
        print(f"\nTranslation table saved to: {TRANSLATION_TABLE_FILE}")
    else:
        print("All strings already cached — no API calls needed.")

    # Step 3: apply translations and write merged output
    print()
    print("Applying translations and writing merged dataset...")
    translated_records = apply_translation(records, table)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in translated_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(translated_records)} translated records to: {OUTPUT_FILE}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()