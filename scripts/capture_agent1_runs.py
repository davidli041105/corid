"""
Capture Agent 1's runs across the 5 representative test bundles.

For each bundle, runs Agent 1 and saves TWO files to disk:

  datasets/agent1_runs/<ip>_<port>/
    ├── trace.json              -- full message history (system, user,
    │                              assistant turns with tool_calls,
    │                              tool responses)
    └── output_to_agent2.json   -- semantic output Agent 2 would consume
                                   (bundle_analysis, tool_findings,
                                   current_hypothesis)

Existing files are overwritten — latest run wins.

Also provides `load_captured_output()` for later replay: read a stored
output_to_agent2.json back as a dict so Agent 2 can run on it without
re-invoking Agent 1.

Usage:
    python scripts/capture_agent1_runs.py            # run all 5 bundles
    python scripts/capture_agent1_runs.py --quick   # just print paths if exist
"""

import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loader import load_clean_records_list
from normalization import normalize_record
from agent1_extractor import run_agent_1


# --- Paths ---

OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "datasets" / "agent1_runs"


# --- The 5 representative test bundles ---
# Same set used in test_agent1_multi.py — match by (vendor, model).

TEST_TARGETS = [
    ("Kyocera",     "Printer Panel",              "Easy — direct evidence"),
    ("Telesquare",  "TLR-2005KSH",                "Direct — vendor in CPE"),
    ("TRENDnet",    "TV-IP110W",                  "Hard — vendor invisible"),
    ("DrayTek",     "Vigor ADSL router",          "Hard — vendor invisible"),
    ("GeoVision",   "GeoHttpServer for webcams",  "Medium — partial evidence"),
]


# --- Bundle ID for directory naming ---

def bundle_id(record: dict) -> str:
    """Produce a stable filesystem-safe identifier for a bundle's directory.

    Uses ip_str + port from the raw record. Both are reliably present in
    Shodan records and together uniquely identify the (host, service)
    observation.
    """
    ip = record.get("ip_str") or "noip"
    port = record.get("port") or "noport"
    return f"{ip}_{port}"


# --- Output extraction ---

def extract_output_to_agent2(agent1_output: dict) -> dict:
    """Pull out just the semantic output that Agent 2 consumes.

    Agent 1's full output contains bookkeeping fields (trace, warnings,
    attempts, possibly _error). Agent 2 only needs the three semantic
    parts. By splitting the file, we keep replay artifacts clean.
    """
    return {
        "bundle_analysis": agent1_output.get("bundle_analysis", []),
        "tool_findings": agent1_output.get("tool_findings", []),
        "current_hypothesis": agent1_output.get("current_hypothesis", {}),
    }


def extract_trace(agent1_output: dict) -> list[dict]:
    """Pull out the message-history trace."""
    return agent1_output.get("_trace", [])


# --- Capture one run ---

def capture_run(record: dict, note: str = "") -> tuple[Path, dict]:
    """Run Agent 1 on a single record and write the two output files.

    Returns:
      (output_dir, agent1_output) — the directory where files were written
      and the raw agent1 output (so caller can also inspect it).
    """
    bundle = normalize_record(record)
    rid = bundle_id(record)
    out_dir = OUTPUT_ROOT / rid
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  -> Running Agent 1 on {record.get('vendor')} / {record.get('model')} "
          f"({rid}{', ' + note if note else ''})...")

    a1_output = run_agent_1(bundle, verbose=False)

    # Write trace.json
    trace = extract_trace(a1_output)
    trace_path = out_dir / "trace.json"
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, ensure_ascii=False, indent=2)

    # Write output_to_agent2.json
    out2 = extract_output_to_agent2(a1_output)
    out2_path = out_dir / "output_to_agent2.json"
    with open(out2_path, "w", encoding="utf-8") as f:
        json.dump(out2, f, ensure_ascii=False, indent=2)

    # Also save a small summary file with bookkeeping info — useful for
    # debugging without having to load the whole trace.
    summary = {
        "bundle_id": rid,
        "vendor": record.get("vendor"),
        "model": record.get("model"),
        "devicetype": record.get("devicetype"),
        "_attempts": a1_output.get("_attempts"),
        "_warnings": a1_output.get("_warnings", []),
        "_error": a1_output.get("_error"),
        "trace_message_count": len(trace),
    }
    with open(out_dir / "_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"     wrote: {out2_path.name}, {trace_path.name}, _summary.json")
    return out_dir, a1_output


# --- Replay helper ---

def load_captured_output(bundle_id_str: str) -> dict:
    """Load a previously-captured output_to_agent2.json by bundle id.

    Returns the dict that Agent 2 expects as input. Use this to replay
    Agent 2 against captured Agent 1 outputs without re-invoking Agent 1.

    Raises FileNotFoundError if no capture exists for that bundle_id.
    """
    path = OUTPUT_ROOT / bundle_id_str / "output_to_agent2.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No captured output found for bundle_id={bundle_id_str!r}. "
            f"Expected: {path}"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def list_captured_bundles() -> list[str]:
    """Return the list of bundle_ids that have captured output on disk."""
    if not OUTPUT_ROOT.exists():
        return []
    return sorted(
        p.name for p in OUTPUT_ROOT.iterdir()
        if p.is_dir() and (p / "output_to_agent2.json").exists()
    )


# --- Find records ---

def find_record(records: list[dict], vendor: str, model: str) -> dict | None:
    """Find the first record matching the target vendor+model."""
    for r in records:
        if r.get("vendor") == vendor and r.get("model") == model:
            return r
    return None


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="Just list existing captures without running anything.")
    args = parser.parse_args()

    if args.quick:
        captured = list_captured_bundles()
        print(f"Captured bundles ({len(captured)}):")
        for bid in captured:
            print(f"  {bid}")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {OUTPUT_ROOT}\n")

    print("Loading clean records...")
    records = load_clean_records_list()
    print(f"  {len(records)} records.\n")

    print(f"Capturing Agent 1 runs on {len(TEST_TARGETS)} bundles...\n")
    for vendor, model, note in TEST_TARGETS:
        record = find_record(records, vendor, model)
        if record is None:
            print(f"  SKIP: no record found for {vendor}/{model}")
            continue
        capture_run(record, note=note)
        print()

    print("Done.")
    print()
    captured = list_captured_bundles()
    print(f"Total captured bundles now on disk: {len(captured)}")
    for bid in captured:
        print(f"  {bid}")


if __name__ == "__main__":
    main()
