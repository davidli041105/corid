"""
Bundle Normalization — §4 of CoRID. Shodan schema.

Takes a raw Shodan record and produces a structured "bundle" that M3 will
reason over. The bundle has two parts:

  evidence  — fields M3 reads (banner, http details, hostnames, etc.),
              grouped by structural role. M3 NEVER sees the `labels` part.
  labels    — ground truth (devicetype, vendor, model, product_raw).
              Used downstream for evaluation only — passed alongside the
              bundle but explicitly separated.

This is a DETERMINISTIC PRE-STAGE, not a tool. It runs on every record
before M3 ever sees the data. It is distinct from `response_regularization`
(tool #4 in the final spec), which is an M3-callable tool that further
trims an already-normalized bundle based on M3's judgment.

Design principles:
  - Less interference, more raw evidence. M3 decides what's useful.
  - Drop only clear label-leakage fields (`product`, `cpe`, `cpe23`, `tags`)
    and clear non-evidence noise (Shodan-internal opaque hashes).
  - Keep network/host context (`org`, `isp`, `cloud`) as evidence even
    though they don't identify the device — they diversify the evidence
    source and may help M3 in edge cases.
  - Vendor-specific parsed dicts (e.g., `kyocera_printer_panel`) pass
    through unchanged. Yes, the field name contains the vendor — accepted
    as part of the evidence M3 reads.
"""

import re
from typing import Any, Optional


# --- Cleaning rules ---

HEADER_MAX_CHARS = 2000

# Shodan's `data` field (raw banner) sometimes has very long content;
# truncate to keep token costs bounded.
DATA_MAX_CHARS = 2000

REDACTED_PATTERN = re.compile(r"<REDACTED>")

# Fields inside Shodan's `http` dict that are opaque integer hashes,
# carrying no human-readable identification signal. Dropped from the bundle.
HTTP_HASH_FIELDS = {
    "title_hash", "robots_hash", "sitemap_hash", "dom_hash",
    "headers_hash", "server_hash", "html_hash", "securitytxt_hash",
}

# Fields to strip from the raw record entirely — label leakage.
# `product`: literally concatenates vendor + model.
# `cpe`, `cpe23`: structured CPE strings encoding vendor + model.
# `tags`: Shodan category tags that may include device-category equivalents.
LEAKAGE_FIELDS = {"product", "cpe", "cpe23", "tags"}

# Shodan-internal noise we don't want in evidence.
# `_shodan`: scanner metadata (region, module, crawler info).
# `hash`: opaque integer fingerprint.
# `timestamp`: scan timestamp.
# `opts`: usually empty.
# `ip` (integer form): we keep `ip_str` instead which is human-readable.
NOISE_FIELDS = {"_shodan", "hash", "timestamp", "opts", "ip"}


def _clean_value(val: Any) -> Any:
    """Return val if it's meaningful, else None.

    Treats None, empty strings, and the literal 'null' as missing.
    Replaces <REDACTED> placeholders. Recursively cleans dicts and lists.
    """
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() == "null":
            return None
        return REDACTED_PATTERN.sub("[date]", s)
    if isinstance(val, dict):
        cleaned = {k: _clean_value(v) for k, v in val.items()}
        # Drop keys whose cleaned value is None
        cleaned = {k: v for k, v in cleaned.items() if v is not None}
        return cleaned if cleaned else None
    if isinstance(val, list):
        cleaned = [_clean_value(v) for v in val]
        cleaned = [v for v in cleaned if v is not None]
        return cleaned if cleaned else None
    return val


def _truncate(s: Optional[str], max_chars: int, name: str) -> Optional[str]:
    """Truncate a long string and mark the truncation explicitly."""
    if s is None or not isinstance(s, str):
        return s
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"  [...truncated, original {name} was {len(s)} chars]"


def _clean_http(http: Optional[dict]) -> Optional[dict]:
    """Clean Shodan's `http` sub-dict: drop opaque hash fields,
    truncate `html` if very long, recursively clean."""
    if not isinstance(http, dict):
        return None
    out = {}
    for k, v in http.items():
        if k in HTTP_HASH_FIELDS:
            continue
        cleaned = _clean_value(v)
        if cleaned is None:
            continue
        # `html` can be very large; truncate.
        if k == "html" and isinstance(cleaned, str):
            cleaned = _truncate(cleaned, HEADER_MAX_CHARS, "html")
        out[k] = cleaned
    return out if out else None


# --- Main entry point ---

def normalize_record(record: dict) -> dict:
    """Convert a raw Shodan record into a CoRID bundle.

    Returns a dict with two top-level keys:
      - 'evidence': what M3 sees
      - 'labels': ground truth, used for evaluation only
    """

    # Build the evidence dict by passing through most fields after cleaning,
    # while dropping label-leakage fields, Shodan-internal noise, and
    # the label fields themselves.
    evidence: dict[str, Any] = {}

    # The fields we explicitly handle. Everything else from the source
    # record that isn't a label, leakage, or noise gets passed through
    # cleaned (preserving the "less interference" principle).
    label_keys = {"devicetype", "vendor", "model"}
    skip_keys = LEAKAGE_FIELDS | NOISE_FIELDS | label_keys

    for k, v in record.items():
        if k in skip_keys:
            continue

        if k == "http":
            cleaned = _clean_http(v)
        elif k == "data":
            # Truncate the raw banner.
            cleaned = _clean_value(v)
            if isinstance(cleaned, str):
                cleaned = _truncate(cleaned, DATA_MAX_CHARS, "data")
        else:
            cleaned = _clean_value(v)

        if cleaned is not None:
            evidence[k] = cleaned

    # Ground-truth labels, kept separate. `product` is preserved here as
    # `product_raw` for synonym-aware evaluation (first entry canonical,
    # but Shodan's `product` is usually a single string, not comma-separated).
    labels = {
        "device_category": _clean_value(record.get("devicetype")),
        "vendor": _clean_value(record.get("vendor")),
        "model": _clean_value(record.get("model")),
        "product_raw": _clean_value(record.get("product")),
    }
    # Drop any None-valued label entries
    labels = {k: v for k, v in labels.items() if v is not None}

    return {
        "evidence": evidence,
        "labels": labels,
    }


# --- Convenience: bulk normalization ---

def normalize_all(records: list[dict]) -> list[dict]:
    """Apply normalize_record to a list of raw records."""
    return [normalize_record(r) for r in records]


# --- Debug / dev: print a sample bundle ---

if __name__ == "__main__":
    import json
    from loader import load_clean_records_list

    records = load_clean_records_list()
    print(f"Loaded {len(records)} clean records.\n")

    if not records:
        print("No records to normalize. Did you run merge_data.py?")
    else:
        print("First raw record (selected fields):")
        r = records[0]
        for k in ["ip_str", "port", "devicetype", "vendor", "model",
                  "product", "org", "isp"]:
            v = r.get(k)
            if isinstance(v, str) and len(v) > 80:
                v = v[:80] + "..."
            print(f"  {k}: {v!r}")

        print()
        print("Normalized bundle (full):")
        bundle = normalize_record(r)
        print(json.dumps(bundle, ensure_ascii=False, indent=2)[:3000])
