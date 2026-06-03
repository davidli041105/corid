"""
Bundle Normalization — §4 of CoRID.

Takes a raw record from sample_500.jsonl (or the merged dataset) and produces
a structured "bundle" that M3 will reason over. The bundle has two parts:

  evidence  — fields M3 reads (banner, header, server, title, etc.), grouped
              by structural role. M3 NEVER sees the `labels` part.
  labels    — ground truth (device_type_sub, manufacturer, device, product).
              Used downstream for evaluation only — passed alongside the
              bundle but explicitly separated.

This is a DETERMINISTIC PRE-STAGE, not a tool. It runs on every record
before M3 ever sees the data. It is distinct from `response_regularization`
(tool #4 in the final spec), which is an M3-callable tool that further
trims an already-normalized bundle based on M3's judgment.

The normalizer's job is small but real:
  1. Pick out the identifying-evidence fields from the raw record. Drop
     non-identifying fields (geo/network context, ASN, etc.) and metadata
     that's not useful for device identification.
  2. Strip obvious noise (<REDACTED> placeholders, empty strings, the literal
     string "null").
  3. Truncate the `header` field to the first 2000 chars (pathological
     headers can be 1.6M chars; most identifying info is at the top).
  4. Group evidence by structural role: network_context, server_claim,
     banner_evidence, cert_evidence.
  5. Keep ground-truth labels separate so M3 can never see them during
     reasoning. Retain `product_raw` only on the label side for synonym-
     aware evaluation downstream.

Design notes:
  - The bundle is a plain dict, not a class. Easier to serialize and debug.
  - Empty groups are still present in the output (as empty dicts), so M3
    sees a consistent schema across all bundles.
  - `product` field is NOT included in evidence. It's the same field as
    the label-source `product_raw` and including it in evidence would leak
    the answer to M3. Kept only on the labels side, for synonym-aware
    evaluation logic.

This module does NOT call any LLMs. Pure deterministic transformation.
"""

import re
from typing import Optional


# --- Field allocations: which raw fields go where in the bundle ---
#
# Each entry is (raw_field_name, output_key_within_group). We don't just
# pass the raw field name through because some renaming clarifies semantics
# downstream (e.g., raw `service` -> bundle `service_name`).

NETWORK_CONTEXT_FIELDS = [
    ("ip", "ip"),
    ("port", "port"),
    ("service", "service_name"),
    ("protocol", "protocol"),
]

SERVER_CLAIM_FIELDS = [
    ("server", "server"),
    ("title", "title"),
]

BANNER_EVIDENCE_FIELDS = [
    ("banner", "banner"),
    ("header", "header"),  # gets truncated, see HEADER_MAX_CHARS below
]

CERT_EVIDENCE_FIELDS = [
    ("cert_issuer", "cert_issuer"),
    ("cert_subject", "cert_subject"),
]

# Ground-truth labels — separated explicitly so M3 cannot accidentally see them.
# `product_raw` carries the comma-separated alternatives for synonym-aware
# evaluation: the first entry is canonical, the rest are synonyms, and an
# M3 commit matches if it produces any of these forms.
LABEL_FIELDS = [
    ("device_type_sub", "device_category"),
    ("manufacturer", "vendor"),
    ("device", "model"),
    ("product", "product_raw"),
]


# --- Cleaning rules ---

HEADER_MAX_CHARS = 2000

REDACTED_PATTERN = re.compile(r"<REDACTED>")


def _clean_value(val):
    """Return val if it's a meaningful populated value, else None.

    Treats None, empty strings, and the literal string 'null' as missing.
    Strips whitespace from strings. Replaces <REDACTED> placeholders that
    appear in dates with a clean marker."""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        if not s or s.lower() == "null":
            return None
        # Replace <REDACTED> placeholders so they don't show up as spurious
        # tokens for M3 (or for the statistics store later).
        s = REDACTED_PATTERN.sub("[date]", s)
        return s
    return val


def _populate_group(record: dict, field_specs: list[tuple[str, str]]) -> dict:
    """Pull the listed fields out of the raw record, cleaning each.

    Skips fields whose cleaned value is None — the output group only contains
    populated fields. This keeps bundles compact.
    """
    out = {}
    for raw_name, out_name in field_specs:
        cleaned = _clean_value(record.get(raw_name))
        if cleaned is not None:
            out[out_name] = cleaned
    return out


def _truncate_header(header: Optional[str]) -> Optional[str]:
    """Truncate the header field to HEADER_MAX_CHARS. Returns None if input
    is None. Marks the truncation explicitly so M3 knows the value is cut off."""
    if header is None:
        return None
    if len(header) <= HEADER_MAX_CHARS:
        return header
    return header[:HEADER_MAX_CHARS] + f"  [...truncated, original was {len(header)} chars]"


# --- Main entry point ---

def normalize_record(record: dict) -> dict:
    """Convert a raw record into a CoRID bundle.

    Returns a dict with two top-level keys:
      - 'evidence': what M3 sees
      - 'labels': ground truth, used for evaluation only

    The 'evidence' part has sub-groups for clarity:
      - network_context: ip, port, service, protocol
      - server_claim: server, title
      - banner_evidence: banner, header (truncated)
      - cert_evidence: cert_issuer, cert_subject
    """
    evidence = {
        "network_context": _populate_group(record, NETWORK_CONTEXT_FIELDS),
        "server_claim": _populate_group(record, SERVER_CLAIM_FIELDS),
        "banner_evidence": _populate_group(record, BANNER_EVIDENCE_FIELDS),
        "cert_evidence": _populate_group(record, CERT_EVIDENCE_FIELDS),
    }

    # Apply header truncation. The banner_evidence group is the place
    # the giant headers live.
    if "header" in evidence["banner_evidence"]:
        evidence["banner_evidence"]["header"] = _truncate_header(
            evidence["banner_evidence"]["header"]
        )

    # Ground truth labels — kept entirely separate.
    labels = _populate_group(record, LABEL_FIELDS)

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
    # When run directly, normalize the first record from the 500-sample
    # and print it as a pretty JSON blob. Useful for eyeballing the schema.
    import json
    from scripts.old.sampler import load_sample

    sample = load_sample()
    print(f"Loaded {len(sample)} records from sample.")
    print()
    print("First raw record (selected fields):")
    r = sample[0]
    for k in ["ip", "port", "service", "server", "title", "banner",
              "device_type_sub", "manufacturer", "device"]:
        v = r.get(k)
        if isinstance(v, str) and len(v) > 100:
            v = v[:100] + "..."
        print(f"  {k}: {v!r}")

    print()
    print("Normalized bundle (full):")
    bundle = normalize_record(r)
    print(json.dumps(bundle, ensure_ascii=False, indent=2))
