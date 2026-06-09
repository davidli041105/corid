"""
Vocabulary lists for CoRID.

Two accumulating memories (§8 of the spec):
  - Device Category vocabulary: canonical names for device types
  - Vendor vocabulary: canonical vendor names

Both start empty at cold-start and grow as M3 commits new labels.
First-seen canonicalization: when M3 commits a new value, it's added to
the vocabulary verbatim. Subsequent commits should reuse the canonical
form when semantically equivalent (M3 owns the equivalence judgment via
the lookup tools).

Persistence: each list lives in its own JSON file. Files are
write-through: every addition flushes immediately so partial runs don't
lose vocabulary state.

This module is the source of truth for vocabulary content. The
lookup_device_category and lookup_vendor tools read from here; M3's
commit path writes here.
"""

import json
from pathlib import Path
from typing import Optional


VOCAB_DIR = Path(__file__).resolve().parent / "datasets"
DEVICE_CATEGORY_FILE = VOCAB_DIR / "vocab_device_category.json"
VENDOR_FILE = VOCAB_DIR / "vocab_vendor.json"


class Vocabulary:
    """An accumulating list of canonical strings, persisted as JSON.

    Order is preserved (insertion order), so first-seen is also first-listed.
    Membership tests are case-sensitive — canonicalization is M3's
    responsibility, not the data structure's. The list stores exactly what
    M3 committed.
    """

    def __init__(self, path: Path):
        self.path = path
        self._items: list[str] = []
        self._set: set[str] = set()
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self._items = json.load(f)
            self._set = set(self._items)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._items, f, ensure_ascii=False, indent=2)

    def list_all(self) -> list[str]:
        """Return a copy of the current vocabulary."""
        return list(self._items)

    def contains(self, value: str) -> bool:
        """Exact-match membership check."""
        return value in self._set

    def add(self, value: str) -> bool:
        """Add a new entry. Returns True if added, False if it already
        existed. Persists immediately."""
        if value in self._set:
            return False
        self._items.append(value)
        self._set.add(value)
        self._save()
        return True


# Module-level singletons. Construct lazily so import doesn't touch disk.
_device_category_vocab: Optional[Vocabulary] = None
_vendor_vocab: Optional[Vocabulary] = None


def device_category_vocab() -> Vocabulary:
    global _device_category_vocab
    if _device_category_vocab is None:
        _device_category_vocab = Vocabulary(DEVICE_CATEGORY_FILE)
    return _device_category_vocab


def vendor_vocab() -> Vocabulary:
    global _vendor_vocab
    if _vendor_vocab is None:
        _vendor_vocab = Vocabulary(VENDOR_FILE)
    return _vendor_vocab


if __name__ == "__main__":
    print("Device categories:", device_category_vocab().list_all())
    print("Vendors:", vendor_vocab().list_all())
