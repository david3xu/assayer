"""Content addressing.

Every quality claim Assayer makes is bound to the hash of the bytes it was made
about. One canonical encoding is used everywhere so a hash computed at generate
time matches the one recomputed at review time.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

_CHUNK = 1024 * 1024


def canonical_json(obj: Any) -> bytes:
    """Encode `obj` so equal values always produce equal bytes."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj))


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def short(digest: str, n: int = 12) -> str:
    """The prefix used for run directory names and in the report."""
    return digest[:n]
