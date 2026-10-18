"""Shared JSON helpers: one canonical encoding for hashes, one atomic writer."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _isoformat(value: object) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    raise TypeError(f"cannot canonicalise {value!r}")


def canonical_json(value: Any) -> bytes:
    """Key-sorted, whitespace-free JSON bytes (datetimes as UTC ISO); hash input."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=_isoformat
    ).encode()


def atomic_write_json(path: Path, document: Any) -> None:
    """Write indented, key-sorted JSON by atomic rename; creates parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(document, indent=2, sort_keys=True))
    os.replace(tmp, path)
