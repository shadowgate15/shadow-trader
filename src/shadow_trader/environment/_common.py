"""Small helpers shared by the configuration modules: hashing and number checks."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict
from typing import Any, cast

from shadow_trader.preparation._io import canonical_json

__all__ = ["HashableConfig", "canonical_hash", "check_int", "check_number", "sha256"]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value: Any) -> str:
    """SHA-256 of ``value``'s canonical JSON (sorted keys, floats via ``repr``)."""
    return sha256(canonical_json(value))


class HashableConfig:
    """Mixin for frozen config dataclasses: canonical JSON and a stable hash."""

    def canonical_json(self) -> bytes:
        return canonical_json(asdict(cast(Any, self)))

    def config_hash(self) -> str:
        return sha256(self.canonical_json())


def check_number(value: Any, name: str, error: type[Exception] = ValueError) -> float:
    """``value`` as a finite float; ``error`` is raised for a bool, non-number or
    non-finite value."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise error(f"{name} must be a number")
    if not math.isfinite(value):
        raise error(f"{name} must be finite")
    return float(value)


def check_int(value: Any, name: str, error: type[Exception] = ValueError) -> int:
    """``value`` as an int; ``error`` is raised for a bool or non-integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise error(f"{name} must be an integer")
    return value
