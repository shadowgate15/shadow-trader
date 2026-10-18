"""Determinism: the golden hash, worker seeding and ``make_env_fn``.

``hash_parts`` is the golden-replay hash: SHA-256 over the little-endian raw bytes
of each part's four Ledger tables (``steps``, ``fills``, ``closed_lots``,
``episodes``, columns in schema order) followed by its stacked ``Observation``
arrays (``features``, ``valid``, ``position``, ``account``). A scripted run that
produces the same bytes is the same behaviour. Anything outside the four tables
(``summary.json``, the manifest) is deliberately not hashed.

``make_env_fn(...)`` returns a zero-argument, picklable callable for Gymnasium
vector environments. ``make_env_fns(...)`` makes one per worker, each seeded from
``SeedSequence.spawn`` so workers are independent and reproducible. The env has no
autoreset logic: that is the vector env's job.
"""

from __future__ import annotations

import hashlib
import platform
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from shadow_trader.preparation.dataset import PreparedDataset
from shadow_trader.preparation.scaler import Scaler

from .env_config import EnvConfig
from .ledger import Ledger

if TYPE_CHECKING:
    from .env import TradingEnv

__all__ = [
    "GOLDEN_PLATFORM",
    "LEDGER_TABLES",
    "OBSERVATION_KEYS",
    "EnvFactory",
    "GoldenPart",
    "current_platform",
    "hash_parts",
    "make_env_fn",
    "make_env_fns",
    "seed_int",
]

LEDGER_TABLES = ("steps", "fills", "closed_lots", "episodes")
OBSERVATION_KEYS = ("features", "valid", "position", "account")
#: The only platform the golden may be regenerated on (see ``current_platform``).
GOLDEN_PLATFORM = "darwin-arm64"


def current_platform() -> str:
    """``<system>-<machine>``, lower case: the golden hash's platform tag."""
    return f"{platform.system()}-{platform.machine()}".lower()


@dataclass(frozen=True)
class GoldenPart:
    """One scripted run: its Ledger and the stacked observations it saw.

    ``observations`` maps each of ``OBSERVATION_KEYS`` to an array with a leading
    axis over the observations returned by ``reset`` and every ``step``, in order.
    """

    name: str
    ledger: Ledger
    observations: Mapping[str, np.ndarray]


# --- hashing ------------------------------------------------------------------


def _feed(digest: hashlib._Hash, data: bytes) -> None:
    digest.update(struct.pack("<Q", len(data)))
    digest.update(data)


def _column_bytes(series: pl.Series) -> bytes:
    """Validity bytes then values, little endian; NaNs are canonicalised."""
    valid = series.is_not_null().to_numpy().astype("u1")
    dtype = series.dtype
    filled: np.ndarray
    if dtype == pl.String:
        parts = [(v or "").encode() for v in series.to_list()]
        values = b"".join(struct.pack("<I", len(p)) + p for p in parts)
        return valid.tobytes() + values
    if dtype == pl.Boolean:
        filled = series.fill_null(False).to_numpy().astype("u1")
    elif dtype == pl.Datetime:
        micros = series.dt.cast_time_unit("us").dt.timestamp("us").fill_null(0)
        filled = micros.to_numpy().astype("<i8")
    elif dtype.is_float():
        filled = series.fill_null(0.0).to_numpy().astype("<f8")
        filled = np.where(np.isnan(filled), np.float64("nan"), filled).astype("<f8")
    elif dtype.is_integer():
        filled = series.fill_null(0).to_numpy().astype("<i8")
    else:
        raise TypeError(f"cannot hash a {dtype} column ({series.name})")
    return valid.tobytes() + np.ascontiguousarray(filled).tobytes()


def _feed_frame(digest: hashlib._Hash, name: str, frame: pl.DataFrame) -> None:
    _feed(digest, name.encode())
    digest.update(struct.pack("<Q", frame.height))
    for column in frame.columns:
        _feed(digest, column.encode())
        _feed(digest, _column_bytes(frame[column]))


_OBSERVATION_DTYPES = {
    "features": "<f4",
    "valid": "u1",
    "position": "<f4",
    "account": "<f4",
}


def hash_parts(parts: Sequence[GoldenPart]) -> str:
    """The golden hash of ``parts`` (hex SHA-256)."""
    digest = hashlib.sha256()
    for part in parts:
        _feed(digest, part.name.encode())
        for table in LEDGER_TABLES:
            _feed_frame(digest, table, getattr(part.ledger, table))
        for key in OBSERVATION_KEYS:
            array = np.asarray(part.observations[key])
            _feed(digest, key.encode())
            digest.update(struct.pack(f"<{array.ndim + 1}Q", array.ndim, *array.shape))
            raw = np.ascontiguousarray(array.astype(_OBSERVATION_DTYPES[key]))
            _feed(digest, raw.tobytes())
    return digest.hexdigest()


# --- seeding and env factories --------------------------------------------------


def seed_int(seq: np.random.SeedSequence) -> int:
    """A 128-bit integer seed drawn from ``seq`` (does not advance ``seq``)."""
    return int.from_bytes(seq.generate_state(4, np.uint32).tobytes(), "little")


@dataclass(frozen=True)
class EnvFactory:
    """A picklable zero-argument env constructor (Dataset, Scaler and config)."""

    dataset: PreparedDataset
    scaler: Scaler
    config: EnvConfig
    seed: int | None = None
    record_ledger: bool = False

    def __call__(self) -> TradingEnv:
        from .env import make_env

        return make_env(
            self.dataset,
            self.scaler,
            self.config,
            seed=self.seed,
            record_ledger=self.record_ledger,
        )


def make_env_fn(
    dataset: PreparedDataset,
    scaler: Scaler,
    config: EnvConfig,
    *,
    seed: int | np.random.SeedSequence | None = None,
    record_ledger: bool = False,
) -> EnvFactory:
    """A zero-argument callable that builds the env, for vector environments.

    A ``SeedSequence`` seed is turned into an integer seed (``seed_int``); pass a
    spawned child per worker, or use ``make_env_fns``.
    """
    if isinstance(seed, np.random.SeedSequence):
        seed = seed_int(seed)
    return EnvFactory(dataset, scaler, config, seed, record_ledger)


def make_env_fns(
    dataset: PreparedDataset,
    scaler: Scaler,
    config: EnvConfig,
    num_envs: int,
    *,
    seed: int | np.random.SeedSequence,
    record_ledger: bool = False,
) -> list[EnvFactory]:
    """``num_envs`` env factories seeded by ``SeedSequence(seed).spawn(num_envs)``.

    An integer ``seed`` is wrapped in a fresh ``SeedSequence`` (so repeating the
    call repeats the seeds); a ``SeedSequence`` is spawned from directly, which
    advances its spawn counter.
    """
    root = (
        seed
        if isinstance(seed, np.random.SeedSequence)
        else np.random.SeedSequence(seed)
    )
    return [
        make_env_fn(dataset, scaler, config, seed=child, record_ledger=record_ledger)
        for child in root.spawn(num_envs)
    ]
