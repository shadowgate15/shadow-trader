"""Write-once raw quote log for live runs (ADR 0001, ADR 0003).

Every live quote is appended as one JSON line ``{source_ts, arrival_ts, bid,
ask}`` to an hourly file named by *arrival* hour, so an append only ever
touches the current hour. When an hour has passed its file is gzip-compressed
into the raw cache and its ``.meta.json`` sidecar hashes the compressed bytes;
a closed hour is never touched again. The log is the single source of truth
for rebuilding live Bars and live Datasets.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import IO

from .bars import Quote
from .dataset import RawCache

_PERIOD = "%Y%m%dT%H"
_GZ = "jsonl.gz"


def raw_log_dir(data_root: Path, source: str, instrument: str) -> Path:
    return Path(data_root) / "raw" / "live" / source / instrument


class RawQuoteLog:
    """Appends quotes to the current hour's file; compresses closed hours."""

    def __init__(self, data_root: Path, source: str, instrument: str) -> None:
        self._cache = RawCache(data_root)
        self._source = source
        self._instrument = instrument
        self._dir = raw_log_dir(data_root, source, instrument)
        self._period: str | None = None
        self._handle: IO[str] | None = None

    def append(self, quote: Quote, arrival: datetime) -> None:
        arrival = arrival.astimezone(UTC)
        period = arrival.strftime(_PERIOD)
        if period != self._period:
            self._close_handle()
            self._period = period
            self._dir.mkdir(parents=True, exist_ok=True)
            self.compress_closed(before=period)
            self._handle = (self._dir / f"{period}.jsonl").open("a")
        assert self._handle is not None
        line = {
            "source_ts": quote.ts.astimezone(UTC).isoformat(),
            "arrival_ts": arrival.isoformat(),
            "bid": str(quote.bid),
            "ask": str(quote.ask),
        }
        self._handle.write(json.dumps(line) + "\n")
        self._handle.flush()

    def close(self) -> None:
        """Release the open hour's file; it stays uncompressed until it closes."""
        self._close_handle()
        self._period = None

    def compress_closed(self, before: str) -> None:
        """Compress every plain hourly file older than period ``before``."""
        if not self._dir.exists():
            return
        for path in sorted(self._dir.glob("*.jsonl")):
            if path.stem >= before:
                continue
            self._cache.store(
                "live/" + self._source,
                self._instrument,
                path.stem,
                _GZ,
                gzip.compress(path.read_bytes(), mtime=0),
                request={"source": self._source, "instrument": self._instrument},
                fetched_at=datetime.now(UTC),
                http={},
            )
            path.unlink()

    def _close_handle(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def read_raw_log(
    data_root: Path, source: str, instrument: str
) -> Iterator[tuple[Quote, datetime]]:
    """Every logged quote with its arrival time, in arrival order."""
    directory = raw_log_dir(data_root, source, instrument)
    if not directory.exists():
        return
    files = sorted(
        (p for p in directory.iterdir() if p.name.endswith((".jsonl", f".{_GZ}"))),
        key=lambda p: p.name.split(".")[0],
    )
    for path in files:
        raw = path.read_bytes()
        text = (gzip.decompress(raw) if path.name.endswith(_GZ) else raw).decode()
        for line in text.splitlines():
            entry = json.loads(line)
            quote = Quote(
                datetime.fromisoformat(entry["source_ts"]),
                Decimal(entry["bid"]),
                Decimal(entry["ask"]),
            )
            yield quote, datetime.fromisoformat(entry["arrival_ts"])


__all__ = ["RawQuoteLog", "raw_log_dir", "read_raw_log"]
