"""Turn the cached raw m1 weeks into the stored M5 Bars, validated and summarised.

A Bar is built from *complete* five-minute buckets only: a bucket missing any
of its five m1 rows produces no Bar and falls into a Gap (no minimum-count
fallback). Only real Bars are stored; Gaps stay implicit and nothing is
forward-filled. Bid and ask are aggregated separately and kept unaltered.

There is deliberately NO cross-side validation (ask >= bid or any bound on the
spread): crossed and zero-spread rows are legitimate in this vendor's data, so
they are kept and merely counted in the summary.

Parquet schema (``<DATA_DIR>/bars/EURUSD_M5.parquet``), one row per Bar,
sorted by ``time``:

    time        timestamp[ms, tz=UTC]  bucket open
    bid_open, bid_high, bid_low, bid_close      float64
    ask_open, ask_high, ask_low, ask_close      float64

Mid is derived from the stored sides by :func:`mid`, not stored.
"""

import csv
import gzip
import io
import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from shadow_trader import config
from shadow_trader.fetch import TIMESTAMP_FORMAT, Week

BAR_INTERVAL = timedelta(minutes=5)
# Longest closure tolerated anywhere. A plain weekend is ~48-49h, but the
# vendor's Christmas / New Year closures in the pinned range run to 79h (and the
# early closes around them leave a few 4h-36h Gaps), so the ceiling is set just
# above that: 4 days. Anything longer aborts unless it is a configured known
# vendor break (config.KNOWN_MISSING_WEEKS).
MAX_WEEKEND_GAP = timedelta(days=4)
CORRIDOR_END = timedelta(hours=46)  # shortest weekend Gap seen is ~47h
PIP = 1e-4

SIDES = ("bid", "ask")
FIELDS = ("open", "high", "low", "close")
PRICE_COLUMNS = tuple(f"{side}_{f}" for side in SIDES for f in FIELDS)
COLUMNS = ("time", *PRICE_COLUMNS)
SCHEMA = pa.schema(
    [pa.field("time", pa.timestamp("ms", tz="UTC"))]
    + [pa.field(c, pa.float64()) for c in PRICE_COLUMNS]
)

# Upper edges of the Gap histogram buckets; the last bucket is open-ended. The
# 2h-to-46h corridor is what `gap_threshold` depends on: watch it.
HISTOGRAM_EDGES = (
    ("<=15m", timedelta(minutes=15)),
    ("15m-1h", timedelta(hours=1)),
    ("1h-2h", timedelta(hours=2)),
    ("2h-46h", CORRIDOR_END),
    ("46h-4d", MAX_WEEKEND_GAP),
)
HISTOGRAM_LAST = ">4d"


class BarsError(Exception):
    """The derived Bars violate an ingest invariant; ingest must not continue."""


@dataclass(frozen=True)
class Gap:
    start: datetime  # time of the last Bar before the break
    end: datetime  # time of the first Bar after it

    @property
    def length(self) -> timedelta:
        """Time with no Bar: from the end of the earlier Bar to the next open."""
        return self.end - self.start - BAR_INTERVAL


@dataclass
class RowCounts:
    total: int = 0
    crossed: int = 0  # ask below bid in any of the four components
    zero_spread: int = 0  # ask equal to bid in any component
    deepest_crossing: float = 0.0  # price units, bid minus ask, >= 0


@dataclass(frozen=True)
class Summary:
    bar_count: int
    first: datetime
    last: datetime
    gaps: list[Gap]
    rows: RowCounts
    histogram: dict[str, int] = field(default_factory=dict)

    def render(self) -> str:
        r = self.rows

        def pct(n: int) -> str:
            return f"{100 * n / r.total:.2f}%" if r.total else "n/a"

        lines = [
            f"bars: {self.bar_count}",
            f"range: {self.first:%Y-%m-%d %H:%M} to {self.last:%Y-%m-%d %H:%M} UTC",
            f"gaps: {len(self.gaps)}",
        ]
        if self.gaps:
            largest = max(self.gaps, key=lambda g: g.length)
            smallest = min(self.gaps, key=lambda g: g.length)
            lines.append(
                f"largest gap: {largest.length} "
                f"({largest.start:%Y-%m-%d %H:%M} to {largest.end:%Y-%m-%d %H:%M})"
            )
            lines.append(f"gap min: {smallest.length}  max: {largest.length}")
        else:
            lines.append("largest gap: none")
        lines.append("gap histogram:")
        lines += [f"  {label:>8}: {n}" for label, n in self.histogram.items()]
        lines += [
            f"m1 rows: {r.total}",
            f"crossed rows (any component): {r.crossed} ({pct(r.crossed)})",
            f"zero-spread rows: {r.zero_spread} ({pct(r.zero_spread)})",
            f"deepest crossing: {r.deepest_crossing / PIP:.1f} pips",
        ]
        return "\n".join(lines)


def floor_bucket(moment: datetime) -> datetime:
    return moment.replace(minute=moment.minute - moment.minute % 5, second=0)


def parse_week(week: Week) -> list[tuple[datetime, tuple[float, ...]]]:
    try:
        text = gzip.decompress(week.path.read_bytes()).decode()
        reader = csv.reader(io.StringIO(text))
        next(reader)
        rows = [
            (
                datetime.strptime(row[0], TIMESTAMP_FORMAT).replace(tzinfo=UTC),
                tuple(float(x) for x in row[1:9]),
            )
            for row in reader
            if row
        ]
    except (OSError, EOFError, ValueError, IndexError, StopIteration) as error:
        raise BarsError(f"{week.path}: unreadable weekly file ({error})") from error
    # Checked per m1 row too: a NaN would otherwise vanish inside max()/min().
    for moment, prices in rows:
        if not all(math.isfinite(p) and p > 0 for p in prices):
            raise BarsError(f"price at {moment} is not finite and positive")
    return rows


def count_row(counts: RowCounts, prices: tuple[float, ...]) -> None:
    counts.total += 1
    depths = [prices[i] - prices[i + 4] for i in range(4)]  # bid - ask
    if any(d > 0 for d in depths):
        counts.crossed += 1
        counts.deepest_crossing = max(counts.deepest_crossing, *depths)
    if any(d == 0 for d in depths):
        counts.zero_spread += 1


def aggregate(
    weeks: list[Week], counts: RowCounts, progress: Callable[[str], None]
) -> dict[str, list]:
    """Columnar Bars from complete five-minute buckets of the weekly files."""
    columns: dict[str, list] = {c: [] for c in COLUMNS}
    for n, week in enumerate(weeks, 1):
        buckets: dict[datetime, dict[datetime, tuple[float, ...]]] = {}
        for moment, prices in parse_week(week):
            count_row(counts, prices)
            buckets.setdefault(floor_bucket(moment), {})[moment] = prices
        for opens in sorted(buckets):
            rows = buckets[opens]
            if len(rows) != 5:
                continue  # incomplete bucket: absent, widens a Gap
            ordered = [rows[k] for k in sorted(rows)]
            columns["time"].append(opens)
            for side, base in (("bid", 0), ("ask", 4)):
                columns[f"{side}_open"].append(ordered[0][base])
                columns[f"{side}_high"].append(max(r[base + 1] for r in ordered))
                columns[f"{side}_low"].append(min(r[base + 2] for r in ordered))
                columns[f"{side}_close"].append(ordered[-1][base + 3])
        progress(
            f"aggregating {n}/{len(weeks)} {week.path.name}: "
            f"{len(columns['time'])} bars so far"
        )
    return columns


def clip(columns: dict[str, list]) -> dict[str, list]:
    """Keep only Bars inside the pinned History range (weekly files overhang it)."""
    keep = [config.HISTORY_START <= t < config.HISTORY_END for t in columns["time"]]
    return {
        name: [v for v, k in zip(values, keep, strict=True) if k]
        for name, values in columns.items()
    }


def find_gaps(times: list[datetime]) -> list[Gap]:
    return [Gap(a, b) for a, b in pairwise(times) if b - a > BAR_INTERVAL]


def is_known_break(gap: Gap) -> bool:
    """True if the Gap is one configured vendor-unpublished week plus its weekends."""
    return any(
        gap.start < opens
        and gap.end > closes
        and gap.length <= (closes - opens) + 2 * MAX_WEEKEND_GAP
        for opens, closes in config.KNOWN_MISSING_WEEKS
    )


def validate(columns: dict[str, list]) -> None:
    """Abort (BarsError) unless all five ingest invariants hold."""
    times = columns["time"]
    if not times:
        raise BarsError("no Bars were produced")
    for a, b in pairwise(times):
        if b <= a:
            raise BarsError(f"timestamps not strictly increasing at {b}")
    for name in PRICE_COLUMNS:
        for t, v in zip(times, columns[name], strict=True):
            if not math.isfinite(v) or v <= 0:
                raise BarsError(f"{name} at {t} is not finite and positive: {v}")
    for side in SIDES:
        series = zip(*(columns[f"{side}_{f}"] for f in FIELDS), strict=True)
        for t, (o, h, low, c) in zip(times, series, strict=True):
            if h < max(o, c) or low > min(o, c) or h < low:
                raise BarsError(f"{side} OHLC inconsistent at {t}")
    for a, b in pairwise(times):
        if (b - a) % BAR_INTERVAL:
            raise BarsError(f"unexpected Bar interval {b - a} at {b}")
    for gap in find_gaps(times):
        if gap.length > MAX_WEEKEND_GAP and not is_known_break(gap):
            raise BarsError(
                f"Gap longer than a weekend: {gap.length} ({gap.start} to {gap.end})"
            )


def histogram(gaps: list[Gap]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for gap in gaps:
        label = next(
            (name for name, edge in HISTOGRAM_EDGES if gap.length <= edge),
            HISTOGRAM_LAST,
        )
        counts[label] += 1
    labels = [name for name, _ in HISTOGRAM_EDGES] + [HISTOGRAM_LAST]
    return {label: counts[label] for label in labels}


def mid(table: pa.Table) -> dict[str, list[float]]:
    """Mid open/high/low/close, derived from the stored bid and ask columns."""
    return {
        f"mid_{f}": [
            (b + a) / 2
            for b, a in zip(
                table[f"bid_{f}"].to_pylist(),
                table[f"ask_{f}"].to_pylist(),
                strict=True,
            )
        ]
        for f in FIELDS
    }


def bars_path() -> Path:
    return config.DATA_DIR / "bars" / "EURUSD_M5.parquet"


def ingest(
    weeks: list[Week],
    out: Path,
    *,
    progress: Callable[[str], None] = print,
) -> Summary:
    """Aggregate, validate, write Parquet, print the summary. Aborts on violation."""
    counts = RowCounts()
    columns = clip(aggregate(weeks, counts, progress))
    validate(columns)
    gaps = find_gaps(columns["time"])
    summary = Summary(
        len(columns["time"]),
        columns["time"][0],
        columns["time"][-1],
        gaps,
        counts,
        histogram(gaps),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".part")
    pq.write_table(pa.table(columns, schema=SCHEMA), partial)
    partial.replace(out)
    progress(f"wrote {out}")
    progress(summary.render())
    return summary
