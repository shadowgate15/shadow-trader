"""A synthetic ingestion Dataset and prepared-config helpers.

Closes double every Bar within a Session (``BASE * 2**i``), so a log return
over ``h`` Bars is exactly ``h * ln 2``; a Session's first ``h`` Bars have no
return over ``h`` because windows never cross a Session start.

Three weekly Sessions are written, each long enough for the spec's 288-Bar
lookback: the default split makes Session 1 train, 2 validation and 3 test, and
the default Embargo is ``E = 288 + EPISODE_BARS`` Bars.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from shadow_trader.ingestion.bars import Bar, gap_bar
from shadow_trader.ingestion.config import load_config
from shadow_trader.ingestion.dataset import DatasetRef, write_dataset
from shadow_trader.preparation.config import PreparedConfig, load_prepared_config

SESSION_1 = datetime(2024, 1, 7, 22, 0, tzinfo=UTC)
SESSION_2 = datetime(2024, 1, 14, 22, 0, tzinfo=UTC)
SESSION_3 = datetime(2024, 1, 21, 22, 0, tzinfo=UTC)
SESSIONS = (SESSION_1, SESSION_2, SESSION_3)
BASE = 1.1
BARS_PER_SESSION = 700
LONGEST_LOOKBACK = 288
EPISODE_BARS = 2
EMBARGO = LONGEST_LOOKBACK + EPISODE_BARS
GAP_INDEX = 5  # the Gap slot in the first Session


def priced(
    instrument: str,
    opened: datetime,
    session: datetime,
    i: int,
    flags: tuple[str, ...] = (),
) -> Bar:
    close = BASE * 2.0**i
    return Bar(
        instrument=instrument,
        open_time=opened,
        open=close,
        high=close,
        low=close,
        close=close,
        spread_close=0.0001,
        spread_mean=0.0001,
        tick_count=10 + i,
        is_gap=False,
        session_id=session,
        is_rollover=False,
        quality_flags=flags,
    )


def session_bars(
    instrument: str,
    session: datetime,
    count: int = BARS_PER_SESSION,
    gaps: tuple[int, ...] = (GAP_INDEX,),
    flags: dict[int, tuple[str, ...]] | None = None,
) -> list[Bar]:
    """One Session of Bars; ``gaps`` and ``flags`` apply to the first Session only."""
    first = session == SESSION_1
    bars: list[Bar] = []
    for i in range(count):
        opened = session + timedelta(minutes=5 * i)
        if first and i in gaps:
            bars.append(gap_bar(instrument, opened, session))
        else:
            bar_flags = (flags or {}).get(i, ()) if first else ()
            bars.append(priced(instrument, opened, session, i, bar_flags))
    return bars


def write_ingestion(
    root: Path,
    instruments: tuple[str, ...] = ("EURUSD",),
    bar_size: int = 5,
    count: int = BARS_PER_SESSION,
    gaps: tuple[int, ...] = (GAP_INDEX,),
    flags: dict[int, tuple[str, ...]] | None = None,
    sessions: tuple[datetime, ...] = SESSIONS,
) -> DatasetRef:
    """Write a synthetic ingestion Dataset under ``root / "ingestion"``."""
    path = root / "ingest.toml"
    names = ", ".join(f'"{pair}"' for pair in instruments)
    path.write_text(
        f'[paths]\ndata_root = "{root / "ingestion"}"\n'
        f"[build]\ninstruments = [{names}]\n"
    )
    bars = {
        pair: [
            bar for s in sessions for bar in session_bars(pair, s, count, gaps, flags)
        ]
        for pair in instruments
    }
    return write_dataset(load_config(path), bar_size, bars)


def iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def split_toml(
    train_start: datetime = SESSION_1,
    validation_start: datetime = SESSION_2,
    test_start: datetime = SESSION_3,
    test_end: datetime | None = None,
    episode_bars: int | None = EPISODE_BARS,
) -> str:
    text = (
        '[split]\nscheme = "chronological"\n'
        f"train_start = {iso(train_start)}\n"
        f"validation_start = {iso(validation_start)}\n"
        f"test_start = {iso(test_start)}\n"
    )
    if test_end is not None:
        text += f"test_end = {iso(test_end)}\n"
    if episode_bars is not None:
        text += f"[embargo]\nepisode_bars = {episode_bars}\n"
    return text


def write_prepared_toml(
    root: Path, dataset_id: str, extra: str = "", split: str | None = None
) -> Path:
    path = root / "prepare.toml"
    path.write_text(
        f'[paths]\ndata_root = "{root / "ingestion"}"\n'
        f'prepared_root = "{root / "prepared"}"\n'
        f'[input]\ndataset_id = "{dataset_id}"\n'
        "[spec]\nversion = 1\n" + (split_toml() if split is None else split) + extra
    )
    return path


def prepared_config(
    root: Path, dataset_id: str, extra: str = "", split: str | None = None
) -> PreparedConfig:
    return load_prepared_config(write_prepared_toml(root, dataset_id, extra, split))


def write_bars(root: Path, bars: dict[str, list[Bar]], bar_size: int = 5) -> DatasetRef:
    """Write caller-supplied Bars as a synthetic ingestion Dataset."""
    path = root / "ingest.toml"
    names = ", ".join(f'"{pair}"' for pair in bars)
    path.write_text(
        f'[paths]\ndata_root = "{root / "ingestion"}"\n'
        f"[build]\ninstruments = [{names}]\n"
    )
    return write_dataset(load_config(path), bar_size, bars)


def prepared_frame(root: Path, bars: list[Bar], version: int = 2):
    """Features and anomaly policy over one Instrument's Bars, read back from disk.

    Skips the split (these Bars are far shorter than a split's Embargo): the
    frame is written as a plain prepared Dataset and read through the public API.
    """
    from shadow_trader.preparation import SPECS, open_prepared
    from shadow_trader.preparation.build import featured_frame
    from shadow_trader.preparation.dataset import write_prepared

    spec = SPECS[version]
    ref = write_prepared(
        root / "prepared",
        {"EURUSD": featured_frame(bars, spec)},
        parent_ingestion="synthetic",
        spec=spec,
    )
    return open_prepared(root / "prepared", ref.dataset_id).read("EURUSD")
