"""A tiny synthetic ingestion Dataset and prepared-config helpers.

Closes double every Bar within a Session (``BASE * 2**i``), so a log return
over ``h`` Bars is exactly ``h * ln 2``; a Session's first ``h`` Bars have no
return over ``h`` because windows never cross a Session start.
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
BASE = 1.1
BARS_PER_SESSION = 16
GAP_INDEX = 5  # the Gap slot in the first Session


def priced(instrument: str, opened: datetime, session: datetime, i: int) -> Bar:
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
    )


def session_bars(instrument: str, session: datetime) -> list[Bar]:
    bars: list[Bar] = []
    for i in range(BARS_PER_SESSION):
        opened = session + timedelta(minutes=5 * i)
        if session == SESSION_1 and i == GAP_INDEX:
            bars.append(gap_bar(instrument, opened, session))
        else:
            bars.append(priced(instrument, opened, session, i))
    return bars


def write_ingestion(
    root: Path, instruments: tuple[str, ...] = ("EURUSD",), bar_size: int = 5
) -> DatasetRef:
    """Write a synthetic ingestion Dataset under ``root / "ingestion"``."""
    path = root / "ingest.toml"
    names = ", ".join(f'"{pair}"' for pair in instruments)
    path.write_text(
        f'[paths]\ndata_root = "{root / "ingestion"}"\n'
        f"[build]\ninstruments = [{names}]\n"
    )
    bars = {
        pair: session_bars(pair, SESSION_1) + session_bars(pair, SESSION_2)
        for pair in instruments
    }
    return write_dataset(load_config(path), bar_size, bars)


def write_prepared_toml(root: Path, dataset_id: str, extra: str = "") -> Path:
    path = root / "prepare.toml"
    path.write_text(
        f'[paths]\ndata_root = "{root / "ingestion"}"\n'
        f'prepared_root = "{root / "prepared"}"\n'
        f'[input]\ndataset_id = "{dataset_id}"\n'
        "[spec]\nversion = 1\n" + extra
    )
    return path


def prepared_config(root: Path, dataset_id: str, extra: str = "") -> PreparedConfig:
    return load_prepared_config(write_prepared_toml(root, dataset_id, extra))
