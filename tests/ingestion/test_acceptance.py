"""v1 acceptance: backfill reproducibility and replay-vs-batch parity (spec #28).

The fixture tests run always. The ``slow`` tests run the real EUR/USD history from
the raw cache: set ``SHADOW_TRADER_RAW_ROOT`` to a data_root holding
``raw/histdata/EURUSD/<YYYYMM>.zip`` for 2009-01 onwards (fetched with
``ingest fetch``). ``SHADOW_TRADER_REPLAY_RANGE`` (``YYYY-MM`` or
``YYYY-MM:YYYY-MM``, default ``2024-01``) picks the replay-parity range.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import time
from pathlib import Path

import pytest

from shadow_trader.ingestion.backfill import build_backfill
from shadow_trader.ingestion.config import Config, load_config
from shadow_trader.ingestion.dataset import open_dataset
from shadow_trader.ingestion.live import LiveIngester
from shadow_trader.ingestion.replay import ReplayAdapter

from .test_backfill import CONFIG_TOML, fixture_csv, install_raw, make_config


def prices(bars):
    """Bars without quality flags: backfill applies the Gap run-length rule and does
    not run the tick detector, which the stream-equals-batch property covers."""
    return [dataclasses.replace(b, quality_flags=()) for b in bars]


def replay_bars(config: Config, bar_size: int):
    sealed = []
    ingester = LiveIngester(
        config,
        ReplayAdapter.from_raw_cache(config, "EURUSD"),
        sink=lambda bar, revision: sealed.append((bar, revision)),
        bar_size=bar_size,
    )
    ingester.run()
    return ingester, sealed


def test_second_build_from_the_same_raw_cache_reproduces_dataset_id(tmp_path):
    config = make_config(tmp_path, "EURUSD", "2024-01")
    install_raw(config, "EURUSD", "202401", fixture_csv("eurusd_winter"))

    first = build_backfill(config, 5)
    second = build_backfill(config, 5)

    assert second.ref.dataset_id == first.ref.dataset_id


def test_same_raw_bytes_in_a_fresh_data_root_reproduce_dataset_id(tmp_path):
    ids = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        config = make_config(tmp_path / name, "EURUSD", "2024-01")
        install_raw(config, "EURUSD", "202401", fixture_csv("eurusd_winter"))
        ids.append(build_backfill(config, 5).ref.dataset_id)

    assert ids[0] == ids[1]


def test_replay_live_sealed_bars_equal_batch_bars_with_no_revisions(tmp_path):
    config = make_config(tmp_path, "EURUSD", "2024-01")
    install_raw(config, "EURUSD", "202401", fixture_csv("eurusd_winter"))
    batch = open_dataset(
        config.paths.data_root, build_backfill(config, 5).ref.dataset_id
    ).read("EURUSD")

    ingester, sealed = replay_bars(config, 5)

    assert ingester.late_dropped == 0
    assert all(revision == 0 for _, revision in sealed)
    assert prices([bar for bar, _ in sealed]) == prices(batch)
    assert prices(ingester.revisions.latest()) == prices(batch)


# --- real-cache acceptance (opt in with -m slow) -------------------------------


def _real_root() -> Path:
    root = os.environ.get("SHADOW_TRADER_RAW_ROOT")
    if not root or not (Path(root) / "raw/histdata/EURUSD").is_dir():
        pytest.skip("set SHADOW_TRADER_RAW_ROOT to a data_root with EURUSD raw months")
    return Path(root)


def _periods(root: Path) -> list[str]:
    found = sorted(
        p.name.removesuffix(".zip")
        for p in (root / "raw/histdata/EURUSD").glob("??????.zip")
    )
    if not found or found[0] > "200901":
        pytest.skip("raw cache does not reach back to 2009-01")
    return [p for p in found if p >= "200901"]


def _real_config(tmp_path: Path, root: Path, start: str, end: str) -> Config:
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    if not (data / "raw").exists():
        (data / "raw").symlink_to(root / "raw", target_is_directory=True)
    path = tmp_path / "ingest.toml"
    path.write_text(CONFIG_TOML.format(root=data, pair="EURUSD", start=start, end=end))
    return load_config(path)


def _month(period: str) -> str:
    return f"{period[:4]}-{period[4:]}"


@pytest.mark.slow
def test_real_eurusd_2009_to_present_builds_and_dataset_id_reproduces(
    tmp_path, record_property
):
    root = _real_root()
    periods = _periods(root)
    config = _real_config(tmp_path, root, "2009-01", _month(periods[-1]))

    started = time.monotonic()
    first = build_backfill(config, 5)
    built = time.monotonic() - started
    second = build_backfill(config, 5)

    assert second.ref.dataset_id == first.ref.dataset_id
    bars = open_dataset(config.paths.data_root, first.ref.dataset_id).read("EURUSD")
    report = json.loads(first.report_path.read_text()) if first.report_path else {}
    record_property("dataset_id", first.ref.dataset_id)
    record_property("build_seconds", round(built, 1))
    record_property("bars", len(bars))
    record_property(
        "warnings", [(w.code, w.message, w.locator) for w in first.warnings]
    )
    record_property("report_path", str(first.report_path))
    print(
        f"dataset_id={first.ref.dataset_id} bars={len(bars)} build={built:.0f}s "
        f"report={first.report_path} warnings="
        f"{[(w.code, w.message, w.locator) for w in first.warnings]} "
        f"report_keys={sorted(report)}"
    )
    assert len(bars) > 0


@pytest.mark.slow
def test_real_eurusd_replay_live_sealed_bars_equal_batch(tmp_path):
    root = _real_root()
    _periods(root)  # skips when the cache is incomplete
    chosen = os.environ.get("SHADOW_TRADER_REPLAY_RANGE", "2024-01")
    start, _, end = chosen.partition(":")
    assert re.fullmatch(r"\d{4}-\d{2}", start)
    config = _real_config(tmp_path, root, start, end or start)

    batch = open_dataset(
        config.paths.data_root, build_backfill(config, 5).ref.dataset_id
    ).read("EURUSD")
    started = time.monotonic()
    ingester, sealed = replay_bars(config, 5)
    print(
        f"replay {chosen}: {len(sealed)} emissions in {time.monotonic() - started:.0f}s"
    )

    assert ingester.late_dropped == 0
    assert prices(ingester.revisions.latest()) == prices(batch)
