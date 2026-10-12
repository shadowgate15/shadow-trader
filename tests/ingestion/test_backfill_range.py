"""Backfill range semantics: default end, grid clipping, lineage, bad input order."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest

from shadow_trader.ingestion.backfill import BuildFailure, build_backfill
from shadow_trader.ingestion.config import Config
from shadow_trader.ingestion.dataset import open_dataset

from .test_backfill import build_csv, fixture_csv, install_raw, make_config


def without_end(config: Config) -> Config:
    return dataclasses.replace(
        config, build=dataclasses.replace(config.build, end=None)
    )


def test_end_defaults_to_the_last_complete_utc_month(tmp_path):
    config = without_end(make_config(tmp_path, "EURUSD", "2024-01"))
    install_raw(config, "EURUSD", "202401", fixture_csv("eurusd_winter"))
    now = datetime(2024, 3, 10, tzinfo=UTC)  # last complete month: 2024-02

    with pytest.raises(BuildFailure, match=r"EURUSD.*202402"):
        build_backfill(config, 5, now=now)


def test_default_end_builds_every_month_up_to_the_last_complete_one(tmp_path):
    config = without_end(make_config(tmp_path, "EURUSD", "2024-01"))
    install_raw(config, "EURUSD", "202401", fixture_csv("eurusd_winter"))
    install_raw(config, "EURUSD", "202402", b"20240207 170000000,1.10000,1.10010,0\n")

    result = build_backfill(config, 5, now=datetime(2024, 3, 10, tzinfo=UTC))

    bars = open_dataset(config.paths.data_root, result.ref.dataset_id).read("EURUSD")
    assert any(b.open_time.month == 2 for b in bars)


def test_grid_is_clipped_to_the_declared_range(tmp_path):
    # Session Sun 2024-01-28 17:00 NY .. Fri 02-02 17:00 NY straddles Jan/Feb.
    boundary = datetime(2024, 2, 1, 5, 0, tzinfo=UTC)  # Feb 1 00:00 New York

    january = make_config(tmp_path, "EURUSD", "2024-01")
    install_raw(january, "EURUSD", "202401", b"20240129 090000000,1.1,1.2,0\n")
    ref = build_backfill(january, 60).ref
    bars = open_dataset(january.paths.data_root, ref.dataset_id).read("EURUSD")
    assert max(b.open_time for b in bars) < boundary

    (tmp_path / "feb").mkdir()
    february = make_config(tmp_path / "feb", "EURUSD", "2024-02")
    install_raw(february, "EURUSD", "202402", b"20240202 090000000,1.1,1.2,0\n")
    ref = build_backfill(february, 60).ref
    bars = open_dataset(february.paths.data_root, ref.dataset_id).read("EURUSD")
    assert min(b.open_time for b in bars) >= boundary


def test_out_of_order_ticks_fail_the_build_with_a_failure_report(tmp_path):
    lines = [
        "20240108 100000000,1.10000,1.10010,0",
        "20240108 090000000,1.10000,1.10010,0",  # an hour earlier: slot closed
    ]
    config, build = build_csv(tmp_path, "EURUSD", "2024-01", lines)

    with pytest.raises(BuildFailure, match=r"out of order.*EURUSD/202401") as caught:
        build()
    assert caught.value.report_path is not None
    assert not (config.paths.data_root / "datasets").exists()


def test_backfill_records_the_previous_backfill_of_the_series_as_parent(tmp_path):
    config = make_config(tmp_path, "EURUSD", "2024-01")
    install_raw(config, "EURUSD", "202401", fixture_csv("eurusd_winter"))
    first = build_backfill(config, 5)
    assert first.ref.manifest["parent"] is None

    longer = dataclasses.replace(
        config, build=dataclasses.replace(config.build, end="2024-02")
    )
    install_raw(longer, "EURUSD", "202402", b"20240207 170000000,1.10000,1.10010,0\n")
    second = build_backfill(longer, 5)

    assert second.ref.dataset_id != first.ref.dataset_id
    assert second.ref.manifest["parent"] == first.ref.dataset_id
