"""The Python API fixed by the spec (section 9), imported from the package root."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from shadow_trader.ingestion import (
    BarBuilder,
    DatasetRef,
    LiveIngester,
    build_dataset,
    open_dataset,
)
from shadow_trader.ingestion.bars import Quote
from shadow_trader.ingestion.replay import ReplayAdapter

from .test_backfill import fixture_csv, install_raw, make_config

OPEN = datetime(2024, 1, 7, 22, 0, tzinfo=UTC)


def test_build_dataset_builds_from_the_raw_cache_and_open_dataset_reads_it(tmp_path):
    config = make_config(tmp_path, "EURUSD", "2024-01")
    install_raw(config, "EURUSD", "202401", fixture_csv("eurusd_winter"))

    ref = build_dataset(config, 5)

    assert isinstance(ref, DatasetRef)
    dataset = open_dataset(config.paths.data_root, ref.dataset_id)
    assert dataset.manifest["bar_size"] == 5
    assert dataset.read("EURUSD")


def test_bar_builder_pushes_quotes_and_pulls_closed_bars():
    builder = BarBuilder("EURUSD", 5, 5)

    assert (
        builder.push(Quote(OPEN + timedelta(minutes=1), Decimal("1.1"), Decimal("1.2")))
        == []
    )
    closed = builder.push(
        Quote(OPEN + timedelta(minutes=6), Decimal("1.1"), Decimal("1.2"))
    )

    assert [b.open_time for b in closed] == [OPEN]


def test_live_ingester_runs_from_config_and_adapter_alone(tmp_path):
    config = make_config(tmp_path, "EURUSD", "2024-01")
    quote = Quote(OPEN + timedelta(minutes=1), Decimal("1.1"), Decimal("1.2"))

    ingester = LiveIngester(config, ReplayAdapter([quote], instrument="EURUSD"))
    ingester.run()

    assert ingester.dropped_out_of_session == 0
