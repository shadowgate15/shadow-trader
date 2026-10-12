"""build-live: a live Dataset from the raw quote log, sealed slots only."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from shadow_trader import main
from shadow_trader.ingestion.bars import Quote
from shadow_trader.ingestion.dataset import build_dataset, open_dataset
from shadow_trader.ingestion.live import LiveIngester, SourceDown, SourceUp
from shadow_trader.ingestion.livedataset import build_live
from shadow_trader.ingestion.replay import ReplayAdapter

from .test_dataset import make_bars, make_config
from .test_live import OPEN, q, ticking_clock

END = OPEN + timedelta(hours=4)


def record(tmp_path, events, start=OPEN, **kwargs):
    """Run a live ingester so the raw quote log (and run report) exist."""
    config = make_config(tmp_path)
    LiveIngester(
        config,
        ReplayAdapter(events, instrument="EURUSD"),
        sink=lambda bar, revision: None,
        clock=ticking_clock(),
        start=start,
        **kwargs,
    ).run()
    return config


def test_only_sealed_slots_enter_the_live_dataset(tmp_path):
    config = record(
        tmp_path,
        [q(timedelta(minutes=1)), q(timedelta(minutes=7)), q(timedelta(hours=3))],
    )

    ref = build_live(config, 5, OPEN, END)

    bars = open_dataset(config.paths.data_root, ref.dataset_id).read("EURUSD")
    # slot 1:55 seals at 3:00 (end + 1 h repair window); 2:00 is still repairable
    assert bars[0].open_time == OPEN
    assert bars[-1].open_time == OPEN + timedelta(hours=1, minutes=55)
    assert [b.tick_count for b in bars[:3]] == [1, 1, None]
    assert ref.manifest["source"] == "replay"
    assert ref.manifest["parent"] is None


def build_bars_of(config, ref):
    return open_dataset(config.paths.data_root, ref.dataset_id).read("EURUSD")


def test_a_late_quote_inside_the_repair_window_is_built_at_its_latest_revision(
    tmp_path,
):
    config = record(
        tmp_path,
        [
            q(timedelta(minutes=1), bid="1.10000", ask="1.10010"),
            q(timedelta(minutes=6), bid="1.10020", ask="1.10030"),
            # arrives after slot 0 closed, inside the repair window: Revision 1
            q(timedelta(minutes=2), bid="1.10500", ask="1.10510"),
            q(timedelta(hours=3)),
        ],
    )

    bars = build_bars_of(config, build_live(config, 5, OPEN, END))

    assert (bars[0].open, bars[0].high, bars[0].tick_count) == (1.1, 1.105, 2)


def test_source_down_overlap_and_long_gap_runs_are_flagged_priced_bars_are_not(
    tmp_path,
):
    down = OPEN + timedelta(minutes=10)
    config = record(
        tmp_path,
        [
            q(timedelta(minutes=1)),
            SourceDown(down),
            SourceUp(down + timedelta(minutes=5)),  # exactly slot 10..15
            q(timedelta(minutes=16)),
            q(timedelta(hours=3)),
        ],
    )

    bars = build_bars_of(config, build_live(config, 5, OPEN, END))

    flags = {b.open_time - OPEN: b.quality_flags for b in bars}
    assert flags[timedelta(minutes=5)] == ()  # plain Gap: short, no outage
    assert flags[timedelta(minutes=10)] == ("suspected_outage",)  # source-down
    assert flags[timedelta(minutes=15)] == ()  # priced Bar
    # the 16 min .. 3 h silence is a long run: the run-length rule re-applies
    assert flags[timedelta(minutes=20)] == ("suspected_outage",)


def test_manifest_records_live_source_raw_inputs_and_intervals(tmp_path):
    down = OPEN + timedelta(minutes=10)
    config = record(
        tmp_path,
        [
            q(timedelta(minutes=1)),
            SourceDown(down),
            SourceUp(down + timedelta(minutes=5)),
            q(timedelta(hours=3)),
        ],
    )

    manifest = build_live(config, 5, OPEN, END).manifest

    assert manifest["source"] == "replay"
    assert manifest["view"] == "latest"
    assert any(k.startswith("live/replay/EURUSD/") for k in manifest["raw_inputs"])
    live = manifest["live"]
    assert live["source_down"]["EURUSD"][0]["start"] == down.isoformat()
    assert len(live["runs"]["EURUSD"]) == 1


def test_a_live_dataset_never_takes_a_backfill_dataset_as_parent(tmp_path):
    config = record(tmp_path, [q(timedelta(minutes=1)), q(timedelta(hours=3))])
    backfill = build_dataset(config, 5, make_bars(start=OPEN))

    live_ref = build_live(config, 5, OPEN, END)

    assert backfill.manifest["source"] == "histdata"
    assert live_ref.manifest["parent"] is None
    assert live_ref.dataset_id != backfill.dataset_id


def test_a_later_build_has_the_previous_live_dataset_as_parent_with_hardlinks(
    tmp_path,
):
    new_year = datetime(2023, 12, 31, 22, 0, tzinfo=UTC)
    config = record(
        tmp_path,
        [
            Quote(new_year + timedelta(minutes=1), Decimal("1.1"), Decimal("1.1001")),
            Quote(new_year + timedelta(hours=3), Decimal("1.1"), Decimal("1.1001")),
            Quote(OPEN + timedelta(minutes=1), Decimal("1.2"), Decimal("1.2001")),
            Quote(OPEN + timedelta(hours=3), Decimal("1.2"), Decimal("1.2001")),
        ],
        start=new_year,
    )
    first = build_live(config, 5, new_year, new_year + timedelta(hours=2))
    second = build_live(config, 5, new_year, END)

    assert second.manifest["parent"] == first.dataset_id
    old = "bars/instrument=EURUSD/year=2023/part-0.parquet"
    assert (first.path / old).stat().st_ino == (second.path / old).stat().st_ino


def test_cli_build_live_builds_the_range_and_prints_the_dataset_id(tmp_path, capsys):
    config = record(tmp_path, [q(timedelta(minutes=1)), q(timedelta(hours=3))])

    code = main(
        [
            "ingest",
            "build-live",
            "--config",
            str(tmp_path / "ingest.toml"),
            "--from",
            OPEN.isoformat(),
            "--to",
            END.isoformat(),
        ]
    )

    expected = build_live(config, 5, OPEN, END).dataset_id
    assert code == 0
    assert expected in capsys.readouterr().out


def test_cli_build_live_rejects_a_bad_range_as_a_config_error(tmp_path, capsys):
    record(tmp_path, [q(timedelta(minutes=1))])
    base = ["ingest", "build-live", "--config", str(tmp_path / "ingest.toml")]

    assert main([*base, "--from", "yesterday", "--to", "today"]) == 2
    assert main([*base, "--from", END.isoformat(), "--to", OPEN.isoformat()]) == 2
