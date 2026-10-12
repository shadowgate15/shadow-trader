"""Gap versus outage semantics, driven through ``build_backfill`` (no network)."""

from __future__ import annotations

import io
import zipfile
from datetime import UTC, datetime, timedelta

import pytest

from shadow_trader.ingestion.backfill import BuildFailure, build_backfill
from shadow_trader.ingestion.config import load_config
from shadow_trader.ingestion.dataset import RawCache, open_dataset
from tests.ingestion.test_backfill import (
    build_csv,
    fixture_csv,
    install_raw,
    make_config,
)

# Sunday 2024-01-07 17:00 New York (EST) is 22:00 UTC: the first Session's open.
OPEN = datetime(2024, 1, 7, 22, 0, tzinfo=UTC)


def tick_line(minutes_after_open: int) -> str:
    local = OPEN - timedelta(hours=5) + timedelta(minutes=minutes_after_open)
    return f"{local:%Y%m%d %H%M%S}000,1.10000,1.10010,0"


def build_bars(tmp_path, tick_minutes, bar_size=5):
    lines = [tick_line(m) for m in tick_minutes]
    config, build = build_csv(tmp_path, "EURUSD", "2024-01", lines, bar_size)
    ref = build().ref
    return open_dataset(config.paths.data_root, ref.dataset_id).read("EURUSD")


def flags_at(bars, minutes_after_open: int):
    (bar,) = [
        b for b in bars if b.open_time == OPEN + timedelta(minutes=minutes_after_open)
    ]
    return bar.quality_flags


def test_run_of_gaps_at_the_threshold_is_flagged_and_stays_null_priced(tmp_path):
    # ticks at 0 and 35 min leave six empty 5-minute slots (5..30): exactly 30 min
    bars = build_bars(tmp_path, [0, 35])

    for minute in range(5, 35, 5):
        assert flags_at(bars, minute) == ("suspected_outage",)
    gap = next(b for b in bars if b.open_time == OPEN + timedelta(minutes=5))
    assert gap.is_gap and gap.open is None and gap.close is None
    assert flags_at(bars, 0) == ()
    assert flags_at(bars, 35) == ()


def test_run_shorter_than_the_threshold_stays_a_plain_gap(tmp_path):
    # five empty slots (5..25) is 25 min
    bars = build_bars(tmp_path, [0, 30])

    for minute in range(5, 30, 5):
        assert flags_at(bars, minute) == ()


@pytest.mark.parametrize(
    ("bar_size", "empty_slots", "flagged"),
    [(15, 2, True), (15, 1, False), (1, 30, True), (1, 29, False)],
)
def test_threshold_is_a_duration_independent_of_bar_size(
    tmp_path, bar_size, empty_slots, flagged
):
    last = (empty_slots + 1) * bar_size
    bars = build_bars(tmp_path, [0, last], bar_size)

    expected = ("suspected_outage",) if flagged else ()
    assert flags_at(bars, bar_size) == expected


def test_runs_do_not_cross_sessions(tmp_path):
    week = 7 * 24 * 60
    # one Gap at the end of Session 1 and five at the start of Session 2:
    # six in all (30 min) but split by the weekly closure
    bars = build_bars(tmp_path, [0, 5 * 24 * 60 - 10, week, week + 30])

    assert flags_at(bars, 5 * 24 * 60 - 5) == ()  # last slot of Session 1
    for minute in range(5, 30, 5):
        assert flags_at(bars, week + minute) == ()


def raw_dataset_dirs(config):
    folder = config.paths.data_root / "datasets"
    return list(folder.iterdir()) if folder.exists() else []


def test_empty_raw_month_fails_naming_the_period_and_writes_no_manifest(tmp_path):
    config, build = build_csv(tmp_path, "EURUSD", "2024-01", [])

    with pytest.raises(BuildFailure, match=r"empty.*EURUSD.*202401"):
        build()
    assert raw_dataset_dirs(config) == []


def test_corrupt_raw_month_fails_naming_the_period(tmp_path):
    config = make_config(tmp_path, "EURUSD", "2024-01")
    cache = RawCache(config.paths.data_root)
    cache.store(
        "histdata",
        "EURUSD",
        "202401",
        "zip",
        b"not a zip",
        request={},
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        http={},
    )

    with pytest.raises(BuildFailure, match=r"corrupt.*EURUSD.*202401"):
        build_backfill(config, 5)
    assert raw_dataset_dirs(config) == []


def test_zip_without_the_month_member_is_corrupt(tmp_path):
    config = make_config(tmp_path, "EURUSD", "2024-01")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("something_else.csv", b"x")
    RawCache(config.paths.data_root).store(
        "histdata",
        "EURUSD",
        "202401",
        "zip",
        buf.getvalue(),
        request={},
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        http={},
    )

    with pytest.raises(BuildFailure, match=r"corrupt.*EURUSD.*202401"):
        build_backfill(config, 5)


def test_a_missing_middle_month_fails_naming_it(tmp_path):
    config = make_config(tmp_path, "EURUSD", "2024-03", start="2024-01")
    install_raw(config, "EURUSD", "202401", fixture_csv("eurusd_winter"))
    install_raw(config, "EURUSD", "202403", fixture_csv("eurusd_dst_edge"))

    with pytest.raises(BuildFailure, match=r"missing.*EURUSD.*202402"):
        build_backfill(config, 5)
    assert raw_dataset_dirs(config) == []


ACK_TOML = """
[paths]
data_root = "{root}"

[build]
instruments = ["EURUSD"]
start = "{start}"
end = "{end}"
{acknowledged}
[rules.gap]
outage_threshold_minutes = {threshold}
"""


def ack_config(tmp_path, start, end, acknowledged=(), threshold=30, name="ingest"):
    entries = "".join(
        f'[[build.acknowledged_missing]]\ninstrument = "EURUSD"\nmonth = "{m}"\n'
        f'reason = "HistData lacks it"\n'
        for m in acknowledged
    )
    path = tmp_path / f"{name}.toml"
    path.write_text(
        ACK_TOML.format(
            root=tmp_path / "data",
            start=start,
            end=end,
            acknowledged=entries,
            threshold=threshold,
        )
    )
    return load_config(path)


def csv_line(stamp: str) -> bytes:
    return f"{stamp},1.10000,1.10010,0\n".encode()


def three_month_config(tmp_path, acknowledged=("2024-02",), **kwargs):
    config = ack_config(tmp_path, "2024-01", "2024-03", acknowledged, **kwargs)
    install_raw(config, "EURUSD", "202401", csv_line("20240107 170000000"))
    install_raw(config, "EURUSD", "202403", csv_line("20240303 170000000"))
    return config


def test_acknowledged_month_builds_as_source_missing_gap_rows(tmp_path):
    config = three_month_config(tmp_path)

    ref = build_backfill(config, 5).ref

    bars = open_dataset(config.paths.data_root, ref.dataset_id).read("EURUSD")
    feb_start = datetime(2024, 2, 1, 5, 0, tzinfo=UTC)  # midnight New York
    feb_end = datetime(2024, 3, 1, 5, 0, tzinfo=UTC)
    february = [b for b in bars if feb_start <= b.open_time < feb_end]
    # every in-Session slot of the month, none missing
    assert len(february) > 4 * 120 * 12
    assert all(b.is_gap and b.open is None for b in february)
    assert all(b.quality_flags == ("source_missing",) for b in february)
    jan_slot = datetime(2024, 1, 31, 17, tzinfo=UTC)
    january = next(b for b in bars if b.open_time == jan_slot)
    assert january.quality_flags == ("suspected_outage",)
    assert ref.manifest["acknowledged_missing"] == [
        {"instrument": "EURUSD", "month": "2024-02", "reason": "HistData lacks it"}
    ]
    flag_counts = ref.manifest["coverage"]["EURUSD"]["flag_counts"]
    assert flag_counts["source_missing"] == len(february)


def dataset_id(tmp_path, name, **kwargs):
    (tmp_path / name).mkdir()
    config = three_month_config(tmp_path / name, **kwargs)
    return build_backfill(config, 5).ref.dataset_id


def test_changing_the_threshold_or_acknowledged_list_changes_the_dataset_id(tmp_path):
    base = dataset_id(tmp_path, "base")

    assert dataset_id(tmp_path, "same") == base
    assert dataset_id(tmp_path, "threshold", threshold=45) != base
    assert dataset_id(tmp_path, "note", acknowledged=("2024-02", "2023-11")) != base


def test_leading_and_trailing_edges_follow_the_missing_month_rule(tmp_path):
    config = ack_config(tmp_path, "2023-12", "2024-02", ("2023-12", "2024-02"))
    install_raw(config, "EURUSD", "202401", csv_line("20240107 170000000"))

    ref = build_backfill(config, 5).ref

    bars = open_dataset(config.paths.data_root, ref.dataset_id).read("EURUSD")
    assert bars[0].open_time < datetime(2023, 12, 2, tzinfo=UTC)
    assert bars[0].quality_flags == ("source_missing",)
    assert bars[-1].open_time > datetime(2024, 2, 28, tzinfo=UTC)
    assert bars[-1].quality_flags == ("source_missing",)


def test_unacknowledged_leading_month_before_the_first_tick_fails(tmp_path):
    config = ack_config(tmp_path, "2023-12", "2024-01")
    install_raw(config, "EURUSD", "202401", csv_line("20240107 170000000"))

    with pytest.raises(BuildFailure, match=r"missing.*EURUSD.*202312"):
        build_backfill(config, 5)


def test_cli_exits_1_naming_the_period_and_writes_no_manifest(tmp_path, capsys):
    from shadow_trader import main

    config = ack_config(tmp_path, "2024-01", "2024-01")

    assert main(["ingest", "build", "--config", str(tmp_path / "ingest.toml")]) == 1

    assert "202401" in capsys.readouterr().err
    assert list(config.paths.data_root.glob("datasets/*/manifest.json")) == []


def test_cli_builds_an_acknowledged_month(tmp_path, capsys):
    from shadow_trader import main

    config = three_month_config(tmp_path)

    assert main(["ingest", "build", "--config", str(tmp_path / "ingest.toml")]) == 0

    [manifest] = config.paths.data_root.glob("datasets/*/manifest.json")
    assert "2024-02" in manifest.read_text()
