"""Backfill build from HistData ticks, through ``build_backfill`` (no network)."""

from __future__ import annotations

import gzip
import io
import os
import zipfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from shadow_trader.ingestion.anomaly import BAR_FLAGS, TICK_FLAGS
from shadow_trader.ingestion.backfill import BuildFailure, build_backfill
from shadow_trader.ingestion.config import Config, ConfigError, load_config
from shadow_trader.ingestion.dataset import RawCache, open_dataset
from tests.ingestion.reference_bars import reference_bars

FIXTURES = Path(__file__).parent / "fixtures"

CONFIG_TOML = """
[paths]
data_root = "{root}"

[build]
instruments = ["{pair}"]
start = "{start}"
end = "{end}"
"""


def make_config(tmp_path: Path, pair: str, month: str, start: str | None = None):
    path = tmp_path / "ingest.toml"
    path.write_text(
        CONFIG_TOML.format(
            root=tmp_path / "data", pair=pair, start=start or month, end=month
        )
    )
    return load_config(path)


def install_raw(config: Config, pair: str, period: str, csv: bytes) -> None:
    """Store ``csv`` as a HistData month zip in the raw cache."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr(f"DAT_ASCII_{pair}_T_{period}.csv", csv)
    RawCache(config.paths.data_root).store(
        "histdata",
        pair,
        period,
        "zip",
        buf.getvalue(),
        request={},
        fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
        http={},
    )


def fixture_csv(name: str) -> bytes:
    return gzip.decompress((FIXTURES / f"{name}.csv.gz").read_bytes())


def oracle_ticks(csv: bytes, utc_offset_hours):
    """Independent parse: ``utc_offset_hours(naive)`` gives NY's offset from UTC."""
    ticks = []
    for line in csv.decode().splitlines():
        stamp, bid, ask, _ = line.split(",")
        naive = datetime.strptime(stamp[:15], "%Y%m%d %H%M%S").replace(
            microsecond=int(stamp[15:]) * 1000
        )
        ticks.append(
            (
                (naive - timedelta(hours=utc_offset_hours(naive))).replace(tzinfo=UTC),
                Decimal(bid),
                Decimal(ask),
            )
        )
    return ticks


def assert_matches_reference(bars, ticks, minutes: int) -> None:
    expected, dropped = reference_bars(ticks, minutes)
    assert dropped == 0
    assert len(bars) == len(expected)
    for bar, row in zip(bars, expected, strict=True):
        for key, value in row.items():
            assert getattr(bar, key) == value, (key, bar.open_time)
        # Gap rows carry only outage flags; priced Bars only anomaly flags
        allowed = ("suspected_outage",) if bar.is_gap else (*TICK_FLAGS, *BAR_FLAGS)
        assert set(bar.quality_flags) <= set(allowed), (bar.open_time, bar)


def build_fixture(tmp_path, name, pair, month, minutes=5, scale_toml=""):
    config = make_config(tmp_path, pair, month)
    install_raw(config, pair, month.replace("-", ""), fixture_csv(name))
    result = build_backfill(config, minutes)
    return config, result


def test_winter_week_matches_reference_and_first_bar_opens_2200_utc(tmp_path):
    config, result = build_fixture(tmp_path, "eurusd_winter", "EURUSD", "2024-01")

    bars = open_dataset(config.paths.data_root, result.ref.dataset_id).read("EURUSD")
    ticks = oracle_ticks(fixture_csv("eurusd_winter"), lambda _: -5)
    assert_matches_reference(bars, ticks, 5)
    assert bars[0].open_time == datetime(2024, 1, 7, 22, 0, tzinfo=UTC)
    assert any(b.is_gap for b in bars)
    assert any(not b.is_gap for b in bars)


def test_missing_raw_month_fails_naming_the_period(tmp_path):
    config = make_config(tmp_path, "EURUSD", "2024-01")
    with pytest.raises(BuildFailure, match=r"EURUSD.*202401"):
        build_backfill(config, 5)


def session_first_opens(bars):
    firsts = {}
    for bar in bars:
        firsts.setdefault(bar.session_id, bar.open_time)
    return list(firsts.values())


def test_summer_week_first_bar_opens_2100_utc(tmp_path):
    config, result = build_fixture(tmp_path, "eurusd_summer", "EURUSD", "2024-07")

    bars = open_dataset(config.paths.data_root, result.ref.dataset_id).read("EURUSD")
    ticks = oracle_ticks(fixture_csv("eurusd_summer"), lambda _: -4)
    assert_matches_reference(bars, ticks, 5)
    assert session_first_opens(bars) == [datetime(2024, 7, 14, 21, 0, tzinfo=UTC)]


def test_dst_edge_spanning_fixture_follows_the_calendar_in_the_gap_week(tmp_path):
    config, result = build_fixture(tmp_path, "eurusd_dst_edge", "EURUSD", "2024-03")

    bars = open_dataset(config.paths.data_root, result.ref.dataset_id).read("EURUSD")
    # The fixture's week of 2024-03-10 has only an opening tick (17:00 raw), which
    # fits New York local exactly and fixed UTC-5 an hour late: weak evidence. US
    # clocks moved forward that morning but EU clocks have not, so the calendar
    # rule is fixed UTC-5 and wins; both weeks are then read at UTC-5.
    ticks = oracle_ticks(fixture_csv("eurusd_dst_edge"), lambda _: -5)
    assert_matches_reference(bars, ticks, 5)
    assert session_first_opens(bars) == [
        datetime(2024, 3, 3, 22, 0, tzinfo=UTC),
        datetime(2024, 3, 10, 21, 0, tzinfo=UTC),  # session open; first tick 22:00
    ]
    clock = result.ref.manifest["session_clock"]["EURUSD"]
    assert clock["calendar"] == ["2024-03-10"]


@pytest.mark.parametrize("minutes", [1, 15, 60])
def test_other_bar_sizes_are_built_directly_from_ticks(tmp_path, minutes):
    config, result = build_fixture(
        tmp_path, "eurusd_winter", "EURUSD", "2024-01", minutes
    )

    bars = open_dataset(config.paths.data_root, result.ref.dataset_id).read("EURUSD")
    ticks = oracle_ticks(fixture_csv("eurusd_winter"), lambda _: -5)
    assert_matches_reference(bars, ticks, minutes)


def test_usdjpy_slice_uses_the_three_decimal_scale(tmp_path):
    config, result = build_fixture(tmp_path, "usdjpy_winter", "USDJPY", "2024-01")

    bars = open_dataset(config.paths.data_root, result.ref.dataset_id).read("USDJPY")
    ticks = oracle_ticks(fixture_csv("usdjpy_winter"), lambda _: -5)
    assert_matches_reference(bars, ticks, 5)
    assert config.instruments["USDJPY"].price_scale == 3


def build_csv(tmp_path, pair, month, lines, minutes=5):
    config = make_config(tmp_path, pair, month)
    install_raw(
        config, pair, month.replace("-", ""), ("\n".join(lines) + "\n").encode()
    )
    return config, lambda: build_backfill(config, minutes)


@pytest.mark.parametrize(
    ("session_tick", "stamp"),
    [
        ("20240303", "20240310 023000000"),  # nonexistent: 02:00 -> 03:00
        ("20241027", "20241103 013000000"),  # ambiguous: 01:00-02:00 twice
    ],
)
def test_tick_at_dst_edge_is_dropped_out_of_session_not_fatal(
    tmp_path, session_tick, stamp
):
    # The offset is fixed per Session week and the clock change happens while the
    # market is closed, so a wall-clock tick there is just outside any Session
    month = f"{stamp[:4]}-{stamp[4:6]}"
    lines = [
        f"{session_tick} 170000000,1.10000,1.10010,0",
        f"{stamp},1.10000,1.10010,0",
    ]
    _, build = build_csv(tmp_path, "EURUSD", month, lines)

    result = build()

    [warning] = [w for w in result.warnings if w.code == "out_of_session_ticks"]
    assert warning.message.startswith("1 ")


def test_tick_with_more_precision_than_the_price_scale_fails(tmp_path):
    lines = [
        "20240107 170000000,1.10000,1.10010,0",
        "20240107 170001000,1.100005,1.10010,0",
    ]
    _, build = build_csv(tmp_path, "EURUSD", "2024-01", lines)

    with pytest.raises(BuildFailure, match=r"precision.*EURUSD/202401 row 2"):
        build()


def test_zero_padded_extra_decimals_are_not_over_precision(tmp_path):
    lines = ["20240107 170000000,1.100000,1.100100,0"]
    _, build = build_csv(tmp_path, "EURUSD", "2024-01", lines)

    build()


def test_unsupported_bar_size_fails_the_build(tmp_path):
    _, build = build_csv(
        tmp_path, "EURUSD", "2024-01", ["20240107 170000000,1.1,1.2,0"], minutes=7
    )

    with pytest.raises(ConfigError, match="unsupported bar size 7"):
        build()


def codes(result):
    return [w.code for w in result.warnings]


def test_month_before_the_history_floor_warns_but_builds(tmp_path):
    config = make_config(tmp_path, "EURUSD", "2008-12")
    install_raw(config, "EURUSD", "200812", b"20081207 170000000,1.30000,1.30010,0\n")

    result = build_backfill(config, 5)

    thin = [w for w in result.warnings if w.code == "thin_data"]
    assert len(thin) == 1
    assert "EURUSD" in thin[0].message
    assert thin[0].locator == "EURUSD/200812"
    assert result.ref.manifest["row_count"] > 0


def test_month_at_or_after_the_floor_does_not_warn_thin_data(tmp_path):
    _, result = build_fixture(tmp_path, "eurusd_winter", "EURUSD", "2024-01")

    assert "thin_data" not in codes(result)


def test_ticks_outside_any_session_are_dropped_counted_and_warned(tmp_path):
    lines = [
        "20240107 170000000,1.10000,1.10010,0",
        "20240112 170000000,1.10000,1.10010,0",  # Fri 17:00 NY: closure
        "20240113 120000000,1.10000,1.10010,0",  # Saturday
    ]
    _, build = build_csv(tmp_path, "EURUSD", "2024-01", lines)

    result = build()

    [warning] = [w for w in result.warnings if w.code == "out_of_session_ticks"]
    assert "2 " in warning.message
    assert warning.locator == "EURUSD"


@pytest.mark.parametrize(
    ("first_tick", "warns", "signature"),
    [
        ("170000", False, False),  # exactly the open
        ("173000", False, False),
        ("174900", False, False),  # 49 min
        ("175500", True, True),  # 55 min: start of the 55-65 signature
        ("180000", True, True),  # exactly 1 h
        ("181500", True, False),  # 75 min: off by more than 1 h, no signature
        ("183000", True, False),  # 90 min: off by more than 1 h only
        ("185500", True, True),  # 115 min: start of the 115-125 signature
        ("190000", True, True),  # 120 min
        ("191500", True, False),  # 135 min
    ],
)
def test_session_alignment_warns_per_thresholds_and_does_not_fail(
    tmp_path, first_tick, warns, signature
):
    lines = [
        f"20240107 {first_tick}000,1.10000,1.10010,0",
        "20240107 200000000,1.10000,1.10010,0",
    ]
    _, build = build_csv(tmp_path, "EURUSD", "2024-01", lines)

    result = build()  # warnings never fail the build

    found = [w for w in result.warnings if w.code == "session_alignment"]
    assert bool(found) == warns
    if warns:
        assert ("wrong-offset" in found[0].message) == signature
        assert found[0].locator == "EURUSD/week-2024-01-07"


@pytest.mark.parametrize("first_tick", ["160000", "150500"])
def test_first_tick_an_hour_or_more_before_the_open_fits_no_clock_rule(
    tmp_path, first_tick
):
    lines = [
        f"20240107 {first_tick}000,1.10000,1.10010,0",
        "20240107 200000000,1.10000,1.10010,0",
    ]
    _, build = build_csv(tmp_path, "EURUSD", "2024-01", lines)

    with pytest.raises(
        BuildFailure, match=r"EURUSD week of 2024-01-07.*before the open"
    ):
        build()


def crossed_lines(count):
    """One in-Session quote per second with ask below bid, from the open."""
    return [
        f"20240107 1700{second:02d}000,1.10010,1.10000,0" for second in range(count)
    ]


def test_backfill_flags_ticks_and_records_tick_flag_counts_in_the_manifest(tmp_path):
    _, build = build_csv(tmp_path, "EURUSD", "2024-01", crossed_lines(3))

    result = build()

    first = open_dataset(tmp_path / "data", result.ref.dataset_id).read("EURUSD")[0]
    assert first.quality_flags == ("crossed_quote",)
    coverage = result.ref.manifest["coverage"]["EURUSD"]
    assert coverage["tick_flag_counts"] == {"crossed_quote": 3}
    assert coverage["flag_counts"]["crossed_quote"] == 1


def test_alert_fraction_exceeded_is_a_build_warning(tmp_path):
    _, build = build_csv(tmp_path, "EURUSD", "2024-01", crossed_lines(3))

    result = build()

    [warning] = [
        w for w in result.warnings if w.code == "anomaly_alert_fraction_exceeded"
    ]
    assert warning.locator == "EURUSD"


def cli_config(tmp_path, month="2024-01", sizes="[5, 15]"):
    path = tmp_path / "cli.toml"
    path.write_text(
        f'[paths]\ndata_root = "{tmp_path / "data"}"\n'
        f'[build]\ninstruments = ["EURUSD"]\nbar_sizes = {sizes}\n'
        f'start = "{month}"\nend = "{month}"\n'
    )
    return path, load_config(path)


def test_cli_build_writes_one_dataset_per_bar_size_and_exits_0(tmp_path, capsys):
    from shadow_trader import main

    path, config = cli_config(tmp_path)
    install_raw(config, "EURUSD", "202401", fixture_csv("eurusd_winter"))

    assert main(["ingest", "build", "--config", str(path)]) == 0

    out = capsys.readouterr().out
    datasets = sorted((config.paths.data_root / "datasets").iterdir())
    assert len(datasets) == 2
    for folder in datasets:
        assert folder.name in out


def test_cli_build_exits_1_naming_the_missing_month(tmp_path, capsys):
    from shadow_trader import main

    path, _ = cli_config(tmp_path)

    assert main(["ingest", "build", "--config", str(path)]) == 1

    assert "202401" in capsys.readouterr().err


def test_cli_build_prints_warnings_but_exits_0(tmp_path, capsys):
    from shadow_trader import main

    path, config = cli_config(tmp_path, "2008-12", "[5]")
    install_raw(config, "EURUSD", "200812", b"20081207 170000000,1.30000,1.30010,0\n")

    assert main(["ingest", "build", "--config", str(path)]) == 0

    assert "thin_data" in capsys.readouterr().out


@pytest.mark.slow
def test_full_month_from_the_real_raw_cache_matches_reference(tmp_path):
    """Opt in with ``-m slow``; set SHADOW_TRADER_RAW_ROOT to a data_root that
    holds ``raw/histdata/EURUSD/202401.zip`` (fetched with ``ingest fetch``)."""
    root = os.environ.get("SHADOW_TRADER_RAW_ROOT")
    source = Path(root) / "raw/histdata/EURUSD/202401.zip" if root else None
    if source is None or not source.is_file():
        pytest.skip("set SHADOW_TRADER_RAW_ROOT to a data_root with EURUSD 202401")
    config = make_config(tmp_path, "EURUSD", "2024-01")
    with zipfile.ZipFile(source) as archive:
        csv = archive.read("DAT_ASCII_EURUSD_T_202401.csv")
    install_raw(config, "EURUSD", "202401", csv)

    result = build_backfill(config, 5)

    bars = open_dataset(config.paths.data_root, result.ref.dataset_id).read("EURUSD")
    assert_matches_reference(bars, oracle_ticks(csv, lambda _: -5), 5)
