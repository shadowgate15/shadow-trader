"""The advisory live-vs-backfill reconciliation diff."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from shadow_trader import main
from shadow_trader.ingestion.bars import Bar
from shadow_trader.ingestion.dataset import write_dataset
from shadow_trader.ingestion.reconcile import ReconciliationRefused, reconcile

from .test_dataset import make_config

OPEN = datetime(2024, 1, 7, 22, 0, tzinfo=UTC)
PIP = 0.0001


def bar(i: int, instrument: str = "EURUSD", **changes) -> Bar:
    price = 1.1 + i * 0.00001
    base = Bar(
        instrument=instrument,
        open_time=OPEN + timedelta(minutes=5 * i),
        open=price,
        high=price + 0.0002,
        low=price - 0.0002,
        close=price + 0.0001,
        spread_close=0.0001,
        spread_mean=0.0001,
        tick_count=10,
        is_gap=False,
        session_id=OPEN,
        is_rollover=False,
    )
    return dataclasses.replace(base, **changes)


def gap(i: int) -> Bar:
    return bar(
        i,
        open=None,
        high=None,
        low=None,
        close=None,
        spread_close=None,
        spread_mean=None,
        tick_count=None,
        is_gap=True,
    )


def datasets(config, live, backfill, bar_size=5):
    live_ref = write_dataset(config, bar_size, {"EURUSD": live}, source="replay")
    back_ref = write_dataset(config, bar_size, {"EURUSD": backfill})
    return live_ref.dataset_id, back_ref.dataset_id


def report_of(config, live, backfill):
    return json.loads(reconcile(config, live, backfill).read_text())


def test_identical_series_match_within_tolerance_and_tolerances_come_from_config(
    tmp_path,
):
    config = make_config(tmp_path, "[reconciliation]\nprice_pips = 0.7\n")
    series = [bar(i) for i in range(10)]
    live_id, back_id = datasets(config, series, series)

    path = reconcile(config, live_id, back_id)

    report = json.loads(path.read_text())
    assert path == (
        config.paths.data_root
        / "reports/reconciliation"
        / f"reconciliation-{live_id}-{back_id}.json"
    )
    assert report["report_version"] == 1
    assert report["summary"]["matched"] == 10
    assert report["summary"]["within"] == 10
    assert report["summary"]["beyond"] == 0
    assert report["tolerances"]["price_pips"] == 0.7
    assert report["tolerances"]["spread_relative"] == 0.3


def test_price_beyond_tolerance_is_counted_drifted_and_listed_worst_first(tmp_path):
    config = make_config(tmp_path)
    backfill = [bar(i) for i in range(10)]
    live = list(backfill)
    live[3] = bar(3, close=(backfill[3].close or 0) + 2 * PIP)  # 2 pips off
    live[4] = bar(4, close=(backfill[4].close or 0) + 0.3 * PIP)  # inside 0.5 pip
    live_id, back_id = datasets(config, live, backfill)

    report = report_of(config, live_id, back_id)

    assert (report["summary"]["within"], report["summary"]["beyond"]) == (9, 1)
    assert report["drift"]["close"]["max_abs"] == pytest.approx(2.0)
    worst = report["worst_bars"]["listed"]
    assert [w["open_time"] for w in worst] == [
        backfill[3].open_time.isoformat(),
        backfill[4].open_time.isoformat(),
    ]
    assert worst[0]["price_deviation_pips"] == pytest.approx(2.0)
    assert worst[0]["beyond_tolerance"] and not worst[1]["beyond_tolerance"]


def test_spread_tolerance_is_the_larger_of_absolute_and_relative(tmp_path):
    config = make_config(tmp_path)
    backfill = [bar(i, spread_mean=1.0 * PIP) for i in range(3)]
    live = [
        bar(0, spread_mean=1.25 * PIP),  # 0.25 pip: inside max(0.3 pip, 30%)
        bar(1, spread_mean=1.35 * PIP),  # 0.35 pip: beyond both
        bar(2, spread_mean=1.0 * PIP),
    ]
    live_id, back_id = datasets(config, live, backfill)

    summary = report_of(config, live_id, back_id)["summary"]

    assert (summary["within"], summary["beyond"]) == (2, 1)


def test_tick_count_ratio_is_reported_separately_from_within_beyond(tmp_path):
    config = make_config(tmp_path)
    backfill = [bar(i, tick_count=10) for i in range(3)]
    live = [bar(0, tick_count=21), bar(1, tick_count=4), bar(2, tick_count=20)]
    live_id, back_id = datasets(config, live, backfill)

    report = report_of(config, live_id, back_id)

    assert report["tick_ratio"]["outside_tolerance"] == 2  # 2.1 and 0.4; 2.0 is in
    assert report["summary"]["within"] == 3


def test_one_sided_gaps_are_always_listed_outside_the_500_cap(tmp_path):
    config = make_config(tmp_path)
    count = 520
    backfill = [bar(i) for i in range(count)]
    live = [
        bar(i, close=(backfill[i].close or 0) + (i + 1) * PIP) for i in range(count)
    ]
    live[7] = gap(7)
    backfill[9] = gap(9)
    live_id, back_id = datasets(config, live, backfill)

    report = report_of(config, live_id, back_id)

    assert len(report["worst_bars"]["listed"]) == 500
    sides = {g["open_time"]: g["gap_side"] for g in report["one_sided_gaps"]}
    assert sides == {
        backfill[7].open_time.isoformat(): "live",
        backfill[9].open_time.isoformat(): "backfill",
    }
    assert report["summary"]["gap_in_one_only"] == 2
    assert report["summary"]["matched"] == count - 2


def test_overlap_is_matched_on_open_time_with_a_weekly_breakdown(tmp_path):
    config = make_config(tmp_path)
    live = [bar(i) for i in range(1, 6)]  # 1..5
    backfill = [bar(i) for i in range(0, 5)]  # 0..4: overlap is 1..4
    live_id, back_id = datasets(config, live, backfill)

    report = report_of(config, live_id, back_id)

    assert report["summary"]["matched"] == 4
    assert report["overlap"]["start"] == bar(1).open_time.isoformat()
    assert report["weeks"] == {
        "2024-01-07": {
            "matched": 4,
            "within": 4,
            "beyond": 0,
            "live_only": 0,
            "backfill_only": 0,
            "gap_in_one_only": 0,
        }
    }


def tree_digest(root):
    digest = hashlib.sha256()
    for path in sorted(p for p in (root / "datasets").rglob("*") if p.is_file()):
        digest.update(str(path).encode() + path.read_bytes())
    return digest.hexdigest()


def test_the_diff_never_modifies_either_dataset(tmp_path):
    config = make_config(tmp_path)
    series = [bar(i) for i in range(5)]
    live_id, back_id = datasets(config, series, series)
    before = tree_digest(config.paths.data_root)

    reconcile(config, live_id, back_id)

    assert tree_digest(config.paths.data_root) == before


def test_mismatched_bar_size_instrument_or_schema_version_is_refused(tmp_path):
    config = make_config(tmp_path)
    series = [bar(i) for i in range(5)]
    live_id, back_id = datasets(config, series, series)
    ten = write_dataset(config, 10, {"EURUSD": series}).dataset_id
    other = write_dataset(
        config, 5, {"GBPUSD": [bar(i, "GBPUSD") for i in range(5)]}
    ).dataset_id
    manifest_path = config.paths.data_root / "datasets" / back_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 99
    manifest_path.write_text(json.dumps(manifest))
    before = tree_digest(config.paths.data_root)

    for live, backfill in ((live_id, ten), (live_id, other), (live_id, back_id)):
        with pytest.raises(ReconciliationRefused):
            reconcile(config, live, backfill)

    assert tree_digest(config.paths.data_root) == before
    assert not (config.paths.data_root / "reports/reconciliation").exists()


def test_cli_diff_writes_the_report_and_refuses_with_exit_1(tmp_path, capsys):
    config = make_config(tmp_path)
    series = [bar(i) for i in range(5)]
    live_id, back_id = datasets(config, series, series)
    ten = write_dataset(config, 10, {"EURUSD": series}).dataset_id
    base = ["ingest", "diff", "--config", str(tmp_path / "ingest.toml")]

    assert main([*base, "--live", live_id, "--backfill", back_id]) == 0
    assert f"reconciliation-{live_id}-{back_id}.json" in capsys.readouterr().out
    assert main([*base, "--live", live_id, "--backfill", ten]) == 1
    assert "bar size" in capsys.readouterr().err
