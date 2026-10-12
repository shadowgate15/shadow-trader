"""Build report, failed-build report and console summary (spec #28, section 10)."""

from __future__ import annotations

import json

import pytest

from shadow_trader.ingestion.backfill import BuildFailure, BuildWarning
from shadow_trader.ingestion.dataset import open_dataset
from shadow_trader.ingestion.report import WARNING_CODES
from tests.ingestion.test_backfill import build_csv, build_fixture, crossed_lines

# The warning/failure table of spec #28, section 11, as literals.
SPEC_WARNING_CODES = {
    "out_of_session_ticks",
    "session_alignment",
    "thin_data",
    "anomaly_alert_fraction_exceeded",
    "live_quote_after_seal",
}


def test_warning_codes_are_a_closed_list_covering_the_spec_table():
    assert set(WARNING_CODES) == SPEC_WARNING_CODES
    assert len(set(WARNING_CODES)) == len(WARNING_CODES)


def test_a_warning_with_an_unknown_code_is_rejected():
    with pytest.raises(ValueError, match="unknown warning code 'made_up'"):
        BuildWarning("made_up", "x")
    BuildWarning("thin_data", "ok")


def test_build_failure_carries_a_locator():
    failure = BuildFailure("raw month missing: EURUSD 202401", "EURUSD/202401")
    assert failure.locator == "EURUSD/202401"
    assert BuildFailure("plain").locator is None
    assert "EURUSD" in str(failure)


def read_report(result):
    return json.loads((result.ref.path / "build-report.json").read_text())


def test_build_report_sits_beside_the_manifest_with_counts_per_instrument(tmp_path):
    config, result = build_fixture(tmp_path, "eurusd_winter", "EURUSD", "2024-01")

    report = read_report(result)

    assert report["report_version"] == 1
    assert report["status"] == "ok"
    assert report["dataset_id"] == result.ref.dataset_id
    assert report["bar_size"] == 5
    bars = open_dataset(config.paths.data_root, result.ref.dataset_id).read("EURUSD")
    counts = report["instruments"]["EURUSD"]
    assert counts["bars"] == len(bars)
    assert counts["sessions"] == len({b.session_id for b in bars})
    gaps = [b for b in bars if b.is_gap]
    outage = [b for b in gaps if "suspected_outage" in b.quality_flags]
    assert counts["gap_rows"] == {
        "plain": len(gaps) - len(outage),
        "suspected_outage": len(outage),
        "source_missing": 0,
    }
    assert counts["out_of_session_ticks"] == 0


def test_build_report_has_per_flag_counts_and_the_alert_check(tmp_path):
    _, build = build_csv(tmp_path, "EURUSD", "2024-01", crossed_lines(3))

    flags = read_report(build())["instruments"]["EURUSD"]["flags"]

    assert flags["bar_counts"]["crossed_quote"] == 1
    assert {k: v for k, v in flags.items() if k != "bar_counts"} == {
        "tick_counts": {"crossed_quote": 3},
        "non_gap_bars": 1,
        "alerted_bars": 1,
        "alert_fraction": 0.01,
        "alert_exceeded": True,
    }


@pytest.mark.parametrize(
    ("first_tick", "expected"),
    [
        ("170000", []),
        ("180000", [("2024-01-07", 60, True)]),
        ("183000", [("2024-01-07", 90, False)]),
    ],
)
def test_build_report_lists_only_off_weeks_with_the_observed_offset(
    tmp_path, first_tick, expected
):
    lines = [
        f"20240107 {first_tick}000,1.10000,1.10010,0",
        "20240107 200000000,1.10000,1.10010,0",
    ]
    _, build = build_csv(tmp_path, "EURUSD", "2024-01", lines)

    alignment = read_report(build())["instruments"]["EURUSD"]["session_alignment"]

    assert [
        (a["week"], a["offset_minutes"], a["wrong_offset_signature"]) for a in alignment
    ] == expected


def test_build_report_lists_every_warning_as_a_structured_entry(tmp_path):
    from shadow_trader.ingestion.backfill import build_backfill
    from tests.ingestion.test_backfill import install_raw, make_config

    config = make_config(tmp_path, "EURUSD", "2008-12")
    install_raw(config, "EURUSD", "200812", b"20081207 170000000,1.30000,1.30010,0\n")

    result = build_backfill(config, 5)

    assert read_report(result)["warnings"] == [
        {"code": "thin_data", "message": w.message, "locator": "EURUSD/200812"}
        for w in result.warnings
        if w.code == "thin_data"
    ]
    assert {w["code"] for w in read_report(result)["warnings"]} <= set(WARNING_CODES)
    assert len(read_report(result)["warnings"]) == len(result.warnings)


def test_build_report_references_raw_file_hashes_and_the_resolved_config(tmp_path):
    config, result = build_fixture(tmp_path, "eurusd_winter", "EURUSD", "2024-01")

    report = read_report(result)

    assert report["raw_inputs"] == result.ref.manifest["raw_inputs"] != {}
    assert report["config"]["build"]["instruments"] == ["EURUSD"]
    assert report["config"]["build"]["bar_sizes"] == [5]
    assert report["config"]["instruments"]["EURUSD"]["price_scale"] == 5
    assert report["config"]["rules"]["anomaly"]["alert_fraction"] == 0.01
    assert report["config"]["paths"]["data_root"] == str(config.paths.data_root)


def test_reporting_only_changes_alter_the_report_but_never_the_dataset_id(tmp_path):
    import dataclasses

    from shadow_trader.ingestion.backfill import build_backfill

    config, first = build_csv(tmp_path, "EURUSD", "2024-01", crossed_lines(3))
    first = first()
    loose = dataclasses.replace(
        config,
        anomaly_rule=dataclasses.replace(config.anomaly_rule, alert_fraction=1.0),
    )

    second = build_backfill(loose, 5)

    assert second.ref.dataset_id == first.ref.dataset_id
    assert second.ref.manifest == first.ref.manifest
    flags = read_report(second)["instruments"]["EURUSD"]["flags"]
    assert flags["alert_exceeded"] is False
    assert flags["alert_fraction"] == 1.0
    assert not any("report" in p["path"] for p in second.ref.manifest["partitions"])


def failed_reports(tmp_path):
    return sorted((tmp_path / "data" / "reports" / "failed").glob("*.json"))


def test_missing_raw_month_writes_a_failure_report_and_no_dataset(tmp_path):
    from shadow_trader.ingestion.backfill import build_backfill
    from tests.ingestion.test_backfill import make_config

    config = make_config(tmp_path, "EURUSD", "2024-01")

    with pytest.raises(BuildFailure) as raised:
        build_backfill(config, 5)

    [path] = failed_reports(tmp_path)
    report = json.loads(path.read_text())
    assert report["report_version"] == 1
    assert report["status"] == "failed"
    assert "raw month missing" in report["condition"]
    assert report["locator"] == "EURUSD/202401"
    assert report["bar_size"] == 5
    assert raised.value.report_path == path
    assert not (tmp_path / "data" / "datasets").exists()


@pytest.mark.parametrize(
    ("lines", "locator"),
    [
        (["20240107 170000000,1.1,1.1001,0", "garbage"], "EURUSD/202401 row 2"),
        (
            ["20240107 170000000,1.1,1.1001,0", "20240107 170001000,1.100005,1.1,0"],
            "EURUSD/202401 row 2",
        ),
    ],
)
def test_row_failures_locate_the_file_and_row(tmp_path, lines, locator):
    month = "2024-03" if locator.endswith("202403 row 1") else "2024-01"
    _, build = build_csv(tmp_path, "EURUSD", month, lines)

    with pytest.raises(BuildFailure):
        build()

    [path] = failed_reports(tmp_path)
    assert json.loads(path.read_text())["locator"] == locator


def test_console_summary_has_one_line_per_warning_code_then_the_path():
    from pathlib import Path

    from shadow_trader.ingestion.report import console_summary

    warnings = [
        {"code": "thin_data", "message": "a", "locator": "EURUSD/200812"},
        {"code": "session_alignment", "message": "b", "locator": "EURUSD/week-1"},
        {"code": "thin_data", "message": "c", "locator": "EURUSD/200901"},
    ]
    report = {"status": "ok", "warnings": warnings}

    assert console_summary(report, Path("/x/build-report.json")) == [
        "warning: thin_data x2",
        "warning: session_alignment x1",
        "report: /x/build-report.json",
    ]
    assert console_summary({"status": "ok", "warnings": []}, Path("/y.json")) == [
        "report: /y.json"
    ]


def test_console_summary_of_a_failure_names_condition_and_locator():
    from pathlib import Path

    from shadow_trader.ingestion.report import console_summary

    report = {"status": "failed", "condition": "boom", "locator": "EURUSD/202401"}

    assert console_summary(report, Path("/f.json")) == [
        "build failed: boom (EURUSD/202401)",
        "report: /f.json",
    ]


def test_cli_summarises_a_warnings_only_build_from_its_json_and_exits_0(
    tmp_path, capsys
):
    from shadow_trader import main
    from shadow_trader.ingestion.report import console_summary
    from tests.ingestion.test_backfill import cli_config, install_raw

    path, config = cli_config(tmp_path, "2008-12", "[5]")
    install_raw(config, "EURUSD", "200812", b"20081207 170000000,1.30000,1.30010,0\n")

    assert main(["ingest", "build", "--config", str(path)]) == 0

    out = capsys.readouterr().out.splitlines()
    [dataset] = (config.paths.data_root / "datasets").iterdir()
    artifact = dataset / "build-report.json"
    expected = console_summary(json.loads(artifact.read_text()), artifact)
    assert expected[0] == "warning: thin_data x1"
    assert out[-len(expected) :] == expected
    assert not (dataset / "console-summary.txt").exists()


def test_cli_failed_build_exits_1_names_the_failure_report_and_leaves_no_dataset(
    tmp_path, capsys
):
    from shadow_trader import main
    from tests.ingestion.test_backfill import cli_config

    path, config = cli_config(tmp_path)

    assert main(["ingest", "build", "--config", str(path)]) == 1

    [report] = (config.paths.data_root / "reports" / "failed").glob("*.json")
    err = capsys.readouterr().err
    assert "raw month missing" in err
    assert f"report: {report}" in err
    assert not (config.paths.data_root / "datasets").exists()


def gap_run_series(runs):
    from datetime import UTC, datetime, timedelta

    from shadow_trader.ingestion.bars import Bar

    opened = datetime(2024, 1, 7, 22, 0, tzinfo=UTC)
    bars = []
    for i in range(runs):
        for offset, gap in ((0, True), (1, False)):
            bars.append(
                Bar(
                    instrument="EURUSD",
                    open_time=opened + timedelta(minutes=5 * (2 * i + offset)),
                    open=None if gap else 1.0,
                    high=None if gap else 1.0,
                    low=None if gap else 1.0,
                    close=None if gap else 1.0,
                    spread_close=None if gap else 0.0001,
                    spread_mean=None if gap else 0.0001,
                    tick_count=None if gap else 1,
                    is_gap=gap,
                    session_id=opened,
                    is_rollover=False,
                    quality_flags=("suspected_outage",) if gap else (),
                )
            )
    return bars


def test_outage_runs_are_capped_at_200_with_the_exact_total():
    from shadow_trader.ingestion.report import build_report

    report = build_report(
        dataset_id="x",
        bar_size=5,
        series={"EURUSD": gap_run_series(250)},
        dropped_out_of_session={},
    )

    runs = report["instruments"]["EURUSD"]["suspected_outage_runs"]
    assert runs["total"] == 250
    assert len(runs["listed"]) == 200
    assert runs["listed"][0] == {
        "start": "2024-01-07T22:00:00+00:00",
        "end": "2024-01-07T22:05:00+00:00",
        "duration_minutes": 5,
    }
