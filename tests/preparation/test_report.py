import json

import pytest

from shadow_trader.preparation import BuildFailure, build_prepared
from shadow_trader.preparation.config import ReportConfig
from shadow_trader.preparation.report import advisory_warnings

from .synthetic import (
    EMBARGO,
    SESSION_1,
    iso,
    prepared_config,
    split_toml,
    write_ingestion,
)

QUIET = (
    "[report]\nmin_usable_fraction = 0.0\nmin_rows = 1\n"
    "max_distrusted_flag_fraction = 1.0\nmax_scaler_clip_fraction = 1.0\n"
)


def built(tmp_path, extra="", **kwargs):
    parent = write_ingestion(tmp_path, **kwargs)
    config = prepared_config(tmp_path, parent.dataset_id, extra)
    ref = build_prepared(config)
    report_path = config.paths.prepared_root / "builds" / f"{ref.build_id}.report.json"
    return config, ref, json.loads(report_path.read_text())


def test_report_sits_beside_the_build_record_and_describes_the_run(tmp_path):
    _, ref, report = built(tmp_path)

    assert report["report_version"] == 1
    assert report["build_id"] == ref.build_id
    assert report["created_at"]
    assert set(report["durations"]) >= {"total_seconds", "features_seconds"}
    assert report["throughput"]["rows_per_second"] > 0
    assert report["threads"] >= 1
    assert set(report["roles"]) == {"train", "validation", "test"}
    train = report["roles"]["train"]
    manifest = ref.datasets["train"].manifest
    assert train["rows"] == manifest["row_count"]
    assert train["instruments"]["EURUSD"]["rows"] == manifest["row_count"]
    assert train["usable"] == manifest["anomaly_policy"]["usable_count"]
    assert train["usable_fraction"] == pytest.approx(train["usable"] / train["rows"])
    assert train["unusable_reasons"]["gap"] == 1
    assert train["null_counts"]["feat_ret_1"] > 0
    assert "feat_ret_288" in train["null_counts"]
    assert report["manifests"]["train"].endswith(
        f"{ref.datasets['train'].dataset_id}/manifest.json"
    )
    assert isinstance(report["warnings"], list)


def test_rerunning_an_identical_build_overwrites_the_report_atomically(tmp_path):
    config, ref, first = built(tmp_path)
    again = build_prepared(config)

    path = config.paths.prepared_root / "builds" / f"{ref.build_id}.report.json"
    assert again.build_id == ref.build_id
    assert json.loads(path.read_text())["created_at"] >= first["created_at"]
    assert not list(path.parent.glob("*.tmp"))


def test_reports_never_change_hashes(tmp_path):
    parent = write_ingestion(tmp_path)
    plain = build_prepared(prepared_config(tmp_path, parent.dataset_id))
    tuned = build_prepared(
        prepared_config(
            tmp_path,
            parent.dataset_id,
            extra="[report]\nmin_usable_fraction = 0.99\nmin_rows = 10\n",
        )
    )

    assert tuned.build_id == plain.build_id
    assert {r: d.dataset_id for r, d in tuned.datasets.items()} == {
        r: d.dataset_id for r, d in plain.datasets.items()
    }


def codes(report):
    return {w["code"] for w in report["warnings"]}


def test_thresholds_in_the_report_table_drive_advisory_warnings(tmp_path):
    _, ref, report = built(
        tmp_path,
        extra=(
            "[report]\nmin_usable_fraction = 0.9999\nmin_rows = 100000\n"
            "max_distrusted_flag_fraction = 0.0\nmax_scaler_clip_fraction = 0.0\n"
        ),
        flags={50: ("stale_quote",)},
    )

    assert {
        "low_usable_fraction",
        "short_dataset",
        "high_distrusted_flag_fraction",
    } <= codes(report)
    assert ref.build_id  # warnings never fail the build


def test_quiet_thresholds_give_no_warnings(tmp_path):
    _, _, report = built(tmp_path, extra=QUIET)

    assert report["warnings"] == []


def test_an_instrument_with_no_usable_bars_in_a_role_warns():
    roles = {
        "validation": {
            "rows": 20,
            "usable": 10,
            "usable_fraction": 0.5,
            "unusable_reasons": {"gap": 10},
            "instruments": {
                "EURUSD": {"rows": 10, "usable": 10},
                "GBPUSD": {"rows": 10, "usable": 0},
            },
        }
    }

    found = advisory_warnings(roles, {}, ReportConfig(min_rows=1))

    assert [w["code"] for w in found if w["code"] == "all_unusable_instrument"] == [
        "all_unusable_instrument"
    ]
    assert "GBPUSD" in found[0]["message"]


def test_failed_build_writes_a_report_outside_the_store_naming_a_closed_code(
    tmp_path,
):
    parent = write_ingestion(tmp_path)
    config = prepared_config(
        tmp_path, parent.dataset_id, split=split_toml(episode_bars=5000)
    )

    with pytest.raises(BuildFailure) as raised:
        build_prepared(config)

    root = config.paths.prepared_root
    path = raised.value.report_path
    assert path is not None and path.parent == root.with_name("prepared.failed")
    report = json.loads(path.read_text())
    assert report["status"] == "failed"
    assert report["code"] == raised.value.code == "embargo_exceeds_dataset"
    assert report["inputs"]["parent_dataset"] == parent.dataset_id
    assert report["inputs"]["split"]["train_start"] == iso(SESSION_1)
    assert report["inputs"]["embargo_episode_bars"] == 5000
    assert report["counts"]["instruments"] == 1
    assert report["counts"]["rows"]["EURUSD"] > EMBARGO
    for store in ("datasets", "builds", "scalers"):
        assert not (root / store).exists()


def test_bar_size_mismatch_and_missing_parent_have_closed_codes(tmp_path):
    parent = write_ingestion(tmp_path, bar_size=15)
    with pytest.raises(BuildFailure) as mismatch:
        build_prepared(prepared_config(tmp_path, parent.dataset_id))
    with pytest.raises(BuildFailure) as missing:
        build_prepared(prepared_config(tmp_path, "deadbeef"))

    assert mismatch.value.code == "bar_size_mismatch"
    assert missing.value.code == "parent_unavailable"
    assert missing.value.report_path is not None
    assert json.loads(missing.value.report_path.read_text())["code"] == (
        "parent_unavailable"
    )


class Recorder:
    def __init__(self):
        self.events = []

    def start(self, phase, total=None):
        self.events.append(("start", phase, total))

    def advance(self, detail="", outcome=None):
        self.events.append(("advance", detail))

    def warn(self, message):
        self.events.append(("warn", message))

    def status(self, text):
        self.events.append(("status", text))

    def done(self):
        self.events.append(("done",))


def test_build_emits_phase_and_per_instrument_events_and_never_prints(tmp_path, capsys):
    parent = write_ingestion(tmp_path, instruments=("EURUSD", "GBPUSD"))
    sink = Recorder()

    build_prepared(prepared_config(tmp_path, parent.dataset_id, QUIET), progress=sink)

    assert ("start", "features", 2) in sink.events
    assert ("advance", "EURUSD") in sink.events
    assert ("advance", "GBPUSD") in sink.events
    assert ("start", "write", 3) in sink.events
    starts = [e for e in sink.events if e[0] == "start"]
    assert sink.events.count(("done",)) == len(starts)
    out = capsys.readouterr()
    assert out.out == "" and out.err == ""
