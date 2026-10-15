"""Split, Embargo and build record, observed through build_prepared/open_prepared."""

import json
from dataclasses import replace
from datetime import timedelta

import polars as pl
import pytest

from shadow_trader.ingestion.dataset import open_dataset
from shadow_trader.preparation import (
    SPECS,
    BuildFailure,
    FeatureSpec,
    build_prepared,
    open_prepared,
)
from shadow_trader.preparation import build as build_module
from shadow_trader.preparation.config import SpecConfig
from shadow_trader.preparation.spec import FeatureColumn

from .synthetic import (
    BARS_PER_SESSION,
    EMBARGO,
    SESSION_1,
    SESSION_2,
    SESSION_3,
    prepared_config,
    split_toml,
    write_ingestion,
)

BAR = timedelta(minutes=5)
ROLES = ("train", "validation", "test")
KEPT = BARS_PER_SESSION - EMBARGO


def build(tmp_path, split=None, **ingest):
    tmp_path.mkdir(parents=True, exist_ok=True)
    parent = write_ingestion(tmp_path, **ingest)
    config = prepared_config(tmp_path, parent.dataset_id, split=split)
    ref = build_prepared(config)
    opened = {
        role: open_prepared(config.paths.prepared_root, ref.datasets[role].dataset_id)
        for role in ROLES
    }
    return parent, config, ref, opened


def test_one_build_emits_train_validation_and_test_datasets(tmp_path):
    parent, config, ref, opened = build(tmp_path)

    assert set(ref.datasets) == set(ROLES)
    assert len({d.dataset_id for d in ref.datasets.values()}) == 3
    for role, dataset in opened.items():
        assert dataset.manifest["role"] == role
        assert dataset.manifest["build_id"] == ref.build_id
        assert dataset.manifest["parent_ingestion"] == parent.dataset_id
        assert "embargoed" not in dataset.read("EURUSD").columns
    # Trimmed Bars stay in the parent.
    parent_manifest = open_dataset(config.paths.data_root, parent.dataset_id).manifest
    assert parent_manifest["row_count"] == 3 * BARS_PER_SESSION


def test_last_E_bars_are_trimmed_off_train_and_validation_only(tmp_path):
    *_, opened = build(tmp_path)

    train = opened["train"].read("EURUSD")
    validation = opened["validation"].read("EURUSD")
    test = opened["test"].read("EURUSD")

    assert train.height == validation.height == KEPT
    assert test.height == BARS_PER_SESSION  # no tail trim
    assert train["open_time"][0] == SESSION_1  # later head not trimmed
    assert validation["open_time"][0] == SESSION_2
    assert test["open_time"][0] == SESSION_3
    assert train["open_time"][-1] == SESSION_1 + (KEPT - 1) * BAR
    assert validation["open_time"][-1] == SESSION_2 + (KEPT - 1) * BAR
    assert test["open_time"][-1] == SESSION_3 + (BARS_PER_SESSION - 1) * BAR


def test_manifest_records_split_and_embargo(tmp_path):
    *_, opened = build(tmp_path)

    train = opened["train"].manifest
    assert train["split"]["scheme"] == "chronological"
    assert train["split"]["bounds"] == {
        "train_start": SESSION_1.isoformat(),
        "validation_start": SESSION_2.isoformat(),
        "test_start": SESSION_3.isoformat(),
        "test_end": None,
    }
    assert train["embargo"] == {
        "episode_bars": 2,
        "longest_lookback": 288,
        "bars": EMBARGO,
        "nominal_end": SESSION_2.isoformat(),
        "effective_end": (SESSION_1 + KEPT * BAR).isoformat(),
        "trimmed_bars": EMBARGO,
    }
    test = opened["test"].manifest["embargo"]
    assert test["trimmed_bars"] == 0
    assert test["bars"] == EMBARGO
    assert opened["validation"].manifest["embargo"]["trimmed_bars"] == EMBARGO


def test_bounds_snap_forward_to_the_next_session_open(tmp_path):
    split = split_toml(
        train_start=SESSION_1 - timedelta(days=2),
        validation_start=SESSION_1 + timedelta(days=1),
        test_start=SESSION_2 + timedelta(hours=1),
    )
    *_, opened = build(tmp_path, split=split)

    manifest = opened["train"].manifest["split"]
    assert (
        manifest["bounds"]["validation_start"]
        == (SESSION_1 + timedelta(days=1)).isoformat()
    )
    assert manifest["snapped_bounds"] == {
        "train_start": SESSION_1.isoformat(),
        "validation_start": SESSION_2.isoformat(),
        "test_start": SESSION_3.isoformat(),
        "test_end": None,
    }
    assert opened["validation"].read("EURUSD")["open_time"][0] == SESSION_2
    assert opened["test"].read("EURUSD")["open_time"][0] == SESSION_3


def test_test_end_bounds_the_test_range(tmp_path):
    session_4 = SESSION_3 + timedelta(days=7)
    sessions = (SESSION_1, SESSION_2, SESSION_3, session_4)
    split = split_toml(test_end=session_4)
    *_, opened = build(tmp_path, split=split, sessions=sessions)

    test = opened["test"].read("EURUSD")
    assert test.height == BARS_PER_SESSION
    assert test["session_id"].unique().to_list() == [SESSION_3]
    assert opened["test"].manifest["embargo"]["nominal_end"] == session_4.isoformat()


# --- hashes, siblings and the build record -------------------------------------


def test_build_record_lists_dataset_ids_and_shared_refs(tmp_path):
    parent, config, ref, _ = build(tmp_path)
    spec = config.feature_spec

    path = config.paths.prepared_root / "builds" / f"{ref.build_id}.json"
    assert ref.path == path
    record = json.loads(path.read_text())
    assert record["build_id"] == ref.build_id
    assert record["datasets"] == {r: ref.datasets[r].dataset_id for r in ROLES}
    assert record["parent_ingestion"] == parent.dataset_id
    assert record["spec_hash"] == spec.hash
    assert record["policy_hash"] == config.anomaly.hash
    assert record["split"]["scheme"] == "chronological"
    assert record["split"]["snapped_bounds"]["test_start"] == SESSION_3.isoformat()
    assert record["embargo_bars"] == EMBARGO
    assert record["boundary_check"]["passed"] is True
    assert record["polars_version"] == pl.__version__


def test_manifests_name_their_siblings_and_carry_the_boundary_check(tmp_path):
    _, _, ref, opened = build(tmp_path)

    for role, dataset in opened.items():
        others = {r: ref.datasets[r].dataset_id for r in ROLES if r != role}
        assert dataset.manifest["siblings"] == others
        assert dataset.manifest["boundary_check"] == ref.record["boundary_check"]
    assert opened["train"].manifest["boundary_check"]["checks"] == [
        "ordering",
        "session_open",
        "dataset_length",
        "usable_bars",
        "gap_between_splits",
        "window_before_dataset_start",
    ]


def test_rebuilding_gives_the_same_ids(tmp_path):
    _, config, first, _ = build(tmp_path)
    second = build_prepared(config)

    assert second.build_id == first.build_id
    assert {r: d.dataset_id for r, d in second.datasets.items()} == {
        r: d.dataset_id for r, d in first.datasets.items()
    }


def test_hash_covers_embargo_bounds_and_role(tmp_path):
    _, _, base, _ = build(tmp_path / "a")
    _, _, longer, _ = build(tmp_path / "b", split=split_toml(episode_bars=3))
    # Same rows and snapped bounds, different nominal bound.
    _, _, nominal, _ = build(
        tmp_path / "c", split=split_toml(test_start=SESSION_3 - timedelta(hours=1))
    )

    ids = {r: d.dataset_id for r, d in base.datasets.items()}
    assert len(set(ids.values())) == 3  # role is hashed: no two Datasets collide
    for other in (longer, nominal):
        for role in ROLES:
            assert other.datasets[role].dataset_id != ids[role]
        assert other.build_id != base.build_id


def test_later_data_cannot_change_an_earlier_dataset(tmp_path):
    sessions = (SESSION_1, SESSION_2, SESSION_3, SESSION_3 + timedelta(days=7))
    *_, short = build(tmp_path / "a")
    *_, longer = build(tmp_path / "b", sessions=sessions)

    # A grown parent gets new ids (the parent id is hashed) but the rows of the
    # earlier splits do not move.
    for role in ("train", "validation"):
        assert longer[role].read("EURUSD").equals(short[role].read("EURUSD"))
    assert longer["test"].read("EURUSD").height > short["test"].read("EURUSD").height


# --- negative tests: each aborts with nothing written ---------------------------


def attempt(tmp_path, split=None, config_edit=None, **ingest):
    parent = write_ingestion(tmp_path, **ingest)
    config = prepared_config(tmp_path, parent.dataset_id, split=split)
    if config_edit is not None:
        config = config_edit(config)
    with pytest.raises(BuildFailure) as raised:
        build_prepared(config)
    assert not (tmp_path / "prepared").exists()  # atomic: nothing written
    return raised.value


def test_overlapping_bounds_abort_naming_the_ordering_check(tmp_path):
    failure = attempt(tmp_path, split=split_toml(validation_start=SESSION_1))

    assert failure.code == "boundary_check.ordering"
    assert "boundary_check.ordering" in str(failure)


def test_bounds_snapping_to_the_same_session_open_abort(tmp_path):
    # validation_start and test_start both snap forward to the SESSION_2 open.
    split = split_toml(
        validation_start=SESSION_1 + timedelta(hours=1),
        train_start=SESSION_1 - timedelta(hours=1),
        test_start=SESSION_2 - timedelta(hours=1),
    )
    failure = attempt(tmp_path, split=split)

    assert failure.code == "boundary_check.ordering"


def test_a_start_bound_after_the_last_session_aborts(tmp_path):
    failure = attempt(
        tmp_path, split=split_toml(test_start=SESSION_3 + timedelta(days=1))
    )

    assert failure.code == "boundary_check.session_open"


def test_gap_between_splits_below_E_aborts(tmp_path, monkeypatch):
    real = build_module.split_frame
    monkeypatch.setattr(
        build_module, "split_frame", lambda frame, bounds, e: real(frame, bounds, e - 1)
    )

    failure = attempt(tmp_path)

    assert failure.code == "boundary_check.gap_between_splits"
    assert f"E={EMBARGO}" in str(failure)


def test_window_reaching_before_a_dataset_start_aborts(tmp_path, monkeypatch):
    leaky = FeatureColumn(
        name="feat_ret_1",
        family="returns",
        definition="ln(close / close.shift(1)) across Sessions (deliberately wrong)",
        lookback=1,
        expression=lambda: (pl.col("close") / pl.col("close").shift(1)).log(),
    )
    monkeypatch.setitem(SPECS, 99, FeatureSpec(99, 5, (leaky,)))

    failure = attempt(
        tmp_path, config_edit=lambda c: replace(c, spec=SpecConfig(version=99))
    )

    assert failure.code == "boundary_check.window_before_dataset_start"
    assert "validation" in str(failure)


def test_embargo_longer_than_a_dataset_aborts(tmp_path):
    failure = attempt(tmp_path, split=split_toml(episode_bars=BARS_PER_SESSION))

    assert failure.code == "embargo_exceeds_dataset"


def test_a_dataset_without_usable_bars_after_the_trim_aborts(tmp_path):
    # E = 408 leaves 292 train Bars, all still inside the Gap's 288-Bar window.
    failure = attempt(tmp_path, split=split_toml(episode_bars=120))

    assert failure.code == "no_usable_bars"
    assert "train" in str(failure)


def test_extra_checks_plug_in_after_the_built_in_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(
        build_module,
        "EXTRA_CHECKS",
        (("scaler_fit_range", lambda ctx: "fit range reaches into validation"),),
    )

    failure = attempt(tmp_path)

    assert failure.code == "boundary_check.scaler_fit_range"
