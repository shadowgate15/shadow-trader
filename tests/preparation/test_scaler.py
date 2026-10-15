"""Scaler: train-only fit, hash-addressed artifact and ``Scaler.apply``."""

import json
import statistics
from dataclasses import replace
from datetime import timedelta

import polars as pl
import pyarrow.parquet as pq
import pytest

from shadow_trader.preparation import (
    SPECS,
    BuildFailure,
    Scaler,
    ScalerError,
    build_prepared,
    load_prepared_config,
    load_scaler,
    open_prepared,
)
from shadow_trader.preparation import build as build_module
from shadow_trader.preparation import scaler as scaler_module

from .synthetic import (
    SESSION_2,
    prepared_config,
    write_ingestion,
    write_prepared_toml,
)

FEATURE = "feat_ret_1"


def spec_hash_of(scaler):  # noqa: N802 - the frame's spec hash, as its manifest records it
    return scaler.artifact["spec_hash"]


def build(tmp_path, version=1):
    parent = write_ingestion(tmp_path)
    path = write_prepared_toml(tmp_path, parent.dataset_id)
    path.write_text(path.read_text().replace("version = 1", f"version = {version}"))
    config = load_prepared_config(path)
    return parent, config, build_prepared(config)


def dataset(config, ref, role):
    return open_prepared(config.paths.prepared_root, ref.datasets[role].dataset_id)


def built_scaler(tmp_path, version=1):
    _, config, ref = build(tmp_path, version)
    scaler_id = dataset(config, ref, "train").manifest["scaler"]["id"]
    return config, ref, load_scaler(config.paths.prepared_root, scaler_id)


def test_scaler_stats_are_robust_median_and_iqr_of_train_non_null_values(tmp_path):
    config, ref, scaler = built_scaler(tmp_path)

    values = dataset(config, ref, "train").read("EURUSD")[FEATURE].drop_nulls()
    stats = scaler.artifact["instruments"]["EURUSD"]["columns"][FEATURE]

    assert stats["method"] == "robust_clip"
    assert stats["n_fit"] == len(values)
    assert stats["median"] == pytest.approx(statistics.median(values.to_list()))
    p25, _, p75 = statistics.quantiles(values.to_list(), n=4, method="inclusive")
    assert stats["iqr"] == pytest.approx(p75 - p25)
    assert scaler.artifact["fit_dataset"] == ref.datasets["train"].dataset_id
    assert scaler.artifact["spec_hash"] == config.feature_spec.hash
    assert scaler.artifact["spec_version"] == config.feature_spec.version
    assert scaler.artifact["anomaly_policy_hash"] == config.anomaly.hash


def test_train_manifest_names_the_scaler_and_the_others_reference_its_hash(tmp_path):
    _, config, ref = build(tmp_path)
    ids = {r: dataset(config, ref, r).manifest["scaler"]["id"] for r in ref.datasets}

    assert len(set(ids.values())) == 1
    assert dataset(config, ref, "train").manifest["scaler"]["fit"] is True
    assert dataset(config, ref, "validation").manifest["scaler"]["fit"] is False
    assert dataset(config, ref, "test").manifest["scaler"]["fit"] is False
    assert ref.record["scaler"] == ids["train"]
    assert (config.paths.prepared_root / "scalers" / f"{ids['train']}.json").is_file()


def test_prepared_datasets_still_carry_unscaled_features(tmp_path):
    _, config, ref = build(tmp_path)
    frame = dataset(config, ref, "validation").read("EURUSD")

    assert frame[FEATURE].drop_nulls().max() == pytest.approx(0.6931471805599453)


def test_apply_scales_robustly_clips_and_preserves_nulls(tmp_path):
    config, ref, scaler = built_scaler(tmp_path)
    frame = dataset(config, ref, "validation").read("EURUSD")
    stats = scaler.artifact["instruments"]["EURUSD"]["columns"][FEATURE]

    scaled = scaler.apply(frame, spec_hash=spec_hash_of(scaler))

    assert isinstance(scaled, pl.DataFrame)
    assert scaled.columns == frame.columns
    assert scaled[FEATURE].null_count() == frame[FEATURE].null_count()
    raw = frame[FEATURE][5]
    expected = max(-5.0, min(5.0, (raw - stats["median"]) / stats["iqr"]))
    assert scaled[FEATURE][5] == pytest.approx(expected)
    assert scaled[FEATURE].drop_nulls().abs().max() <= 5.0
    assert scaled["close"].equals(frame["close"])


def test_apply_returns_a_lazyframe_for_a_lazyframe(tmp_path):
    config, ref, scaler = built_scaler(tmp_path)
    frame = dataset(config, ref, "validation").read("EURUSD")

    lazy = scaler.apply(frame.lazy(), spec_hash=spec_hash_of(scaler))

    assert isinstance(lazy, pl.LazyFrame)
    assert lazy.collect().equals(scaler.apply(frame, spec_hash=spec_hash_of(scaler)))


def test_apply_refuses_a_spec_hash_mismatch(tmp_path):
    config, ref, scaler = built_scaler(tmp_path)
    frame = dataset(config, ref, "validation").read("EURUSD")

    with pytest.raises(ScalerError, match="spec hash"):
        scaler.apply(frame, spec_hash="0" * 64)
    scaler.apply(frame, spec_hash=scaler.artifact["spec_hash"])  # matches: fine
    with pytest.raises(ScalerError, match="columns"):
        scaler.apply(frame.drop(FEATURE), spec_hash=spec_hash_of(scaler))


def test_time_sin_cos_and_rollover_are_identity_and_the_rest_robust(tmp_path):
    config, ref, scaler = built_scaler(tmp_path, version=2)
    columns = scaler.artifact["instruments"]["EURUSD"]["columns"]

    identity = {n for n, c in columns.items() if c["method"] == "identity"}
    assert identity == {
        "feat_time_hour_sin",
        "feat_time_hour_cos",
        "feat_time_dow_sin",
        "feat_time_dow_cos",
        "feat_time_rollover",
    }
    assert columns["feat_time_since_open"]["method"] == "robust_clip"
    assert columns["feat_time_hour_sin"] == {"method": "identity"}

    frame = dataset(config, ref, "validation").read("EURUSD")
    scaled = scaler.apply(frame, spec_hash=spec_hash_of(scaler))
    for name in identity:
        assert scaled[name].equals(frame[name])


def test_zero_iqr_fails_the_build_loudly_and_writes_nothing(tmp_path):
    parent = write_ingestion(tmp_path, wiggle=False)  # constant returns
    config = prepared_config(tmp_path, parent.dataset_id)

    with pytest.raises(BuildFailure) as raised:
        build_prepared(config)

    assert raised.value.code == "scaler_zero_iqr"
    assert FEATURE in str(raised.value)
    assert not (tmp_path / "prepared").exists()


def test_too_few_fit_values_fail_the_build(tmp_path, monkeypatch):
    monkeypatch.setattr(scaler_module, "MIN_N_FIT", 10_000)
    parent = write_ingestion(tmp_path)
    config = prepared_config(tmp_path, parent.dataset_id)

    with pytest.raises(BuildFailure) as raised:
        build_prepared(config)

    assert raised.value.code == "scaler_insufficient_fit"
    assert not (tmp_path / "prepared").exists()


def test_load_scaler_refuses_a_tampered_artifact(tmp_path):
    _, config, ref = build(tmp_path)
    scaler_id = ref.record["scaler"]
    path = config.paths.prepared_root / "scalers" / f"{scaler_id}.json"
    artifact = json.loads(path.read_text())
    artifact["scaler_version"] += 1
    path.write_text(json.dumps(artifact))

    with pytest.raises(ScalerError, match="hash"):
        load_scaler(config.paths.prepared_root, scaler_id)


def perturb_from(tmp_path, parent, start):
    """Rewrite the parent's priced Bars from ``start`` on, keeping its id."""
    root = tmp_path / "ingestion" / "datasets" / parent.dataset_id
    for path in root.rglob("*.parquet"):
        table = pq.read_table(path)
        frame = pl.DataFrame(table)
        later = pl.col("open_time") >= start
        frame = frame.with_columns(
            *(
                pl.when(later).then(pl.col(c) * 1.5).otherwise(pl.col(c)).alias(c)
                for c in ("open", "high", "low", "close")
            )
        )
        pq.write_table(frame.to_arrow().cast(table.schema), path)


def test_perturbing_validation_and_test_leaves_train_and_scaler_unchanged(tmp_path):
    parent, config, ref = build(tmp_path)
    scaler = load_scaler(config.paths.prepared_root, ref.record["scaler"])
    # The parent's id is held fixed so the train hash can only move through the
    # data: change every Bar from the validation Session on.
    perturb_from(tmp_path, parent, SESSION_2)
    second_root = tmp_path / "second"
    paths = replace(config.paths, prepared_root=second_root)

    second = build_prepared(replace(config, paths=paths))

    assert second.datasets["train"].dataset_id == ref.datasets["train"].dataset_id
    assert second.record["scaler"] == ref.record["scaler"]
    assert load_scaler(second_root, ref.record["scaler"]).artifact == scaler.artifact
    assert second.datasets["validation"].dataset_id != (
        ref.datasets["validation"].dataset_id
    )
    assert second.datasets["test"].dataset_id != ref.datasets["test"].dataset_id


def test_fit_range_reaching_into_validation_aborts_the_build(tmp_path, monkeypatch):
    real = build_module.fit_scaler

    def reaching(*args, **kwargs):
        artifact = dict(real(*args, **kwargs).artifact)
        start, _ = artifact["fit_range"]
        end = (SESSION_2 + timedelta(minutes=5)).isoformat()
        return Scaler({**artifact, "fit_range": [start, end]})

    monkeypatch.setattr(build_module, "fit_scaler", reaching)
    parent = write_ingestion(tmp_path)
    config = prepared_config(tmp_path, parent.dataset_id)

    with pytest.raises(BuildFailure) as raised:
        build_prepared(config)

    assert raised.value.code == "boundary_check.scaler_fit_range"
    assert not (tmp_path / "prepared").exists()


def test_boundary_check_block_lists_the_scaler_check(tmp_path):
    _, config, ref = build(tmp_path)

    checks = dataset(config, ref, "train").manifest["boundary_check"]["checks"]
    assert "scaler_fit_range" in checks


def test_fit_range_check_compares_against_the_split_bounds_not_the_fit_frame(
    tmp_path,
):
    # A fit range that starts before the train bound lies outside the split even
    # though a frame-derived range would never reveal it.
    from shadow_trader.preparation.boundary import BoundaryContext
    from shadow_trader.preparation.split import RoleSlice, SplitBounds

    _, config, ref = build(tmp_path)
    train = dataset(config, ref, "train")
    scaler = load_scaler(config.paths.prepared_root, train.manifest["scaler"]["id"])
    frame = train.read("EURUSD")
    first = frame["open_time"][0]
    bound = {"train_start": first, "validation_start": SESSION_2, "test_start": None}
    ctx = BoundaryContext(
        SplitBounds(bound, bound),
        {"train": RoleSlice("train", frame, frame.height, 0, None, None)},
        frame["open_time"],
        0,
        SPECS[1],
    )
    assert scaler_module.scaler_fit_range_check(scaler)(ctx) is None

    # The frame itself is unchanged; only the split bound moves past the fit start.
    late = {**bound, "train_start": frame["open_time"][10]}
    moved = BoundaryContext(
        SplitBounds(late, late), ctx.roles, ctx.parent_open_times, 0, SPECS[1]
    )
    detail = scaler_module.scaler_fit_range_check(scaler)(moved)
    assert detail is not None and "train bounds" in detail


def test_fit_matches_train_compares_artifact_to_the_stored_train_rows(tmp_path):
    _, config, ref = build(tmp_path)
    train = dataset(config, ref, "train")
    scaler = load_scaler(config.paths.prepared_root, train.manifest["scaler"]["id"])
    frames = {"EURUSD": train.read("EURUSD")}

    assert scaler_module.fit_matches_train(scaler, frames, train.dataset_id) is None

    other = Scaler({**scaler.artifact, "fit_dataset": "someone-else"})
    detail = scaler_module.fit_matches_train(other, frames, train.dataset_id)
    assert detail is not None and "fit_dataset" in detail
    start = scaler.artifact["fit_range"][0]
    shifted = Scaler({**scaler.artifact, "fit_range": [start, "2030-01-01T00:00:00"]})
    detail = scaler_module.fit_matches_train(shifted, frames, train.dataset_id)
    assert detail is not None and "fit range" in detail


def test_apply_requires_the_spec_hash_unless_explicitly_opted_out(tmp_path):
    config, ref, scaler = built_scaler(tmp_path)
    frame = dataset(config, ref, "validation").read("EURUSD")

    with pytest.raises(TypeError, match="spec_hash"):
        scaler.apply(frame)  # type: ignore[call-arg]
    # None is the explicit opt-out; the feat_* column set is still checked.
    assert scaler.apply(frame, spec_hash=None).columns == frame.columns
