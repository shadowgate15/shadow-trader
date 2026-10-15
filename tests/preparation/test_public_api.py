"""The Python API and CLI fixed by the spec (#73), imported from the package root."""

from __future__ import annotations

import inspect

import pytest

import shadow_trader.preparation as preparation
from shadow_trader import main
from shadow_trader.preparation import (
    FEATURE_SPEC,
    SPECS,
    FeatureSpec,
    LiveFeatureSpec,
    Scaler,
    ScalerError,
    build_prepared,
    check_compat,
    check_tail_parity,
    compute_features,
    evaluate_tail,
    load_prepared_config,
    load_scaler,
    open_prepared,
)

from .synthetic import write_ingestion, write_prepared_toml


def test_the_spec_named_api_is_exported_from_the_package_root():
    for name in (
        "build_prepared",
        "open_prepared",
        "load_scaler",
        "compute_features",
        "FeatureSpec",
        "evaluate_tail",
        "check_compat",
        "check_tail_parity",
        "Scaler",
    ):
        assert name in preparation.__all__
        assert hasattr(preparation, name)


def test_scaler_apply_is_a_public_method():
    assert callable(Scaler.apply)
    assert "spec_hash" in inspect.signature(Scaler.apply).parameters


def test_feature_spec_names_the_definition_and_live_spec_adds_the_tail_size():
    # The spec (story 56) says FeatureSpec exposes version, hash, per-column
    # warm-up and min_tail_bars. The definition class FeatureSpec holds version,
    # hash and lookbacks; the live-facing LiveFeatureSpec (the FEATURE_SPEC
    # constant) adds the per-column warm-up and min_tail_bars.
    spec = SPECS[FEATURE_SPEC.version]
    assert isinstance(spec, FeatureSpec)
    assert isinstance(FEATURE_SPEC, LiveFeatureSpec)
    assert FEATURE_SPEC.spec is spec
    assert (FEATURE_SPEC.version, FEATURE_SPEC.hash) == (spec.version, spec.hash)
    assert set(FEATURE_SPEC.warm_up) == set(spec.names)
    assert FEATURE_SPEC.min_tail_bars >= spec.longest_lookback
    assert callable(compute_features)
    assert callable(evaluate_tail) and callable(check_compat)
    assert callable(check_tail_parity)


def test_build_open_and_load_scaler_work_from_the_package_root(tmp_path):
    parent = write_ingestion(tmp_path)
    path = write_prepared_toml(tmp_path, parent.dataset_id)
    config = load_prepared_config(path)

    ref = build_prepared(config)

    train = open_prepared(config.paths.prepared_root, ref.datasets["train"].dataset_id)
    assert train.manifest["role"] == "train"
    assert train.read("EURUSD").height
    scaler = load_scaler(config.paths.prepared_root, train.manifest["scaler"]["id"])
    assert scaler.apply(
        train.read("EURUSD"), spec_hash=train.manifest["feature_spec"]["hash"]
    ).columns == train.read("EURUSD").columns
    # fit under spec v1, so the live spec (v3) is refused
    with pytest.raises(ScalerError):
        check_compat(scaler, FEATURE_SPEC)


@pytest.mark.parametrize("command", ["build", "verify"])
def test_prepare_group_exposes_build_and_verify(command, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["prepare", command, "--help"])
    assert exc.value.code == 0
    assert "--config" in capsys.readouterr().out
