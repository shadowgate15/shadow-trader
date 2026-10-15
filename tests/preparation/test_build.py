import math

from shadow_trader.preparation import build_prepared, open_prepared

from .synthetic import BARS_PER_SESSION, GAP_INDEX, prepared_config, write_ingestion

LN2 = math.log(2.0)
HORIZONS = (1, 3, 6, 12, 24, 48, 288)


def build(tmp_path):
    parent = write_ingestion(tmp_path)
    config = prepared_config(tmp_path, parent.dataset_id)
    ref = build_prepared(config)
    return (
        parent,
        config,
        ref,
        open_prepared(config.paths.prepared_root, ref.dataset_id),
    )


def test_build_carries_bar_columns_and_returns_features(tmp_path):
    parent, _, _, dataset = build(tmp_path)

    frame = dataset.read("EURUSD")

    assert frame.height == 2 * BARS_PER_SESSION
    assert [c for c in frame.columns if c.startswith("feat_")] == [
        f"feat_ret_{h}" for h in HORIZONS
    ]
    assert {"open_time", "close", "is_gap", "quality_flags"} <= set(frame.columns)
    assert frame["open_time"].is_sorted()
    assert dataset.manifest["parent_ingestion"] == parent.dataset_id
    assert dataset.manifest["kind"] == "prepared"


def test_returns_are_log_returns_that_never_cross_a_session_start(tmp_path):
    *_, dataset = build(tmp_path)
    second = dataset.read("EURUSD").slice(BARS_PER_SESSION)  # no Gap in it

    assert second["feat_ret_1"][0] is None
    assert math.isclose(second["feat_ret_1"][1], LN2, rel_tol=1e-12)
    assert second["feat_ret_3"][2] is None
    assert math.isclose(second["feat_ret_3"][3], 3 * LN2, rel_tol=1e-12)
    assert math.isclose(second["feat_ret_12"][12], 12 * LN2, rel_tol=1e-12)
    assert second["feat_ret_288"].null_count() == second.height


def test_gap_rows_survive_with_null_features(tmp_path):
    *_, dataset = build(tmp_path)
    frame = dataset.read("EURUSD")

    gap = frame.row(GAP_INDEX, named=True)
    after = frame.row(GAP_INDEX + 1, named=True)

    assert gap["is_gap"] and gap["close"] is None
    assert all(gap[c] is None for c in frame.columns if c.startswith("feat_"))
    assert after["feat_ret_1"] is None  # its previous close is the Gap
    assert math.isclose(frame["feat_ret_1"][GAP_INDEX + 2], LN2, rel_tol=1e-12)
    assert frame["feat_ret_1"].is_nan().sum() == 0
