import math

from shadow_trader.preparation import build_prepared, open_prepared

from .synthetic import (
    BARS_PER_SESSION as N,  # ret_288 needs 288 Bars of in-Session history
)
from .synthetic import EMBARGO, prepared_config, write_ingestion

KEPT = N - EMBARGO  # train Bars left after the Embargo trim
GAP = 10
FLAGGED = 20
INFORMATIONAL = 400
FLAGS: dict[int, tuple[str, ...]] = {
    FLAGGED: ("price_spike",),
    INFORMATIONAL: ("zero_spread",),
}
LN2 = math.log(2.0)


def build(tmp_path, extra=""):
    parent = write_ingestion(tmp_path, gaps=(GAP,), flags=FLAGS)
    config = prepared_config(tmp_path, parent.dataset_id, extra)
    ref = build_prepared(config).datasets["train"]
    return ref, open_prepared(config.paths.prepared_root, ref.dataset_id)


def reasons(frame, start, stop):
    return set(frame["unusable_reason"].slice(start, stop - start).to_list())


def test_unusable_reason_follows_the_fixed_priority(tmp_path):
    _, dataset = build(tmp_path)
    frame = dataset.read("EURUSD")

    assert reasons(frame, 0, GAP) == {"dataset_start"}
    assert reasons(frame, GAP, GAP + 1) == {"gap"}
    assert reasons(frame, GAP + 1, FLAGGED) == {"window_contains_gap"}
    assert reasons(frame, FLAGGED, FLAGGED + 1) == {"distrusted_flag"}
    assert reasons(frame, FLAGGED + 1, GAP + 289) == {"window_contains_gap"}
    assert reasons(frame, GAP + 289, FLAGGED + 289) == {
        "window_contains_distrusted_flag"
    }
    assert reasons(frame, FLAGGED + 289, KEPT) == {None}  # incl. informational flag


def test_a_later_split_starts_with_dataset_start_not_session_warm_up(tmp_path):
    parent = write_ingestion(tmp_path, gaps=(GAP,), flags=FLAGS)
    config = prepared_config(tmp_path, parent.dataset_id)
    ref = build_prepared(config).datasets["validation"]
    frame = open_prepared(config.paths.prepared_root, ref.dataset_id).read("EURUSD")

    assert reasons(frame, 0, 288) == {"dataset_start"}
    assert reasons(frame, 288, KEPT) == {None}


def test_usable_iff_reason_is_null_and_rows_are_unaltered(tmp_path):
    _, dataset = build(tmp_path)
    frame = dataset.read("EURUSD")

    assert frame.height == KEPT
    assert frame["usable"].to_list() == [r is None for r in frame["unusable_reason"]]
    assert frame["is_gap"].sum() == 1
    assert frame["close"].null_count() == 1  # only the Gap is unpriced
    assert frame["quality_flags"][FLAGGED].to_list() == ["price_spike"]


def test_features_are_null_only_where_their_own_window_is_bad(tmp_path):
    _, dataset = build(tmp_path)
    frame = dataset.read("EURUSD")

    ret1 = frame["feat_ret_1"]
    assert ret1[GAP + 1] is None  # window holds the Gap
    assert math.isclose(ret1[GAP + 2], LN2, rel_tol=1e-12)
    assert ret1[FLAGGED] is None and ret1[FLAGGED + 1] is None
    assert math.isclose(ret1[FLAGGED + 2], LN2, rel_tol=1e-12)
    assert ret1[INFORMATIONAL] is not None  # informational flag never nulls
    assert frame["feat_ret_288"][FLAGGED + 288] is None
    assert frame["feat_ret_288"][FLAGGED + 289] is not None
    for name in (c for c in frame.columns if c.startswith("feat_")):
        assert frame[name].is_nan().sum() == 0


def test_manifest_records_policy_and_counts(tmp_path):
    ref, _ = build(tmp_path)
    policy = ref.manifest["anomaly_policy"]

    assert policy["distrusted_flags"] == ["crossed_quote", "price_spike", "stale_quote"]
    assert policy["version"] == 1
    assert len(policy["hash"]) == 64
    assert policy["counts_by_reason"] == {
        "gap": 1,
        "distrusted_flag": 1,
        "window_contains_gap": 287,
        "window_contains_distrusted_flag": 10,
        "session_warm_up": 0,
        "dataset_start": GAP,
    }
    assert policy["usable_count"] == KEPT - 309
    assert policy["usable_fraction"] == (KEPT - 309) / KEPT


def test_configured_flag_set_changes_rows_and_dataset_hash(tmp_path):
    default, _ = build(tmp_path)
    ref, dataset = build(tmp_path, '[anomaly]\ndistrusted_flags = ["zero_spread"]\n')
    frame = dataset.read("EURUSD")

    assert ref.manifest["anomaly_policy"]["distrusted_flags"] == ["zero_spread"]
    assert ref.dataset_id != default.dataset_id
    assert frame["unusable_reason"][INFORMATIONAL] == "distrusted_flag"
    assert frame["unusable_reason"][FLAGGED] != "distrusted_flag"
