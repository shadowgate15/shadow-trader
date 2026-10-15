import os
from datetime import UTC, datetime

import polars as pl
import pyarrow.parquet as pq

from shadow_trader.preparation import build_prepared, open_prepared

from .synthetic import (
    SESSION_1,
    SESSION_2,
    SESSION_3,
    prepared_config,
    session_bars,
    split_toml,
    wiggled,
    write_bars,
)

SESSION_2022 = datetime(2022, 6, 5, 22, 0, tzinfo=UTC)
SESSION_2023 = datetime(2023, 6, 4, 22, 0, tzinfo=UTC)
BASE_SESSIONS = (SESSION_2022, SESSION_2023, SESSION_1, SESSION_2)
SPLIT = split_toml(
    train_start=SESSION_2022, validation_start=SESSION_1, test_start=SESSION_2
)
ROLES = ("train", "validation", "test")
CLOSED_2022 = "bars/instrument=EURUSD/year=2022/part-0.parquet"
OPEN_2023 = "bars/instrument=EURUSD/year=2023/part-0.parquet"


def write_ingestion(root, sessions, first_count=700):
    """Every Session wiggled (so scaler IQRs are non-zero); the first may be shorter."""
    bars = []
    for n, session in enumerate(sessions):
        count = first_count if n == 0 else 700
        bars += [
            wiggled(bar, i) if i >= 100 and not bar.is_gap else bar
            for i, bar in enumerate(
                session_bars("EURUSD", session, count, gaps=(), wiggle=False)
            )
        ]
    return write_bars(root, {"EURUSD": bars})


def build(tmp_path, parent):
    config = prepared_config(tmp_path, parent.dataset_id, split=SPLIT)
    return config, build_prepared(config)


def partition_hash(ref, role, rel):
    entries = ref.datasets[role].manifest["partitions"]
    return next(e["hash"] for e in entries if e["path"] == rel)


def test_first_build_has_no_prepared_parent_and_records_parent_ingestion(tmp_path):
    parent = write_ingestion(tmp_path, BASE_SESSIONS)
    _, ref = build(tmp_path, parent)

    for dataset in ref.datasets.values():
        assert dataset.manifest["parent_ingestion"] == parent.dataset_id
        assert dataset.manifest["prepared_parent"] is None


def test_rebuild_after_growth_records_lineage_and_hardlinks_closed_partitions(
    tmp_path,
):
    first_parent = write_ingestion(tmp_path, BASE_SESSIONS)
    _, first = build(tmp_path, first_parent)
    grown = write_ingestion(tmp_path, (*BASE_SESSIONS, SESSION_3))
    _, second = build(tmp_path, grown)

    for role in ROLES:
        new = second.datasets[role]
        assert new.dataset_id != first.datasets[role].dataset_id
        assert new.manifest["parent_ingestion"] == grown.dataset_id
        assert new.manifest["prepared_parent"] == first.datasets[role].dataset_id
        # nothing is edited in place
        assert (first.datasets[role].path / "manifest.json").is_file()

    old = first.datasets["train"].path / CLOSED_2022
    new = second.datasets["train"].path / CLOSED_2022
    assert os.stat(new).st_ino == os.stat(old).st_ino
    assert os.stat(new).st_nlink == 2

    # the open (latest) year is recomputed, never linked
    old_open = first.datasets["train"].path / OPEN_2023
    new_open = second.datasets["train"].path / OPEN_2023
    assert os.stat(new_open).st_ino != os.stat(old_open).st_ino
    assert os.stat(new_open).st_nlink == 1


def test_changed_closed_partition_is_fresh_never_linked(tmp_path):
    first_parent = write_ingestion(tmp_path, BASE_SESSIONS)
    _, first = build(tmp_path, first_parent)
    changed = write_ingestion(tmp_path, BASE_SESSIONS, first_count=699)
    _, second = build(tmp_path, changed)

    old = first.datasets["train"].path / CLOSED_2022
    new = second.datasets["train"].path / CLOSED_2022
    assert partition_hash(first, "train", CLOSED_2022) != partition_hash(
        second, "train", CLOSED_2022
    )
    assert os.stat(new).st_ino != os.stat(old).st_ino
    assert os.stat(new).st_nlink == 1
    assert second.datasets["train"].manifest["prepared_parent"] == (
        first.datasets["train"].dataset_id
    )


def test_hash_is_the_same_whether_linked_or_recomputed(tmp_path):
    first_parent = write_ingestion(tmp_path, BASE_SESSIONS)
    build(tmp_path, first_parent)
    grown = write_ingestion(tmp_path, (*BASE_SESSIONS, SESSION_3))
    config, linked = build(tmp_path, grown)

    fresh_root = tmp_path / "fresh"
    fresh_root.mkdir()
    fresh_parent = write_ingestion(fresh_root, (*BASE_SESSIONS, SESSION_3))
    fresh_config = prepared_config(fresh_root, fresh_parent.dataset_id, split=SPLIT)
    recomputed = build_prepared(fresh_config)

    assert fresh_parent.dataset_id == grown.dataset_id
    for role in ROLES:
        assert linked.datasets[role].dataset_id == recomputed.datasets[role].dataset_id
        assert recomputed.datasets[role].manifest["prepared_parent"] is None
        left = open_prepared(
            config.paths.prepared_root, linked.datasets[role].dataset_id
        )
        right = open_prepared(
            fresh_config.paths.prepared_root, recomputed.datasets[role].dataset_id
        )
        assert left.read("EURUSD").equals(right.read("EURUSD"))


def test_a_corrupted_earlier_partition_is_never_hardlinked(tmp_path):
    first_parent = write_ingestion(tmp_path, BASE_SESSIONS)
    _, first = build(tmp_path, first_parent)
    old = first.datasets["train"].path / CLOSED_2022
    table = pq.read_table(old)
    damaged = pl.DataFrame(table).with_columns(pl.col("close") + 1.0)
    os.unlink(old)  # the stored file is shared by link; replace, do not edit
    pq.write_table(damaged.to_arrow().cast(table.schema), old)

    grown = write_ingestion(tmp_path, (*BASE_SESSIONS, SESSION_3))
    _, second = build(tmp_path, grown)

    new = second.datasets["train"].path / CLOSED_2022
    assert os.stat(new).st_ino != os.stat(old).st_ino
    assert pl.DataFrame(pq.read_table(new)).equals(pl.DataFrame(table))


def test_an_unrelated_split_is_not_chosen_as_prepared_parent(tmp_path):
    base = write_ingestion(tmp_path, BASE_SESSIONS)
    _, first = build(tmp_path, base)
    # Same Instruments, spec and policy, but a different split series, built later.
    other_split = split_toml(
        train_start=SESSION_2023, validation_start=SESSION_1, test_start=SESSION_2
    )
    other_config = prepared_config(tmp_path, base.dataset_id, split=other_split)
    other = build_prepared(other_config)
    assert other.datasets["train"].dataset_id != first.datasets["train"].dataset_id

    grown = write_ingestion(tmp_path, (*BASE_SESSIONS, SESSION_3))
    _, second = build(tmp_path, grown)

    for role in ROLES:
        assert second.datasets[role].manifest["prepared_parent"] == (
            first.datasets[role].dataset_id
        )
