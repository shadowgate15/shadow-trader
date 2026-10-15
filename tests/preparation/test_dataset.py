import json
from datetime import UTC, datetime

import polars as pl
import pytest

from shadow_trader.ingestion.dataset import open_dataset
from shadow_trader.preparation import SPECS, build_prepared, open_prepared
from shadow_trader.preparation.dataset import PreparedDatasetError

from .synthetic import prepared_config, write_ingestion


def build(tmp_path, **kwargs):
    parent = write_ingestion(tmp_path)
    config = prepared_config(tmp_path, parent.dataset_id)
    return parent, config, build_prepared(config, **kwargs)


def test_layout_is_partitioned_by_instrument_and_year_in_a_separate_root(tmp_path):
    parent, config, ref = build(tmp_path)

    root = config.paths.prepared_root / "datasets" / ref.dataset_id
    assert (root / "manifest.json").is_file()
    assert (root / "bars/instrument=EURUSD/year=2024/part-0.parquet").is_file()
    # Ingestion scans never see it, and ingestion's own Dataset is untouched.
    ingestion_datasets = config.paths.data_root / "datasets"
    assert [p.name for p in ingestion_datasets.iterdir()] == [parent.dataset_id]
    assert (
        open_dataset(config.paths.data_root, parent.dataset_id).manifest["row_count"]
        == 32
    )


def test_manifest_records_lineage_spec_and_unhashed_provenance(tmp_path):
    parent, _, ref = build(tmp_path)

    manifest = json.loads((ref.path / "manifest.json").read_text())

    spec = SPECS[1]
    assert manifest["kind"] == "prepared"
    assert manifest["parent_ingestion"] == parent.dataset_id
    assert manifest["bar_size"] == 5
    assert manifest["instruments"] == ["EURUSD"]
    assert manifest["feature_spec"] == {
        "version": 1,
        "hash": spec.hash,
        "columns": list(spec.names),
    }
    assert manifest["row_count"] == 32
    assert manifest["dataset_id"] == manifest["content_hash"] == ref.dataset_id
    for provenance in ("polars_version", "created_at", "builder_git_revision"):
        assert provenance in manifest


def test_rebuilding_gives_the_same_id_whatever_the_provenance(tmp_path):
    from shadow_trader.preparation.dataset import write_prepared

    _, config, first = build(tmp_path)
    second = build_prepared(config)
    assert first.dataset_id == second.dataset_id

    frame = open_prepared(config.paths.prepared_root, first.dataset_id).read("EURUSD")
    elsewhere = tmp_path / "elsewhere"
    again = write_prepared(
        elsewhere,
        {"EURUSD": frame},
        parent_ingestion=first.manifest["parent_ingestion"],
        spec=SPECS[1],
        created_at=datetime(2030, 1, 1, tzinfo=UTC),
    )
    assert again.dataset_id == first.dataset_id
    assert again.manifest["created_at"] != first.manifest["created_at"]


def test_changed_rows_or_parent_give_a_new_id(tmp_path):
    from shadow_trader.preparation.dataset import write_prepared

    _, config, first = build(tmp_path)
    frame = open_prepared(config.paths.prepared_root, first.dataset_id).read("EURUSD")
    parent = first.manifest["parent_ingestion"]
    root = tmp_path / "elsewhere"

    edited = frame.with_columns(pl.col("feat_ret_1") + 1e-9)
    other_rows = write_prepared(
        root, {"EURUSD": edited}, parent_ingestion=parent, spec=SPECS[1]
    )
    other_parent = write_prepared(
        root, {"EURUSD": frame}, parent_ingestion="x", spec=SPECS[1]
    )

    assert len({first.dataset_id, other_rows.dataset_id, other_parent.dataset_id}) == 3


def test_open_refuses_an_incomplete_dataset(tmp_path):
    _, config, ref = build(tmp_path)
    (ref.path / "manifest.json").unlink()

    with pytest.raises(PreparedDatasetError, match="no manifest"):
        open_prepared(config.paths.prepared_root, ref.dataset_id)


def test_features_are_true_parquet_nulls_never_nan(tmp_path):
    _, config, ref = build(tmp_path)
    frame = open_prepared(config.paths.prepared_root, ref.dataset_id).read("EURUSD")

    for column in (c for c in frame.columns if c.startswith("feat_")):
        assert frame[column].is_nan().sum() == 0
        assert frame[column].is_infinite().sum() == 0


def test_progress_reports_phases_through_the_sink(tmp_path):
    class Recorder:
        def __init__(self):
            self.events = []

        def start(self, phase, total=None):
            self.events.append(("start", phase))

        def advance(self, detail="", outcome=None):
            self.events.append(("advance", detail))

        def warn(self, message):
            pass

        def status(self, text):
            pass

        def done(self):
            self.events.append(("done",))

    sink = Recorder()
    build(tmp_path, progress=sink)

    assert [e[1] for e in sink.events if e[0] == "start"] == ["features", "write"]
