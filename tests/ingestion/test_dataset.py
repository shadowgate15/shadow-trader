"""Dataset writer, hasher, manifest and reader, through build/open_dataset."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pyarrow.parquet as pq
import pytest

from shadow_trader.ingestion.bars import Quote, build_bars
from shadow_trader.ingestion.config import Config, load_config
from shadow_trader.ingestion.dataset import DatasetError, open_dataset, write_dataset

# Winter Session opening Sun 2024-01-07 22:00 UTC.
WINTER = datetime(2024, 1, 7, 22, 0, tzinfo=UTC)

CONFIG_TOML = """
[paths]
data_root = "{root}"

[build]
instruments = ["EURUSD"]

{rules}
"""


def make_config(tmp_path, rules="") -> Config:
    path = tmp_path / "ingest.toml"
    path.write_text(CONFIG_TOML.format(root=tmp_path / "data", rules=rules))
    return load_config(path)


def make_bars(start=WINTER):
    ticks = [
        Quote(
            start + timedelta(minutes=m),
            Decimal("1.10000") + Decimal(m) / 100000,
            Decimal("1.10010") + Decimal(m) / 100000,
        )
        for m in (0, 1, 7, 12)
    ]
    bars, _ = build_bars("EURUSD", ticks, 5, 5)
    return {"EURUSD": bars}


def test_build_twice_gives_same_id_and_dataset_reads_back(tmp_path):
    config = make_config(tmp_path)
    bars = make_bars()

    first = write_dataset(config, 5, bars)
    second = write_dataset(config, 5, bars)

    assert first.dataset_id == second.dataset_id
    root = config.paths.data_root / "datasets" / first.dataset_id
    assert (root / "manifest.json").is_file()
    assert (root / "bars/instrument=EURUSD/year=2024/part-0.parquet").is_file()
    dataset = open_dataset(config.paths.data_root, first.dataset_id)
    assert dataset.read("EURUSD") == bars["EURUSD"]


def test_provenance_never_changes_id_but_content_affecting_rules_do(tmp_path):
    config = make_config(tmp_path)
    bars = make_bars()
    base = write_dataset(config, 5, bars)

    other_provenance = write_dataset(
        config,
        5,
        bars,
        created_at=datetime(2030, 1, 1, tzinfo=UTC),
        builder_revision="deadbeef",
    )
    stricter = write_dataset(
        make_config(tmp_path, "[rules.gap]\noutage_threshold_minutes = 45\n"),
        5,
        bars,
    )
    reporting_only = write_dataset(
        make_config(tmp_path, "[rules.anomaly]\nalert_fraction = 0.5\n"), 5, bars
    )

    assert other_provenance.dataset_id == base.dataset_id
    assert reporting_only.dataset_id == base.dataset_id
    assert stricter.dataset_id != base.dataset_id
    assert write_dataset(config, 10, bars).dataset_id != base.dataset_id


def test_contract_schema_metadata_and_manifest_fields(tmp_path):
    config = make_config(tmp_path)
    ref = write_dataset(config, 5, make_bars())

    table = pq.read_table(ref.path / "bars/instrument=EURUSD/year=2024/part-0.parquet")
    assert {f.name: str(f.type) for f in table.schema} == {
        "instrument": "string",
        "open_time": "timestamp[us, tz=UTC]",
        "open": "double",
        "high": "double",
        "low": "double",
        "close": "double",
        "spread_close": "double",
        "spread_mean": "double",
        "tick_count": "int64",
        "is_gap": "bool",
        "session_id": "timestamp[us, tz=UTC]",
        "is_rollover": "bool",
        "quality_flags": "list<element: string>",
    }
    assert table.schema.metadata[b"schema_version"] == b"1"
    assert table.schema.metadata[b"view"] == b"latest"

    manifest = json.loads((ref.path / "manifest.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["view"] == "latest"
    assert manifest["parent"] is None
    assert manifest["dataset_id"] == manifest["content_hash"] == ref.dataset_id
    assert manifest["bar_size"] == 5
    assert manifest["instruments"] == ["EURUSD"]
    assert manifest["row_count"] == len(make_bars()["EURUSD"])
    assert set(manifest) >= {
        "source",
        "time_range",
        "partitions",
        "raw_inputs",
        "acknowledged_missing",
        "builder_version",
        "builder_git_revision",
        "gap_rule_set_version",
        "anomaly_rule_set_version",
        "created_at",
        "coverage",
    }
    (part,) = manifest["partitions"]
    assert set(part) >= {"path", "rows", "gap_count", "hash"}
    coverage = manifest["coverage"]["EURUSD"]
    assert set(coverage) == {
        "sessions",
        "gaps",
        "flagged_bars",
        "flag_counts",
        "tick_flag_counts",
    }


def test_append_writes_new_dir_hardlinks_closed_years_and_records_parent(tmp_path):
    config = make_config(tmp_path)
    y2023 = make_bars(datetime(2023, 1, 8, 22, 0, tzinfo=UTC))["EURUSD"]
    y2024 = make_bars(WINTER)["EURUSD"]
    later = make_bars(WINTER + timedelta(days=7))["EURUSD"]
    first = write_dataset(config, 5, {"EURUSD": y2023 + y2024})

    second = write_dataset(
        config, 5, {"EURUSD": y2023 + y2024 + later}, parent=first.dataset_id
    )

    assert second.dataset_id != first.dataset_id
    assert second.manifest["parent"] == first.dataset_id
    closed = "bars/instrument=EURUSD/year=2023/part-0.parquet"
    opened = "bars/instrument=EURUSD/year=2024/part-0.parquet"
    assert (second.path / closed).samefile(first.path / closed)
    assert not (second.path / opened).samefile(first.path / opened)
    again = open_dataset(config.paths.data_root, second.dataset_id)
    assert again.read("EURUSD") == y2023 + y2024 + later


def test_unknown_schema_version_and_missing_manifest_are_refused(tmp_path):
    config = make_config(tmp_path)
    ref = write_dataset(config, 5, make_bars())
    manifest_path = ref.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    manifest["schema_version"] = 2
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DatasetError, match="schema_version"):
        open_dataset(config.paths.data_root, ref.dataset_id)

    manifest_path.unlink()
    with pytest.raises(DatasetError, match="no manifest"):
        open_dataset(config.paths.data_root, ref.dataset_id)
