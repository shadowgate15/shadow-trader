"""Opt-in smoke: prepare real EUR/USD ingestion data. Opt in with ``-m slow``.

Set ``SHADOW_TRADER_RAW_ROOT`` (the same ``data_root`` the ingestion slow tests
use) to a root holding a built 5-minute EURUSD ingestion Dataset under
``datasets/``. The newest such Dataset is split 60/20/20 by time and prepared
into a temporary root; nothing real is committed or written under ``data_root``.
Skipped when the variable is unset or no such Dataset exists.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from shadow_trader import main
from shadow_trader.preparation import (
    build_prepared,
    load_prepared_config,
    open_prepared,
)

pytestmark = pytest.mark.slow


def _latest_eurusd_manifest() -> tuple[Path, dict[str, Any]]:
    root = os.environ.get("SHADOW_TRADER_RAW_ROOT")
    if not root:
        pytest.skip("set SHADOW_TRADER_RAW_ROOT to a data_root with an EURUSD Dataset")
    found: list[dict[str, Any]] = []
    for path in (Path(root) / "datasets").glob("*/manifest.json"):
        manifest = json.loads(path.read_text())
        if manifest["bar_size"] == 5 and manifest["instruments"] == ["EURUSD"]:
            found.append(manifest)
    if not found:
        pytest.skip("data_root holds no 5-minute EURUSD ingestion Dataset")
    return Path(root).resolve(), max(found, key=lambda m: m["created_at"])


def test_real_eurusd_dataset_prepares_and_verifies(tmp_path, capsys):
    data_root, manifest = _latest_eurusd_manifest()
    start = datetime.fromisoformat(manifest["time_range"]["start"])
    end = datetime.fromisoformat(manifest["time_range"]["end"])
    span = end - start
    if span.days < 365:
        pytest.skip("ingestion Dataset spans under a year; too short to split")
    validation_start = start + span * 0.6
    test_start = start + span * 0.8
    prepared_root = tmp_path / "prepared"
    config_path = tmp_path / "prepare.toml"
    config_path.write_text(
        f'[paths]\ndata_root = "{data_root}"\nprepared_root = "{prepared_root}"\n'
        f'[input]\ndataset_id = "{manifest["dataset_id"]}"\n'
        "[spec]\nversion = 3\n"
        '[split]\nscheme = "chronological"\n'
        f"train_start = {start:%Y-%m-%dT%H:%M:%SZ}\n"
        f"validation_start = {validation_start:%Y-%m-%dT%H:%M:%SZ}\n"
        f"test_start = {test_start:%Y-%m-%dT%H:%M:%SZ}\n"
        "[embargo]\nepisode_bars = 288\n"
    )
    config = load_prepared_config(config_path)

    ref = build_prepared(config)

    assert set(ref.datasets) == {"train", "validation", "test"}
    for role, dataset_ref in ref.datasets.items():
        dataset = open_prepared(config.paths.prepared_root, dataset_ref.dataset_id)
        assert dataset.manifest["role"] == role
        assert dataset.read("EURUSD").height > 0
    assert main(["prepare", "verify", ref.build_id, "--config", str(config_path)]) == 0
    assert "FAIL" not in capsys.readouterr().out
