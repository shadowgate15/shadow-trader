import dataclasses
import os
from pathlib import Path

import pytest

from shadow_trader.ingestion.config import ConfigError, load_config

MINIMAL = """
[paths]
data_root = "/data/shadow"

[build]
instruments = ["EURUSD"]
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "ingest.toml"
    path.write_text(text)
    return path


def test_minimal_config_gets_spec_defaults(tmp_path):
    config = load_config(write(tmp_path, MINIMAL))

    assert config.paths.data_root == Path("/data/shadow")
    assert config.build.instruments == ("EURUSD",)
    assert config.build.bar_sizes == (5,)
    assert config.build.start == "2009-01"
    assert config.build.end is None
    assert config.build.workers == 8
    assert config.build.jobs == os.process_cpu_count()
    assert config.build.acknowledged_missing == ()
    assert config.gap_rule.outage_threshold_minutes == 30
    assert config.anomaly_rule.alert_fraction == 0.01
    assert config.live.adapter == "replay"
    assert config.live.grace_seconds == 1
    assert config.live.repair_window_minutes == 60


def test_price_scale_defaults_by_instrument_class_and_can_be_set(tmp_path):
    text = MINIMAL.replace('["EURUSD"]', '["EURUSD", "USDJPY", "GBPUSD"]')
    text += "\n[instruments.GBPUSD]\nprice_scale = 4\n"
    config = load_config(write(tmp_path, text))

    assert config.instruments["EURUSD"].price_scale == 5
    assert config.instruments["USDJPY"].price_scale == 3
    assert config.instruments["GBPUSD"].price_scale == 4


def test_config_is_frozen(tmp_path):
    config = load_config(write(tmp_path, MINIMAL))

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.gap_rule.outage_threshold_minutes = 1  # ty: ignore[invalid-assignment]


def test_acknowledged_missing_entries_are_parsed(tmp_path):
    text = MINIMAL + (
        "acknowledged_missing = [\n"
        '  { instrument = "EURUSD", month = "2010-03", reason = "vendor hole" },\n'
        '  { instrument = "EURUSD", month = "2010-04" },\n'
        "]\n"
    )
    config = load_config(write(tmp_path, text))

    first, second = config.build.acknowledged_missing
    assert (first.instrument, first.month, first.reason) == (
        "EURUSD",
        "2010-03",
        "vendor hole",
    )
    assert second.reason is None


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (MINIMAL + "\n[rules.gap]\noutage_treshold_minutes = 5\n", "outage_treshold"),
        (MINIMAL.replace("[paths]", "[pahts]"), "pahts"),
        (MINIMAL.replace('data_root = "/data/shadow"', ""), "data_root"),
        (MINIMAL + "bar_sizes = [7]\n", "bar size"),
        (MINIMAL.replace('["EURUSD"]', '["EURGBP"]'), "EURGBP"),
        (MINIMAL + 'start = "2020-05"\nend = "2020-01"\n', "start"),
        (MINIMAL + "\n[rules.gap]\noutage_threshold_minutes = 0\n", "outage"),
        (MINIMAL + "\n[live]\nadapter = 'oanda'\n", "adapter"),
        (MINIMAL + "jobs = 0\n", "jobs"),
        (MINIMAL + "jobs = -3\n", "jobs"),
        (MINIMAL + "workers = 0\n", "workers"),
        (MINIMAL + "workers = -2\n", "workers"),
        ("this is not toml [", "TOML"),
    ],
)
def test_invalid_config_raises_config_error_naming_the_problem(tmp_path, text, message):
    with pytest.raises(ConfigError, match=message):
        load_config(write(tmp_path, text))


def test_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match=r"nope\.toml"):
        load_config(tmp_path / "nope.toml")


def test_content_affecting_and_reporting_only_values_are_distinguishable(tmp_path):
    config = load_config(write(tmp_path, MINIMAL))

    content = config.content_affecting()
    assert content["rules"]["gap"]["outage_threshold_minutes"] == 30
    assert "alert_fraction" not in content["rules"]["anomaly"]
    assert "reconciliation" not in content
    assert "live" not in content

    other = dataclasses.replace(
        config,
        anomaly_rule=dataclasses.replace(config.anomaly_rule, alert_fraction=0.5),
    )
    assert other.content_affecting() == content
    stricter = dataclasses.replace(
        config,
        gap_rule=dataclasses.replace(config.gap_rule, outage_threshold_minutes=5),
    )
    assert stricter.content_affecting() != content


def test_jobs_is_read_from_the_build_table_independently_of_workers(tmp_path):
    config = load_config(write(tmp_path, MINIMAL + "jobs = 3\nworkers = 2\n"))

    assert (config.build.jobs, config.build.workers) == (3, 2)
