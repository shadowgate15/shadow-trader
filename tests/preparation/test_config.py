import re
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from shadow_trader.preparation import SPECS, load_prepared_config
from shadow_trader.preparation.config import ConfigError

from .synthetic import (
    EPISODE_BARS,
    SESSION_1,
    SESSION_2,
    SESSION_3,
    split_toml,
    write_prepared_toml,
)


def test_valid_config_is_frozen_and_resolves_the_spec(tmp_path):
    config = load_prepared_config(write_prepared_toml(tmp_path, "abc"))

    assert config.input.dataset_id == "abc"
    assert config.feature_spec is SPECS[1]
    with pytest.raises(FrozenInstanceError):
        setattr(config.input, "dataset_id", "other")  # noqa: B010


@pytest.mark.parametrize(
    "extra, message",
    [
        ("[bogus]\nx = 1\n", "unknown key"),
        ("[spec]\nversion = 1\n", "invalid TOML"),  # duplicate table
    ],
)
def test_unknown_keys_and_bad_toml_are_config_errors(tmp_path, extra, message):
    with pytest.raises(ConfigError, match=message):
        load_prepared_config(write_prepared_toml(tmp_path, "abc", extra))


def test_unknown_key_inside_a_table_is_an_error(tmp_path):
    path = write_prepared_toml(tmp_path, "abc")
    path.write_text(path.read_text().replace("version = 1", "version = 1\nverison = 2"))

    with pytest.raises(ConfigError, match=r"unknown key.*verison"):
        load_prepared_config(path)


def test_input_must_name_a_dataset_explicitly(tmp_path):
    with pytest.raises(ConfigError, match="explicitly"):
        load_prepared_config(write_prepared_toml(tmp_path, "latest"))
    path = write_prepared_toml(tmp_path, "abc")
    path.write_text(path.read_text().replace('dataset_id = "abc"', ""))
    with pytest.raises(ConfigError, match="dataset_id"):
        load_prepared_config(path)


def test_unknown_spec_version_is_a_config_error(tmp_path):
    path = write_prepared_toml(tmp_path, "abc")
    path.write_text(path.read_text().replace("version = 1", "version = 99"))

    with pytest.raises(ConfigError, match=r"spec\.version"):
        load_prepared_config(path)


def test_prepared_root_must_be_separate_from_the_ingestion_root(tmp_path):
    path = write_prepared_toml(tmp_path, "abc")
    text = path.read_text()
    for target in (str(tmp_path / "ingestion"), str(tmp_path / "ingestion/datasets")):
        path.write_text(text.replace(str(tmp_path / "prepared"), target))
        with pytest.raises(ConfigError, match="separate"):
            load_prepared_config(path)


def test_anomaly_flags_default_and_are_validated(tmp_path):
    default = load_prepared_config(write_prepared_toml(tmp_path, "abc"))
    assert default.anomaly.distrusted_flags == (
        "crossed_quote",
        "price_spike",
        "stale_quote",
    )
    with pytest.raises(ConfigError, match="list of strings"):
        load_prepared_config(
            write_prepared_toml(tmp_path, "abc", "[policy]\ndistrusted_flags = 3\n")
        )
    with pytest.raises(ConfigError, match="unknown key"):
        load_prepared_config(
            write_prepared_toml(tmp_path, "abc", "[policy]\nbogus = 1\n")
        )


def test_the_policy_table_is_named_policy_and_anomaly_is_refused(tmp_path):
    config = load_prepared_config(
        write_prepared_toml(
            tmp_path, "abc", '[policy]\ndistrusted_flags = ["zero_spread"]\n'
        )
    )
    assert config.anomaly.distrusted_flags == ("zero_spread",)
    with pytest.raises(ConfigError, match=r"unknown key.*anomaly"):
        load_prepared_config(
            write_prepared_toml(
                tmp_path, "abc", '[anomaly]\ndistrusted_flags = ["zero_spread"]\n'
            )
        )


def test_split_and_embargo_are_parsed_into_utc_bounds(tmp_path):
    config = load_prepared_config(write_prepared_toml(tmp_path, "abc"))

    assert config.split.scheme == "chronological"
    assert config.split.train_start == SESSION_1
    assert config.split.validation_start == SESSION_2
    assert config.split.test_start == SESSION_3
    assert config.split.test_end is None
    assert config.embargo.episode_bars == EPISODE_BARS


def test_embargo_episode_bars_is_required_and_at_least_one(tmp_path):
    with pytest.raises(ConfigError, match=r"embargo"):
        load_prepared_config(
            write_prepared_toml(tmp_path, "abc", split=split_toml(episode_bars=None))
        )
    for bad in ("0", "-3", "2.5", '"2"'):
        text = split_toml().replace("episode_bars = 2", f"episode_bars = {bad}")
        with pytest.raises(ConfigError, match=r"episode_bars"):
            load_prepared_config(write_prepared_toml(tmp_path, "abc", split=text))


@pytest.mark.parametrize(
    "old, new, message",
    [
        ('scheme = "chronological"', 'scheme = "walk_forward"', "scheme"),
        ("train_start = 2024-01-07T22:00:00Z", "train_start = 2024-01-07", "UTC"),
        (
            "train_start = 2024-01-07T22:00:00Z",
            "train_start = 2024-01-07T22:00:00",
            "UTC",
        ),
        ("test_start = 2024-01-21T22:00:00Z\n", "", "test_start"),
        ("test_start", "test_begin", "unknown key|missing"),
    ],
)
def test_bad_split_tables_are_config_errors(tmp_path, old, new, message):
    text = split_toml().replace(old, new)
    with pytest.raises(ConfigError, match=message):
        load_prepared_config(write_prepared_toml(tmp_path, "abc", split=text))


def test_unknown_distrusted_flag_names_are_refused_with_the_known_set(tmp_path):
    with pytest.raises(ConfigError, match=r"crossd_quote.*known:.*crossed_quote"):
        load_prepared_config(
            write_prepared_toml(
                tmp_path, "abc", '[policy]\ndistrusted_flags = ["crossd_quote"]\n'
            )
        )
    config = load_prepared_config(
        write_prepared_toml(
            tmp_path,
            "abc",
            '[policy]\ndistrusted_flags = ["spread_spike", "suspected_outage"]\n',
        )
    )
    assert config.anomaly.distrusted_flags == ("spread_spike", "suspected_outage")


def test_readme_example_config_loads(tmp_path):
    readme = (Path(__file__).parents[2] / "README.md").read_text()
    section = readme.split("\n## Dataset preparation\n", 1)[1].split("\n## ", 1)[0]
    blocks = re.findall(r"^```toml\n(.*?)^```", section, re.DOTALL | re.MULTILINE)
    assert len(blocks) == 1, "expected one toml example under Dataset preparation"
    text = blocks[0]
    text = text.replace("/data/shadow-trader-prepared", str(tmp_path / "prepared"))
    text = text.replace("/data/shadow-trader", str(tmp_path / "ingestion"))
    text = text.replace("<ingestion dataset_id>", "abc")
    path = tmp_path / "prepare.toml"
    path.write_text(text)

    config = load_prepared_config(path)

    assert config.input.dataset_id == "abc"
    assert config.embargo.episode_bars >= 1


def test_an_empty_distrusted_flags_list_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="at least one"):
        load_prepared_config(
            write_prepared_toml(tmp_path, "abc", "[policy]\ndistrusted_flags = []\n")
        )
