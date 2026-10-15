from dataclasses import FrozenInstanceError

import pytest

from shadow_trader.preparation import SPECS, load_prepared_config
from shadow_trader.preparation.config import ConfigError

from .synthetic import write_prepared_toml


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
