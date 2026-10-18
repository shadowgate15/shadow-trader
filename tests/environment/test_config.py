"""EngineConfig / InstrumentSpec: validation, TOML loading and hashing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shadow_trader.environment import EngineConfig, InstrumentSpec
from shadow_trader.ingestion.config import ConfigError


def test_defaults_are_valid_and_single_pair() -> None:
    config = EngineConfig()
    assert [spec.symbol for spec in config.instruments] == ["EURUSD"]
    assert config.starting_equity == 10_000.0
    assert config.closeout_fraction == 0.5
    assert config.levels == (-1.0, 0.0, 1.0)
    assert config.effective_leverage_cap <= config.instruments[0].regulatory_cap


def test_effective_cap_above_regulatory_cap_is_rejected() -> None:
    with pytest.raises(ConfigError, match="leverage"):
        EngineConfig(leverage_cap=51.0)


def test_effective_cap_equal_to_regulatory_cap_is_accepted() -> None:
    assert EngineConfig(leverage_cap=50.0).effective_leverage_cap == 50.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"starting_equity": 0.0},
        {"closeout_fraction": 0.0},
        {"closeout_fraction": 1.5},
        {"leverage_cap": 0.0},
        {"levels": ()},
        {"levels": (-2.0, 0.0, 1.0)},
        {"levels": (0.0, 0.0, 1.0)},
        {"instruments": ()},
    ],
)
def test_invalid_account_settings_raise(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        EngineConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pip_size": 0.0},
        {"margin_rate": 0.0},
        {"slippage_pips": -0.1},
        {"regulatory_cap": 0.0},
        {"margin_rate": 0.01, "regulatory_cap": 50.0},  # 1/50 = 2% minimum
        {"symbol": "USDJPY"},  # USD-base pairs are not supported yet
        {"symbol": ""},
    ],
)
def test_invalid_instrument_spec_raises(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        InstrumentSpec(**kwargs)


def test_duplicate_instruments_are_rejected() -> None:
    with pytest.raises(ConfigError, match="duplicate"):
        EngineConfig(instruments=(InstrumentSpec(), InstrumentSpec()))


def test_config_hash_is_stable_and_content_sensitive() -> None:
    first, second = EngineConfig(), EngineConfig()
    assert first.config_hash() == second.config_hash()
    assert len(first.config_hash()) == 64
    assert EngineConfig(starting_equity=5_000.0).config_hash() != first.config_hash()
    assert InstrumentSpec().config_hash() == InstrumentSpec().config_hash()


def test_canonical_json_is_key_sorted_and_compact() -> None:
    text = EngineConfig().canonical_json().decode()
    assert text.startswith('{"closeout_fraction":0.5,')
    assert " " not in text


def test_from_toml_overrides_code_defaults(tmp_path: Path) -> None:
    path = tmp_path / "engine.toml"
    path.write_text(
        """
starting_equity = 25000.0
leverage_cap = 20.0
levels = [-1.0, -0.5, 0.0, 0.5, 1.0]

[instruments.EURUSD]
slippage_pips = 0.2

[instruments.GBPUSD]
margin_rate = 0.02
"""
    )
    config = EngineConfig.from_toml(path)
    assert config.starting_equity == 25_000.0
    assert config.levels == (-1.0, -0.5, 0.0, 0.5, 1.0)
    assert [spec.symbol for spec in config.instruments] == ["EURUSD", "GBPUSD"]
    assert config.instruments[0].slippage_pips == 0.2
    assert config.instruments[1].slippage_pips == InstrumentSpec().slippage_pips
    assert config.closeout_fraction == 0.5  # default kept


def test_from_toml_empty_file_gives_defaults(tmp_path: Path) -> None:
    path = tmp_path / "engine.toml"
    path.write_text("")
    assert EngineConfig.from_toml(path) == EngineConfig()


def test_from_toml_rejects_unknown_keys_and_bad_types(tmp_path: Path) -> None:
    path = tmp_path / "engine.toml"
    path.write_text("bogus = 1\n")
    with pytest.raises(ConfigError, match="bogus"):
        EngineConfig.from_toml(path)
    path.write_text('starting_equity = "lots"\n')
    with pytest.raises(ConfigError):
        EngineConfig.from_toml(path)
    path.write_text("[instruments.EURUSD]\nbogus = 1\n")
    with pytest.raises(ConfigError, match="bogus"):
        EngineConfig.from_toml(path)
    with pytest.raises(ConfigError):
        EngineConfig.from_toml(tmp_path / "missing.toml")
