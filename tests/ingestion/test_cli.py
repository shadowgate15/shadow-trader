from pathlib import Path

import pytest

from shadow_trader import main

SUBCOMMANDS = ("fetch", "build", "live", "build-live", "diff")

VALID = """
[paths]
data_root = "/data/shadow"

[build]
instruments = ["EURUSD"]
"""


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "ingest.toml"
    path.write_text(VALID)
    return path


def run(argv: list[str]) -> int:
    try:
        return main(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1


def test_ingest_help_lists_the_five_subcommands(capsys):
    assert run(["ingest", "--help"]) == 0

    out = capsys.readouterr().out
    for name in SUBCOMMANDS:
        assert name in out


@pytest.mark.parametrize("command", SUBCOMMANDS)
def test_every_subcommand_takes_config(command, config_path, capsys):
    extra = ["--from", "2024-01", "--to", "2024-02"] if command == "build-live" else []

    code = run(["ingest", command, "--config", str(config_path), *extra])

    # The skeleton commands are stubs: they load the config, then report
    # that they are not implemented yet (build failure, exit 1).
    assert code == 1
    assert "not implemented" in capsys.readouterr().err


@pytest.mark.parametrize("command", SUBCOMMANDS)
def test_config_is_required(command, capsys):
    assert run(["ingest", command]) == 2


def test_invalid_config_exits_2_with_a_clear_message(tmp_path, capsys):
    path = tmp_path / "bad.toml"
    path.write_text(VALID + "bogus = 1\n")

    assert run(["ingest", "build", "--config", str(path)]) == 2

    assert "bogus" in capsys.readouterr().err


def test_missing_data_root_exits_2(tmp_path, capsys):
    path = tmp_path / "bad.toml"
    path.write_text('[build]\ninstruments = ["EURUSD"]\n')

    assert run(["ingest", "build", "--config", str(path)]) == 2

    assert "data_root" in capsys.readouterr().err


def test_unsupported_bar_size_exits_2(tmp_path, capsys):
    path = tmp_path / "bad.toml"
    path.write_text(VALID + "bar_sizes = [7]\n")

    assert run(["ingest", "build", "--config", str(path)]) == 2

    assert "bar size" in capsys.readouterr().err


def test_flags_may_override_instruments_time_range_and_bar_sizes(config_path, capsys):
    code = run(
        [
            "ingest",
            "build",
            "--config",
            str(config_path),
            "--instruments",
            "USDJPY,GBPUSD",
            "--start",
            "2020-01",
            "--end",
            "2020-06",
            "--bar-sizes",
            "1,15",
        ]
    )

    assert code == 1  # got past parsing and validation to the stub


def test_invalid_override_value_exits_2(config_path, capsys):
    code = run(["ingest", "build", "--config", str(config_path), "--bar-sizes", "7"])

    assert code == 2
    assert "bar size" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flag",
    [
        "--outage-threshold-minutes",
        "--grace-seconds",
        "--repair-window-minutes",
        "--alert-fraction",
        "--price-scale",
    ],
)
def test_overriding_a_rule_value_is_rejected(flag, config_path, capsys):
    code = run(["ingest", "build", "--config", str(config_path), flag, "5"])

    assert code == 2
    assert flag in capsys.readouterr().err
