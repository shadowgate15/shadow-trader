import io
import zipfile
from pathlib import Path

import pytest

from shadow_trader import main
from shadow_trader.ingestion import cli
from shadow_trader.ingestion.config import load_config
from shadow_trader.ingestion.sources import FetchReport, Response

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


def live_config(tmp_path: Path, bar_sizes: str = "[5]") -> Path:
    path = tmp_path / "live.toml"
    path.write_text(
        f'[paths]\ndata_root = "{tmp_path / "data"}"\n'
        f'[build]\ninstruments = ["EURUSD"]\nbar_sizes = {bar_sizes}\n'
        'start = "2024-01"\nend = "2024-01"\n'
        '[live]\nadapter = "replay"\n'
    )
    return path


def test_live_runs_the_configured_replay_adapter_through_the_live_ingester(
    tmp_path, capsys
):
    from .test_backfill import fixture_csv, install_raw

    path = live_config(tmp_path)
    install_raw(load_config(path), "EURUSD", "202401", fixture_csv("eurusd_winter"))

    assert run(["ingest", "live", "--config", str(path)]) == 0

    assert "EURUSD" in capsys.readouterr().out
    data = tmp_path / "data"
    assert list((data / "reports/live/replay/EURUSD").glob("*.json"))
    assert list((data / "raw/live/replay/EURUSD").glob("*.jsonl*"))


def test_live_exits_1_when_the_replayed_raw_month_is_missing(tmp_path, capsys):
    assert run(["ingest", "live", "--config", str(live_config(tmp_path))]) == 1

    assert "202401" in capsys.readouterr().err


def test_live_runs_one_bar_size_so_several_is_a_config_error(tmp_path, capsys):
    path = live_config(tmp_path, "[5, 15]")

    assert run(["ingest", "live", "--config", str(path)]) == 2

    assert "one bar size" in capsys.readouterr().err


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


class FakeTransport:
    def __init__(self, reply):
        self.reply = reply
        self.posts = 0

    def get(self, url, headers):
        return Response(200, b'<input type="hidden" name="tk" value="T">', {})

    def post(self, url, form, headers):
        self.posts += 1
        return self.reply


def fetch_config(tmp_path: Path) -> Path:
    path = tmp_path / "fetch.toml"
    path.write_text(
        f'[paths]\ndata_root = "{tmp_path / "data"}"\n'
        '[build]\ninstruments = ["EURUSD"]\nstart = "2024-03"\nend = "2024-03"\n'
    )
    return path


def test_fetch_downloads_into_the_cache_and_exits_0(tmp_path, monkeypatch, capsys):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("DAT_ASCII_EURUSD_T_202403.csv", b"x")
    transport = FakeTransport(Response(200, buf.getvalue(), {}))
    monkeypatch.setattr(cli, "make_transport", lambda: transport)
    monkeypatch.setattr(cli, "_sleep", lambda s: None)

    code = run(["ingest", "fetch", "--config", str(fetch_config(tmp_path))])

    assert code == 0
    assert (tmp_path / "data/raw/histdata/EURUSD/202403.zip").read_bytes() == (
        buf.getvalue()
    )
    assert "fetched 1" in capsys.readouterr().out


def test_fetch_exits_1_when_a_download_is_rejected(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "make_transport", lambda: FakeTransport(Response(200, b"not a zip", {}))
    )
    monkeypatch.setattr(cli, "_sleep", lambda s: None)

    code = run(["ingest", "fetch", "--config", str(fetch_config(tmp_path))])

    assert code == 1
    assert "202403" in capsys.readouterr().err


def test_fetch_reports_absent_months_as_warnings_with_exit_0(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        cli, "make_transport", lambda: FakeTransport(Response(200, b"", {}))
    )
    monkeypatch.setattr(cli, "_sleep", lambda s: None)

    code = run(["ingest", "fetch", "--config", str(fetch_config(tmp_path))])

    assert code == 0
    assert "absent" in capsys.readouterr().out


def test_fetch_reports_unsettled_months_as_warnings_without_downloading(
    tmp_path, monkeypatch, capsys
):
    transport = FakeTransport(Response(200, b"", {}))
    monkeypatch.setattr(cli, "make_transport", lambda: transport)
    monkeypatch.setattr(cli, "_sleep", lambda s: None)
    path = tmp_path / "future.toml"
    path.write_text(
        f'[paths]\ndata_root = "{tmp_path / "data"}"\n'
        '[build]\ninstruments = ["EURUSD"]\nstart = "2099-01"\nend = "2099-01"\n'
    )

    code = run(["ingest", "fetch", "--config", str(path)])

    assert code == 0
    assert "not settled" in capsys.readouterr().out
    assert transport.posts == 0


def test_fetch_passes_workers_from_the_option_or_the_config(
    tmp_path, monkeypatch, capsys
):
    seen = []

    def spy(*args, **kwargs):
        seen.append(kwargs["workers"])
        return FetchReport()

    monkeypatch.setattr(cli, "fetch_months", spy)
    path = fetch_config(tmp_path)

    assert run(["ingest", "fetch", "--config", str(path)]) == 0
    assert run(["ingest", "fetch", "--config", str(path), "--workers", "3"]) == 0
    path.write_text(
        path.read_text().replace('end = "2024-03"', 'end = "2024-03"\nworkers = 5')
    )
    assert run(["ingest", "fetch", "--config", str(path)]) == 0

    assert seen == [8, 3, 5]


@pytest.mark.parametrize("bad", ["0", "-1"])
def test_fetch_rejects_a_non_positive_workers_option(
    tmp_path, monkeypatch, capsys, bad
):
    transport = FakeTransport(Response(200, b"", {}))
    monkeypatch.setattr(cli, "make_transport", lambda: transport)

    code = run(
        ["ingest", "fetch", "--config", str(fetch_config(tmp_path)), "--workers", bad]
    )

    assert code == 2
    assert "workers" in capsys.readouterr().err
    assert transport.posts == 0


def test_fetch_exits_non_zero_after_an_interrupt(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "fetch_months", lambda *a, **k: FetchReport(interrupted=True)
    )
    path = fetch_config(tmp_path)

    code = run(["ingest", "fetch", "--config", str(path)])

    assert code == cli.EXIT_INTERRUPTED
    assert "interrupted" in capsys.readouterr().out
