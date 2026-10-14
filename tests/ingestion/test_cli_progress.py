"""Progress UX of ``shadow-trader ingest``: stderr only, quiet, plain off a TTY."""

from __future__ import annotations

import io
import zipfile

import pytest
from rich.console import Console

from shadow_trader import main
from shadow_trader.ingestion import cli
from shadow_trader.ingestion.config import load_config
from shadow_trader.ingestion.sources import Response

from .test_backfill import cli_config, fixture_csv, install_raw
from .test_cli import FakeTransport, fetch_config, live_config


def run(argv: list[str]) -> int:
    try:
        return main(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1


@pytest.fixture(autouse=True)
def plain_env(monkeypatch):
    for name in ("NO_COLOR", "FORCE_COLOR", "TTY_COMPATIBLE", "TTY_INTERACTIVE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "_sleep", lambda s: None)


def built(tmp_path, *flags, month="2024-01", sizes="[5]"):
    path, config = cli_config(tmp_path, month, sizes)
    install_raw(config, "EURUSD", month.replace("-", ""), fixture_csv("eurusd_winter"))
    return run(["ingest", "build", "--config", str(path), *flags])


def test_build_progress_is_on_stderr_and_stdout_keeps_only_the_result(tmp_path, capsys):
    assert built(tmp_path) == 0

    captured = capsys.readouterr()
    for phase in ("scanning clocks", "building months", "writing dataset"):
        assert phase in captured.err
        assert phase not in captured.out
    assert "EURUSD 202401" in captured.err
    assert captured.out.startswith("built 5 min dataset ")


def test_quiet_silences_progress_and_leaves_stdout_unchanged(tmp_path, capsys):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    assert built(tmp_path / "a") == 0
    loud = capsys.readouterr()
    assert built(tmp_path / "b", "--quiet") == 0
    quiet = capsys.readouterr()

    assert quiet.err == "" and loud.err != ""
    assert quiet.out.startswith("built 5 min dataset ")
    assert len(quiet.out.splitlines()) == len(loud.out.splitlines())


def test_quiet_is_accepted_before_or_after_the_subcommand(tmp_path, capsys):
    path, config = cli_config(tmp_path, sizes="[5]")
    install_raw(config, "EURUSD", "202401", fixture_csv("eurusd_winter"))

    assert run(["ingest", "-q", "build", "--config", str(path)]) == 0
    assert capsys.readouterr().err == ""
    assert run(["ingest", "build", "--config", str(path), "-q"]) == 0
    assert capsys.readouterr().err == ""


def test_quiet_and_verbose_together_are_rejected(tmp_path, capsys):
    path, _ = cli_config(tmp_path)

    assert run(["ingest", "-q", "-v", "build", "--config", str(path)]) == 2


def test_off_a_tty_the_output_is_plain_lines(tmp_path, capsys):
    assert built(tmp_path) == 0

    err = capsys.readouterr().err
    assert "\x1b" not in err and "\r" not in err
    assert all(line.strip() for line in err.splitlines())


def test_a_tolerated_warning_is_shown_as_it_happens_and_not_when_quiet(
    tmp_path, capsys
):
    path, config = cli_config(tmp_path, "2008-12", "[5]")
    install_raw(config, "EURUSD", "200812", b"20081207 170000000,1.30000,1.30010,0\n")

    assert run(["ingest", "build", "--config", str(path)]) == 0
    captured = capsys.readouterr()
    assert "warning" in captured.err and "thin_data" in captured.err
    assert "thin_data" in captured.out  # the end-of-run summary stays

    assert run(["ingest", "build", "--config", str(path), "-q"]) == 0
    assert "thin_data" not in capsys.readouterr().err


def test_default_caps_repeated_warnings_and_verbose_shows_them_all(
    tmp_path, capsys, monkeypatch
):
    def noisy(config, bar_size, progress):
        for n in range(30):
            progress.warn(f"w{n}")
        raise cli.BuildFailure("stop")

    monkeypatch.setattr(
        cli, "build_backfill", lambda c, b, progress: noisy(c, b, progress)
    )
    path, _ = cli_config(tmp_path)

    run(["ingest", "build", "--config", str(path)])
    err = capsys.readouterr().err
    assert "w0" in err and "w29" not in err and "-v" in err

    run(["ingest", "-v", "build", "--config", str(path)])
    assert "w29" in capsys.readouterr().err


def test_a_failure_names_the_phase_and_the_month(tmp_path, capsys):
    path, _ = cli_config(tmp_path)

    assert run(["ingest", "build", "--config", str(path)]) == 1

    err = capsys.readouterr().err
    assert "scanning clocks" in err and "202401" in err


def test_fetch_shows_each_months_outcome_and_a_tally_on_stderr(
    tmp_path, monkeypatch, capsys
):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("DAT_ASCII_EURUSD_T_202403.csv", b"x")
    monkeypatch.setattr(
        cli, "make_transport", lambda: FakeTransport(Response(200, buf.getvalue(), {}))
    )

    assert run(["ingest", "fetch", "--config", str(fetch_config(tmp_path))]) == 0

    captured = capsys.readouterr()
    assert "fetched EURUSD 202403" in captured.err
    assert "1 fetched" in captured.err  # the tally
    assert captured.out.splitlines() == [
        "fetched 1, skipped 0 cached, absent 0, unsettled 0, rejected 0"
    ]


def test_fetch_lists_cached_months_only_when_verbose(tmp_path, monkeypatch, capsys):
    config_file = fetch_config(tmp_path)
    install_raw(load_config(config_file), "EURUSD", "202403", b"x")
    monkeypatch.setattr(
        cli, "make_transport", lambda: FakeTransport(Response(200, b"", {}))
    )

    run(["ingest", "fetch", "--config", str(config_file)])
    assert "cached EURUSD 202403" not in capsys.readouterr().err

    run(["ingest", "-v", "fetch", "--config", str(config_file)])
    assert "cached EURUSD 202403" in capsys.readouterr().err


def test_fetch_rejection_is_reported_with_its_reason_and_exit_1(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        cli, "make_transport", lambda: FakeTransport(Response(200, b"nope", {}))
    )

    assert run(["ingest", "fetch", "--config", str(fetch_config(tmp_path))]) == 1

    assert "rejected EURUSD 202403" in capsys.readouterr().err


def test_ctrl_c_exits_non_zero_with_one_line_and_no_traceback(
    tmp_path, monkeypatch, capsys
):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "build_backfill", interrupted)
    path, _ = cli_config(tmp_path)

    assert run(["ingest", "build", "--config", str(path)]) == 130

    err = capsys.readouterr().err
    assert err.strip().splitlines()[-1].startswith("interrupted")
    assert "Traceback" not in err


def test_live_prints_a_status_line_and_the_final_result_only_on_stdout(
    tmp_path, capsys
):
    path = live_config(tmp_path)
    install_raw(load_config(path), "EURUSD", "202401", fixture_csv("eurusd_winter"))

    assert run(["ingest", "live", "--config", str(path)]) == 0

    captured = capsys.readouterr()
    assert "bar emissions" in captured.err and "last tick" in captured.err
    assert captured.out.startswith("live EURUSD: ")
    assert len(captured.out.splitlines()) == 1


def tty(monkeypatch, term: str | None = "xterm-256color") -> io.StringIO:
    sink = io.StringIO()
    if term is None:
        monkeypatch.delenv("TERM", raising=False)
    else:
        monkeypatch.setenv("TERM", term)
    monkeypatch.setattr(
        cli,
        "make_console",
        lambda: Console(file=sink, force_terminal=True, width=100),
    )
    return sink


def test_on_a_terminal_a_live_bar_with_counts_and_timing_is_drawn(
    tmp_path, monkeypatch, capsys
):
    sink = tty(monkeypatch)

    assert built(tmp_path) == 0

    drawn = sink.getvalue()
    assert "\x1b" in drawn
    assert "building months" in drawn and "1/1" in drawn
    assert "0:00:0" in drawn  # elapsed
    assert capsys.readouterr().err == ""  # drawn on the console, not stdout


def test_a_dumb_terminal_gets_plain_lines(tmp_path, monkeypatch):
    sink = tty(monkeypatch, "dumb")

    assert built(tmp_path) == 0

    drawn = sink.getvalue()
    assert "\x1b" not in drawn and "building months" in drawn


def test_no_color_drops_colour_codes(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    sink = io.StringIO()
    monkeypatch.setattr(
        cli, "make_console", lambda: Console(file=sink, force_terminal=False)
    )

    assert built(tmp_path) == 0

    assert "\x1b[3" not in sink.getvalue()
