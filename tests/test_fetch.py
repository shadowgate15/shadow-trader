import gzip
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from shadow_trader import cli, config, fetch

HEADER = "DateTime,BidOpen,BidHigh,BidLow,BidClose,AskOpen,AskHigh,AskLow,AskClose"
START = datetime(2016, 1, 4, tzinfo=UTC)
END = datetime(2016, 1, 22, tzinfo=UTC)


def week_bytes(opens: datetime) -> bytes:
    """A gzipped vendor file from Sunday's open to Friday's last minute."""
    rows = [HEADER]
    close = opens + timedelta(days=4, hours=23, minutes=59)
    minutes_open = int((close - opens).total_seconds() // 60) + 1
    for moment in (opens + timedelta(minutes=i) for i in range(minutes_open)):
        ts = moment.strftime("%m/%d/%Y %H:%M:%S.000")
        rows.append(f"{ts},1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1")
    return gzip.compress(("\n".join(rows) + "\n").encode())


def sunday(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 22, tzinfo=UTC)


class FakeSource:
    """Serves {(year, index): bytes}; 404 (None) for anything else."""

    def __init__(self, files: dict[tuple[int, int], bytes]):
        self.files = files
        self.requests: list[tuple[int, int]] = []

    def get(self, year: int, index: int) -> bytes | None:
        self.requests.append((year, index))
        return self.files.get((year, index))


def weeks(numbering: dict[int, int]) -> dict[tuple[int, int], bytes]:
    """Weeks opening Sun 2015-12-27, 2016-01-03, ... keyed by chosen vendor index."""
    opens = [
        sunday(2015, 12, 27),
        sunday(2016, 1, 3),
        sunday(2016, 1, 10),
        sunday(2016, 1, 17),
        sunday(2016, 1, 24),
        sunday(2016, 1, 31),
    ]
    return {(2016, idx): week_bytes(opens[pos]) for pos, idx in numbering.items()}


def run_fetch(source, tmp_path: Path, out: list[str] | None = None):
    lines = out if out is not None else []
    return fetch.fetch_history(
        source, tmp_path, start=START, end=END, progress=lines.append
    )


def test_downloads_every_week_in_range_and_stores_raw_bytes_verbatim(tmp_path):
    files = weeks({1: 1, 2: 2, 3: 3})
    run_fetch(FakeSource(files), tmp_path)
    for (year, index), raw in files.items():
        assert (
            tmp_path / "m1" / "EURUSD" / str(year) / f"{index}.csv.gz"
        ).read_bytes() == raw


def test_second_run_downloads_nothing_already_cached(tmp_path):
    files = weeks({1: 1, 2: 2, 3: 3})
    run_fetch(FakeSource(files), tmp_path)
    again = FakeSource(files)
    run_fetch(again, tmp_path)
    cached = {key for key in files}
    assert not cached & set(again.requests)


def test_discovery_uses_coverage_in_the_data_not_the_filename(tmp_path):
    # Vendor numbering is not the ISO week: indices here are arbitrary and one
    # week even lives under the previous year's directory.
    files = {
        (2016, 7): week_bytes(sunday(2016, 1, 3)),
        (2016, 9): week_bytes(sunday(2016, 1, 10)),
        (2015, 53): week_bytes(sunday(2016, 1, 17)),
    }
    result = run_fetch(FakeSource(files), tmp_path)
    assert len(result.weeks) == 3
    assert result.weeks[0].first == sunday(2016, 1, 3)
    assert result.weeks[-1].last >= END


def test_weeks_outside_the_range_are_not_in_the_result(tmp_path):
    result = run_fetch(
        FakeSource(weeks({0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5})), tmp_path
    )
    assert [w.first for w in result.weeks] == [
        sunday(2016, 1, 3),
        sunday(2016, 1, 10),
        sunday(2016, 1, 17),
    ]


def test_missing_week_inside_the_range_aborts(tmp_path):
    files = weeks({1: 1, 3: 3})  # the week of 2016-01-10 is absent
    with pytest.raises(fetch.HistoryError, match="missing"):
        run_fetch(FakeSource(files), tmp_path)


def test_history_not_reaching_the_start_aborts(tmp_path):
    files = weeks({2: 2, 3: 3})
    with pytest.raises(fetch.HistoryError, match="start"):
        run_fetch(FakeSource(files), tmp_path)


def test_history_not_reaching_the_end_aborts(tmp_path):
    files = weeks({1: 1, 2: 2})
    with pytest.raises(fetch.HistoryError, match="end"):
        run_fetch(FakeSource(files), tmp_path)


def test_corrupt_file_aborts(tmp_path):
    files = weeks({1: 1, 2: 2, 3: 3})
    files[(2016, 2)] = b"not gzip"
    with pytest.raises(fetch.HistoryError):
        run_fetch(FakeSource(files), tmp_path)


def test_progress_is_reported_as_files_are_fetched(tmp_path):
    out: list[str] = []
    run_fetch(FakeSource(weeks({1: 1, 2: 2, 3: 3})), tmp_path, out)
    fetched = [line for line in out if "2016/2" in line]
    assert fetched, out


def test_cli_fetch_data_runs_the_fetch_and_prints_progress(
    tmp_path, monkeypatch, capsys
):
    files = weeks({1: 1, 2: 2, 3: 3})
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "HISTORY_START", START)
    monkeypatch.setattr(config, "HISTORY_END", END)
    monkeypatch.setattr(fetch, "default_source", lambda: FakeSource(files))
    cli.main(["fetch-data"])
    assert "2016/1" in capsys.readouterr().out
    assert (tmp_path / "raw" / "m1" / "EURUSD" / "2016" / "1.csv.gz").exists()


def test_cli_fetch_data_then_writes_bars_parquet_and_prints_the_summary(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "HISTORY_START", START)
    monkeypatch.setattr(config, "HISTORY_END", END)
    monkeypatch.setattr(
        fetch, "default_source", lambda: FakeSource(weeks({1: 1, 2: 2, 3: 3}))
    )
    cli.main(["fetch-data"])
    out = capsys.readouterr().out
    assert out.index("2016/1") < out.index("aggregating") < out.index("bars: ")
    table = pq.read_table(tmp_path / "bars" / "EURUSD_M5.parquet")
    times = table["time"].to_pylist()
    assert times and min(times) >= START and max(times) < END  # clipped to range


def test_vendor_placeholder_for_an_unpublished_week_counts_as_missing(tmp_path):
    files = weeks({1: 1, 2: 2, 3: 3})
    files[(2016, 2)] = gzip.compress((HEADER + "\n\n").encode())
    with pytest.raises(fetch.HistoryError, match="missing"):
        run_fetch(FakeSource(files), tmp_path)


def test_known_vendor_unpublished_week_is_a_gap_not_a_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        config,
        "KNOWN_MISSING_WEEKS",
        ((sunday(2016, 1, 10), sunday(2016, 1, 10) + timedelta(days=4, hours=23)),),
    )
    result = run_fetch(FakeSource(weeks({1: 1, 3: 3})), tmp_path)
    assert len(result.weeks) == 2


def test_a_different_missing_week_still_aborts_despite_a_known_exception(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        config,
        "KNOWN_MISSING_WEEKS",
        ((sunday(2016, 1, 3), sunday(2016, 1, 3) + timedelta(days=4, hours=23)),),
    )
    with pytest.raises(fetch.HistoryError, match="missing"):
        run_fetch(FakeSource(weeks({1: 1, 3: 3})), tmp_path)
