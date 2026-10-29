import itertools
from datetime import UTC, datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from shadow_trader import bars, config, segments

T0 = datetime(2016, 1, 4, tzinfo=UTC)
M5 = timedelta(minutes=5)


def run(start: datetime, count: int) -> list[datetime]:
    return [start + i * M5 for i in range(count)]


def after_gap(times: list[datetime], gap: timedelta) -> datetime:
    """Open time of the next Bar so the Gap (time with no Bar) has this length."""
    return times[-1] + M5 + gap


def test_55_minute_dropout_does_not_split_but_a_weekend_does():
    week1 = run(T0, 100)
    week1 += run(after_gap(week1, timedelta(minutes=55)), 100)
    week2 = run(after_gap(week1, timedelta(hours=49)), 50)
    found = segments.derive_segments(week1 + week2)
    assert [(s.first, s.stop) for s in found] == [(0, 200), (200, 250)]
    assert found[0].start_time == T0
    assert found[1].start_time == week2[0]
    assert found[1].end_time == week2[-1]


def test_gap_of_exactly_the_threshold_splits_and_just_under_does_not():
    a = run(T0, 10)
    at = a + run(after_gap(a, config.GAP_THRESHOLD), 10)
    under = a + run(after_gap(a, config.GAP_THRESHOLD - M5), 10)
    assert len(segments.derive_segments(at)) == 2
    assert len(segments.derive_segments(under)) == 1


def test_segments_are_adjacent_disjoint_and_cover_all_bars():
    times = run(T0, 5)
    times += run(after_gap(times, timedelta(hours=10)), 7)
    times += run(after_gap(times, timedelta(hours=50)), 3)
    found = segments.derive_segments(times)
    assert found[0].first == 0 and found[-1].stop == len(times)
    assert all(a.stop == b.first for a, b in itertools.pairwise(found))
    assert [len(s) for s in found] == [5, 7, 3]


def test_split_is_chronological_adjacent_and_covers_everything():
    split = segments.derive_split(1000)
    assert [(p.name, p.first, p.stop) for p in split.periods] == [
        ("train", 0, 700),
        ("validation", 700, 850),
        ("test", 850, 1000),
    ]


@pytest.mark.parametrize("n", [1, 7, 99, 101, 12345])
def test_split_always_partitions_all_bars(n):
    p = segments.derive_split(n).periods
    assert p[0].first == 0 and p[-1].stop == n
    assert all(a.stop == b.first for a, b in itertools.pairwise(p))
    assert sum(len(x) for x in p) == n


def test_segments_within_a_period_are_clipped_to_it():
    times = run(T0, 10)
    times += run(after_gap(times, timedelta(hours=49)), 10)
    s = segments.derive(times)
    assert s.bar_count == 20
    assert s.split.test.first == 17
    in_test = s.segments_within(s.split.test)
    assert [(x.first, x.stop) for x in in_test] == [(17, 20)]
    in_train = s.segments_within(s.split.train)
    assert [(x.first, x.stop) for x in in_train] == [(0, 10), (10, 14)]


def test_load_reads_the_stored_bars_and_render_lists_boundaries(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    times = run(T0, 10)
    times += run(after_gap(times, timedelta(hours=49)), 10)
    cols = {c: [1.0] * 20 for c in bars.PRICE_COLUMNS}
    path = bars.bars_path()
    path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"time": times, **cols}, schema=bars.SCHEMA), path)
    s = segments.load()
    assert [(x.first, x.stop) for x in s.segments] == [(0, 10), (10, 20)]
    text = s.render()
    assert "segments: 2" in text
    assert "train" in text and "validation" in text and "test" in text
    assert "2 days, 1:00:00" in text  # the Gap that closed segment 1
