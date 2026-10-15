import pickle

from shadow_trader.ingestion.backfill import build_chunk, infer_week_clock
from shadow_trader.ingestion.dataset import RawCache

from .test_backfill import fixture_csv, install_raw, make_config


def test_chunk_returns_picklable_data_and_writes_nothing(tmp_path):
    config = make_config(tmp_path, "EURUSD", "2024-01")
    install_raw(config, "EURUSD", "202401", fixture_csv("eurusd_winter"))
    cache = RawCache(config.paths.data_root)
    clock = infer_week_clock(cache, "EURUSD", ["202401"])
    before = sorted(p for p in config.paths.data_root.rglob("*"))

    chunk = build_chunk(
        cache,
        "EURUSD",
        5,
        config.instruments["EURUSD"].price_scale,
        config.anomaly_rule,
        clock,
        ["202401"],
    )

    assert chunk.bars
    assert chunk.first_ticks
    assert pickle.loads(pickle.dumps(chunk)) == chunk
    assert sorted(config.paths.data_root.rglob("*")) == before
