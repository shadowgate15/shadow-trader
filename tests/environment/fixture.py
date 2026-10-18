"""The small checked-in fixture Dataset for the environment tests.

A prepared Dataset (one Instrument, EURUSD, 5-minute Bars) plus its Scaler lives in
``tests/environment/fixtures/prepared``. It has six Sessions, none crossing a
17:00 New York rollover, so the engine can step every Bar:

- Sessions 1 to 3 hold ``SESSION_BARS`` Bars each (warm-up is ``WARM_UP`` Bars, so
  Bar ``WARM_UP`` is the first usable one);
- Session 4 holds only ``SHORT_SESSION_BARS`` Bars, all in warm-up, so it has no
  usable Bar;
- Sessions 5 and 6 hold ``FINAL_SESSION_BARS`` Bars each and end on the Session's
  true final Bar (Friday 21:55 UTC), so Session-end force-flat applies. Sessions 1
  to 3 stop short of the weekly close: their last Bar is not Session-final.

The feature spec is a small custom one (log returns over 1, 3 and 6 Bars) so the
warm-up is short. Later tickets that need Gaps, rollover or a closeout extend
``build_fixture`` and regenerate: ``uv run python -m tests.environment.fixture``.
"""

from __future__ import annotations

import math
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from shadow_trader.ingestion.bars import Bar
from shadow_trader.preparation.build import featured_frame
from shadow_trader.preparation.dataset import (
    PreparedDataset,
    open_prepared,
    plan_prepared,
)
from shadow_trader.preparation.scaler import (
    Scaler,
    fit_scaler,
    load_scaler,
    write_scaler,
)
from shadow_trader.preparation.spec import SPECS, FeatureSpec

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "prepared"
INSTRUMENT = "EURUSD"
SESSION_OPENS = (
    *(datetime(2024, 1, day, 22, 0, tzinfo=UTC) for day in (7, 14, 21, 28)),
    *(datetime(2024, 2, day, 22, 0, tzinfo=UTC) for day in (4, 11)),
)
# Sessions 5 and 6 run to the weekly close (Friday 21:55 UTC is their Session-final
# Bar); their earlier Bars are not in the fixture, so they are short.
FINAL_SESSION_BARS = 60
FINAL_SESSIONS = (4, 5)
SESSION_BARS = 280  # fewer than 288, the first rollover Bar of a Session
SHORT_SESSION_BARS = 5
WARM_UP = 6
SPEC = FeatureSpec(1, 5, SPECS[1].columns[:3])  # feat_ret_1, feat_ret_3, feat_ret_6
SEED = 20240107


def fixture_bars() -> list[Bar]:
    rng = np.random.default_rng(SEED)
    bars: list[Bar] = []
    price = 1.1
    for session, opened in enumerate(SESSION_OPENS):
        if session in FINAL_SESSIONS:
            count = FINAL_SESSION_BARS
            first = opened + timedelta(days=5) - timedelta(minutes=5 * count)
        else:
            count = SESSION_BARS if session != 3 else SHORT_SESSION_BARS
            first = opened
        for i in range(count):
            move = float(rng.normal(0.0, 3e-4))
            close = round(price * math.exp(move), 5)
            spread = round(0.0001 * (1 + i % 3), 5)
            bars.append(
                Bar(
                    instrument=INSTRUMENT,
                    open_time=first + timedelta(minutes=5 * i),
                    open=price,
                    high=round(max(price, close) + 0.0001, 5),
                    low=round(min(price, close) - 0.0001, 5),
                    close=close,
                    spread_close=spread,
                    spread_mean=spread,
                    tick_count=20 + i % 7,
                    is_gap=False,
                    session_id=opened,
                    is_rollover=False,
                    quality_flags=(),
                )
            )
            price = close
    return bars


def build_fixture(root: Path = FIXTURE_ROOT) -> None:
    if root.exists():
        shutil.rmtree(root)
    frame = featured_frame(fixture_bars(), SPEC)
    plan = plan_prepared({INSTRUMENT: frame}, parent_ingestion="fixture", spec=SPEC)
    scaler = fit_scaler(frame, SPEC, policy_hash="fixture", fit_dataset=plan.dataset_id)
    write_scaler(root, scaler)
    plan.write(
        root,
        {"role": "train", "scaler": {"id": scaler.scaler_id, "fit": True}},
        created_at=datetime(2024, 2, 1, tzinfo=UTC),
    )


def load_fixture(root: Path = FIXTURE_ROOT) -> tuple[PreparedDataset, Scaler]:
    (manifest,) = sorted((root / "datasets").glob("*/manifest.json"))
    dataset = open_prepared(root, manifest.parent.name)
    return dataset, load_scaler(root, dataset.manifest["scaler"]["id"])


if __name__ == "__main__":
    build_fixture()
    sys.stdout.write(f"wrote {FIXTURE_ROOT}\n")
