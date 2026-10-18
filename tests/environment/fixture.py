"""The small checked-in fixture Dataset for the environment tests.

A prepared Dataset (one Instrument, EURUSD, 5-minute Bars) plus its Scaler lives in
``tests/environment/fixtures/prepared``. It has nine Sessions; only the last crosses a
17:00 New York rollover:

- Sessions 1 to 3 hold ``SESSION_BARS`` Bars each (warm-up is ``WARM_UP`` Bars, so
  Bar ``WARM_UP`` is the first usable one);
- Session 4 holds only ``SHORT_SESSION_BARS`` Bars, all in warm-up, so it has no
  usable Bar;
- Sessions 5 and 6 hold ``FINAL_SESSION_BARS`` Bars each and end on the Session's
  true final Bar (Friday 21:55 UTC), so Session-end force-flat applies. Sessions 1
  to 3 stop short of the weekly close: their last Bar is not Session-final;
- Session 7 (``OUTAGE_SESSION``) holds ``OUTAGE_BARS`` Bars with a mid-Session
  outage: Gap rows at ``OUTAGE_GAPS`` and one distrusted (``price_spike``) priced
  Bar at ``DISTRUSTED_SLOT``. The feature look-back keeps the Bars after each
  non-fillable one non-usable for a while, so there are several blind runs;
- Session 8 (``TAIL_SESSION``) holds ``FINAL_SESSION_BARS`` Bars ending on the true
  final Bar, the last ``TAIL_GAPS`` of them Gaps (an unfillable Session-final Bar);
- Session 9 (``ROLLOVER_SESSION``) holds ``ROLLOVER_BARS`` Bars, so it crosses the
  Monday 17:00 New York rollover Bar at slot ``ROLLOVER_SLOT``.

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

from shadow_trader.ingestion.bars import Bar, gap_bar
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
    *(datetime(2024, 2, day, 22, 0, tzinfo=UTC) for day in (4, 11, 18, 25)),
    datetime(2024, 3, 3, 22, 0, tzinfo=UTC),
)
# Sessions 5 and 6 run to the weekly close (Friday 21:55 UTC is their Session-final
# Bar); their earlier Bars are not in the fixture, so they are short.
FINAL_SESSION_BARS = 60
FINAL_SESSIONS = (4, 5, 7)
OUTAGE_SESSION = 6
OUTAGE_BARS = 80
OUTAGE_GAPS = range(30, 35)  # slots of the Gap rows in the outage Session
DISTRUSTED_SLOT = 50  # a priced Bar carrying a distrusted quality flag
TAIL_SESSION = 7
TAIL_GAPS = 3  # trailing Gap rows, the last being the Session-final Bar
ROLLOVER_SESSION = 8
ROLLOVER_BARS = 300  # slot 288 is Monday 17:00 New York, a one-day rollover Bar
ROLLOVER_SLOT = 288
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
            if session == OUTAGE_SESSION:
                count = OUTAGE_BARS
            elif session == ROLLOVER_SESSION:
                count = ROLLOVER_BARS
            first = opened
        for i in range(count):
            slot_time = first + timedelta(minutes=5 * i)
            if (session == OUTAGE_SESSION and i in OUTAGE_GAPS) or (
                session == TAIL_SESSION and i >= count - TAIL_GAPS
            ):
                bars.append(gap_bar(INSTRUMENT, slot_time, opened))
                continue
            move = float(rng.normal(0.0, 3e-4))
            close = round(price * math.exp(move), 5)
            spread = round(0.0001 * (1 + i % 3), 5)
            bars.append(
                Bar(
                    instrument=INSTRUMENT,
                    open_time=slot_time,
                    open=price,
                    high=round(max(price, close) + 0.0001, 5),
                    low=round(min(price, close) - 0.0001, 5),
                    close=close,
                    spread_close=spread,
                    spread_mean=spread,
                    tick_count=20 + i % 7,
                    is_gap=False,
                    session_id=opened,
                    is_rollover=session == ROLLOVER_SESSION and i == ROLLOVER_SLOT,
                    quality_flags=(
                        ("price_spike",)
                        if session == OUTAGE_SESSION and i == DISTRUSTED_SLOT
                        else ()
                    ),
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
