"""Gap stepping through the env, on the fixture's outage Sessions.

Outage Session (ordinal 6, Bar slots in the Session): usable 6-29, Gap 30-34,
priced but not usable 35-40, usable 41-49, distrusted 50 (priced, unfillable) and
not usable through 56, usable 57-79. Tail Session (ordinal 7, ends on the weekly
close): usable 6-56, Gaps 57-59 with 59 the Session-final Bar.
"""

from __future__ import annotations

import polars as pl
import pytest

from shadow_trader.environment import verify_ledger
from shadow_trader.environment.env import TradingEnv, make_env
from shadow_trader.environment.env_config import EnvConfig, EpisodeConfig

from .fixture import DISTRUSTED_SLOT, OUTAGE_SESSION, TAIL_SESSION, load_fixture

FLAT, LONG = 1, 2  # indexes into the default level set (-1, 0, 1)


@pytest.fixture(scope="module")
def prepared():
    return load_fixture()


def build(prepared, window: int = 12, seed: int = 0) -> TradingEnv:
    dataset, scaler = prepared
    config = EnvConfig(episode=EpisodeConfig(window_bars=window))
    return make_env(dataset, scaler, config, seed=seed, record_ledger=True)


def play(env: TradingEnv, session: int, start_bar: int, action: int) -> list[dict]:
    env.reset(options={"session": session, "start_bar": start_bar})
    infos, done = [], False
    while not done:
        _, _, terminated, truncated, info = env.step(action)
        infos.append(info)
        done = terminated or truncated
    return infos


def test_one_step_spans_the_outage_up_to_the_next_decision_bar(prepared) -> None:
    env = build(prepared)
    infos = play(env, OUTAGE_SESSION, 28, LONG)

    assert [i["bars_advanced"] for i in infos[:3]] == [1, 12, 1]  # 29, 30..41, 42
    span = infos[1]
    assert span["gap_bars"] == 5
    assert span["unfillable_bars"] == 5
    assert span["blind_run"] == 11  # every spanned Bar but the decision Bar
    assert infos[0]["gap_bars"] == infos[0]["unfillable_bars"] == 0
    assert infos[0]["blind_run"] == 0


def test_every_stepped_bar_is_a_ledger_row_with_its_decision_and_blind_flags(
    prepared,
) -> None:
    env = build(prepared)
    play(env, OUTAGE_SESSION, 28, LONG)
    ledger = env.ledger()
    verify_ledger(ledger)
    steps = ledger.steps

    assert steps.height == 79 - 28  # one row per Bar to the Session's end
    rows = steps.filter(pl.col("step").is_between(1, 12))  # slots 30..41
    assert rows["is_decision"].to_list() == [False] * 11 + [True]
    assert rows["blind_run"].to_list() == [*range(1, 12), 0]
    assert rows["carried_mark"].to_list() == [True] * 5 + [False] * 7
    assert steps.filter(pl.col("is_decision"))["blind_run"].sum() == 0


def test_no_fill_lands_on_an_unfillable_bar_and_marks_are_carried(prepared) -> None:
    env = build(prepared)
    play(env, OUTAGE_SESSION, 28, LONG)
    ledger = env.ledger()
    carried = ledger.steps.filter(pl.col("carried_mark"))

    assert carried.height == 5 + 1  # the Gap and the distrusted Bar
    assert set(carried["step"]).isdisjoint(set(ledger.fills["step"]))
    gap_rows = carried.filter(pl.col("step") < 14)
    assert gap_rows["mark_price"].n_unique() == 1  # the last trusted close
    assert gap_rows["equity"].n_unique() == 1


def test_a_target_chosen_before_an_outage_fills_at_the_first_fillable_open(
    prepared,
) -> None:
    env = build(prepared)
    play(env, OUTAGE_SESSION, 29, LONG)
    ledger = env.ledger()

    # the order waits through the Gaps (slots 30-34) and fills at slot 35's open:
    # step 0 is slot 30, so slot 35 is step 5
    assert ledger.fills["step"][0] == 5
    assert ledger.fills["units"][0] > 0
    assert ledger.steps["position_units"].to_list()[:5] == [0] * 5


def test_a_distrusted_priced_bar_counts_as_unfillable_not_as_a_gap(prepared) -> None:
    env = build(prepared)
    infos = play(env, OUTAGE_SESSION, DISTRUSTED_SLOT - 1, FLAT)

    (span,) = [i for i in infos if i["unfillable_bars"]]
    assert span["gap_bars"] == 0
    assert span["unfillable_bars"] == 1
    assert span["bars_advanced"] == 57 - DISTRUSTED_SLOT + 1  # slots 50..57


def test_an_unfillable_session_final_bar_liquidates_at_the_last_trusted_close(
    prepared,
) -> None:
    env = build(prepared)
    infos = play(env, TAIL_SESSION, 50, LONG)
    ledger = env.ledger()
    verify_ledger(ledger)

    last = infos[-1]
    assert last["ending"] == "session_end"
    assert last["bars_advanced"] == 3  # slots 57-59 follow the decision at 56
    assert last["unfilled_forced_flat"] is True
    assert last["gap_bars"] == 3
    fills = ledger.fills.filter(pl.col("unfilled_forced_flat"))
    assert fills.height == 1
    assert fills["step"][0] == ledger.steps.height - 1
    assert ledger.steps["position_units"][-1] == 0
    trusted = ledger.steps["mark_price"][-4]  # the last fillable Bar (slot 56)
    assert fills["price"][0] == pytest.approx(trusted - 1e-5, abs=1e-9)


def test_training_windows_stay_n_wall_clock_bars_and_the_step_count_varies(
    prepared,
) -> None:
    env = build(prepared, window=12)
    step_counts: set[int] = set()
    ended_in_a_run = False
    for seed in range(120):
        env.reset(seed=seed)
        done, steps, advanced = False, 0, 0
        while not done:
            _, _, terminated, truncated, info = env.step(FLAT)
            done = terminated or truncated
            steps += 1
            advanced += info["bars_advanced"]
        assert advanced == 12
        step_counts.add(steps)
        ledger = env.ledger()
        verify_ledger(ledger)
        if not ledger.steps["is_decision"][-1]:
            ended_in_a_run = True
            assert info["ending"] == "window_end"
    assert len(step_counts) > 1  # a window with an outage takes fewer steps
    assert ended_in_a_run
