"""The Run summary: counts and totals read straight off the Ledger.

One accumulator, ``EpisodeAccumulator``, folds ``StepEvents`` (one per stepped Bar)
into an ``EpisodeSummary``. The env runs it live to fill ``info["episode"]``;
``summarize(ledger)`` replays the Ledger's rows through the very same accumulator,
so the two cannot disagree. A ``RunSummary`` totals the per-episode summaries.

Only counts and sums live here: no ratios, drawdown, win rate or baselines (metric
definitions belong to evaluation). Costs are USD totals, positive = paid: the
spread and slippage inside the fills and the rollover charges.

Definitions: ``bars`` are stepped Bars, ``decisions`` those that were Decision Bars
(usable), ``clips`` steps whose order was cut back by the margin limit,
``closeouts`` steps that closed on margin closeout, ``forced_flats`` Session-final
force-flat fills, ``synthetic_liquidations`` forced flats on an unfillable Bar,
``gap_spans`` maximal runs of consecutive carried-mark Bars (no usable price) and
``longest_blind_run`` the most consecutive non-Decision Bars.

This module is Gymnasium-free, public, and outside the seam.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .endings import ENDINGS
from .engine import ClosedLot, Fill, StepEvents
from .ledger import Ledger

__all__ = [
    "ENDINGS",
    "SUMMARY_VERSION",
    "EpisodeAccumulator",
    "EpisodeSummary",
    "RunSummary",
    "summarize",
    "summary_document",
]

SUMMARY_VERSION = 1


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    """Counts and totals of one episode."""

    episode_id: int
    session_id: int
    ending: str | None
    bars: int
    decisions: int
    fills: int
    clips: int
    closeouts: int
    forced_flats: int
    synthetic_liquidations: int
    gap_spans: int
    closed_lots: int
    longest_blind_run: int
    spread_cost: float
    slippage_cost: float
    rollover_cost: float
    start_equity: float
    final_equity: float
    net_pnl: float


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Counts and totals of a run: its episodes' summaries and their sums."""

    episodes: tuple[EpisodeSummary, ...]
    episode_count: int
    endings: dict[str, int]
    bars: int
    decisions: int
    fills: int
    clips: int
    closeouts: int
    forced_flats: int
    synthetic_liquidations: int
    gap_spans: int
    closed_lots: int
    longest_blind_run: int
    spread_cost: float
    slippage_cost: float
    rollover_cost: float
    start_equity: float
    final_equity: float
    net_pnl: float


class EpisodeAccumulator:
    """Folds one episode's ``StepEvents`` into an ``EpisodeSummary``."""

    def __init__(self) -> None:
        self._open = False

    def begin(self, episode_id: int, session_id: int, start_equity: float) -> None:
        self._open = True
        self._episode_id = episode_id
        self._session_id = session_id
        self._start = start_equity
        self._final = start_equity
        self._bars = self._decisions = self._fills = self._clips = 0
        self._closeouts = self._forced = self._synthetic = 0
        self._gap_spans = self._closed = self._longest = 0
        self._spread = self._slippage = self._rollover = 0.0
        self._in_gap = False

    def add(self, events: StepEvents, *, is_decision: bool, blind_run: int) -> None:
        """Fold one stepped Bar; ``blind_run`` is the non-Decision run so far."""
        self._bars += 1
        self._decisions += int(is_decision)
        self._clips += int(events.clipped)
        self._closeouts += int(events.closeout)
        self._closed += len(events.closed_lots)
        self._longest = max(self._longest, blind_run)
        self._gap_spans += int(events.carried_mark and not self._in_gap)
        self._in_gap = events.carried_mark
        self._rollover += events.rollover_charge
        self._final = events.equity
        for fill in events.fills:
            self._fills += 1
            self._forced += int(fill.forced_flat)
            self._synthetic += int(fill.unfilled_forced_flat)
            self._spread += fill.spread_cost
            self._slippage += fill.slippage_cost

    def finish(self, ending: str | None) -> EpisodeSummary:
        """The episode's summary so far, labelled with why it ended."""
        self._open = False
        return EpisodeSummary(
            episode_id=self._episode_id,
            session_id=self._session_id,
            ending=ending,
            bars=self._bars,
            decisions=self._decisions,
            fills=self._fills,
            clips=self._clips,
            closeouts=self._closeouts,
            forced_flats=self._forced,
            synthetic_liquidations=self._synthetic,
            gap_spans=self._gap_spans,
            closed_lots=self._closed,
            longest_blind_run=self._longest,
            spread_cost=self._spread,
            slippage_cost=self._slippage,
            rollover_cost=self._rollover,
            start_equity=self._start,
            final_equity=self._final,
            net_pnl=self._final - self._start,
        )


def _events_from_rows(
    step: dict[str, Any],
    fills: list[dict[str, Any]],
    closed: list[dict[str, Any]],
) -> StepEvents:
    """Rebuild the ``StepEvents`` facts a summary needs from Ledger rows."""
    return StepEvents(
        fills=tuple(
            Fill(
                instrument=0,
                units=f["units"],
                price=f["price"],
                spread_cost=f["spread_cost"],
                slippage_cost=f["slippage_cost"],
                clipped=f["clipped"],
                closeout=f["closeout"],
                forced_flat=f["forced_flat"],
                halt=f["halt"],
                unfilled_forced_flat=f["unfilled_forced_flat"],
            )
            for f in fills
        ),
        equity=step["equity"],
        used_margin=step["used_margin"],
        rollover_charge=step["rollover_charge"],
        positions=(step["position_units"],),
        mark_prices=(step["mark_price"],),
        closed_lots=tuple(
            ClosedLot(
                instrument=0,
                units=c["units"],
                open_price=c["open_price"],
                close_price=c["close_price"],
                realised_pnl=c["realised_pnl"],
            )
            for c in closed
        ),
        clipped=step["clipped"],
        closeout=step["closeout"],
        carried_mark=step["carried_mark"],
    )


def _group(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["step"], []).append(row)
    return grouped


def summarize(ledger: Ledger) -> RunSummary:
    """The Run summary of ``ledger``, recomputed purely from its tables."""
    fills = _group(ledger.fills.to_dicts())
    closed = _group(ledger.closed_lots.to_dicts())
    episodes: list[dict[str, Any]] = ledger.episodes.to_dicts()
    if not episodes:  # a ledger without an episodes table is one open episode
        episodes = [
            {
                "episode_id": 0,
                "session_id": 0,
                "start_equity": ledger.starting_equity,
                "ending": None,
            }
        ]
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in ledger.steps.to_dicts():
        by_episode.setdefault(row["episode_id"], []).append(row)
    summaries = []
    for episode in episodes:
        acc = EpisodeAccumulator()
        acc.begin(episode["episode_id"], episode["session_id"], episode["start_equity"])
        for row in by_episode.get(episode["episode_id"], []):
            events = _events_from_rows(
                row, fills.get(row["step"], []), closed.get(row["step"], [])
            )
            acc.add(events, is_decision=row["is_decision"], blind_run=row["blind_run"])
        summaries.append(acc.finish(episode["ending"]))
    return _run_summary(tuple(summaries))


def _run_summary(episodes: tuple[EpisodeSummary, ...]) -> RunSummary:
    endings = dict.fromkeys(ENDINGS, 0)
    for e in episodes:
        if e.ending is not None:
            endings[e.ending] = endings.get(e.ending, 0) + 1
    return RunSummary(
        episodes=episodes,
        episode_count=len(episodes),
        endings=endings,
        bars=sum(e.bars for e in episodes),
        decisions=sum(e.decisions for e in episodes),
        fills=sum(e.fills for e in episodes),
        clips=sum(e.clips for e in episodes),
        closeouts=sum(e.closeouts for e in episodes),
        forced_flats=sum(e.forced_flats for e in episodes),
        synthetic_liquidations=sum(e.synthetic_liquidations for e in episodes),
        gap_spans=sum(e.gap_spans for e in episodes),
        closed_lots=sum(e.closed_lots for e in episodes),
        longest_blind_run=max((e.longest_blind_run for e in episodes), default=0),
        spread_cost=sum(e.spread_cost for e in episodes),
        slippage_cost=sum(e.slippage_cost for e in episodes),
        rollover_cost=sum(e.rollover_cost for e in episodes),
        start_equity=episodes[0].start_equity if episodes else 0.0,
        final_equity=episodes[-1].final_equity if episodes else 0.0,
        net_pnl=sum(e.net_pnl for e in episodes),
    )


def summary_document(summary: RunSummary) -> dict[str, Any]:
    """The JSON-ready ``summary.json`` body for ``summary``."""
    return {"summary_version": SUMMARY_VERSION, "summary": asdict(summary)}
