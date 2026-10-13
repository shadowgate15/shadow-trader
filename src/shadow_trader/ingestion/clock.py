"""Per-week raw-clock inference for HistData ticks.

HistData stamps ticks on a clock that is New York local time (with DST) in most
weeks but behaves as fixed UTC-5 in the weeks from 2019 where US and EU daylight
saving differ (see ADR 0002). The US clock change happens while the market is
closed, so every Session week has one offset and the choice is made per week:
the first tick of the Sunday session and the last tick of the week are compared
with the 17:00 New York open and close under each candidate rule.

A candidate is *contradicted* when the week has a tick outside the Session by
more than ``TOLERANCE`` (before the open or after the close); thin or holiday
weeks only ever deviate inwards, which says nothing about the clock. A week with
no surviving candidate fits no rule and fails. Otherwise it is decided when one
candidate survives (winter: the rules coincide), or when one candidate fits both
ends, or, weaker, when only one candidate fits an end (recorded as
``single_end``, unless that choice disagrees with the calendar rule and the
calendar's is not contradicted, when the calendar wins and the week is recorded
as ``calendar``). A week still undecided (no end fits, or the ends fit different
rules) inherits the rule of its adjacent weeks if every one that exists is
decided on a rule and they agree. Failing that, the calendar rule (see
``calendar_offset``) decides it, provided the week's ticks do not contradict it
(recorded as ``calendar``); any other week fails.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from .session import NY, open_of, week_sunday

ZURICH = ZoneInfo("Europe/Zurich")
ERA_SWITCH = date(2018, 12, 16)  # first Session week stamped on the EU calendar

FIXED_OFFSET = timedelta(hours=-5)
TOLERANCE = timedelta(minutes=30)
NEW_YORK_LOCAL = "new_york_local"
FIXED_UTC_MINUS_5 = "fixed_utc_minus_5"
_LEAD = timedelta(hours=10)  # Sunday ticks before 10:00 belong to the week before
_SUNDAY_NOON = "12"


class WeekClockError(Exception):
    """Weeks whose raw clock cannot be inferred; ``failures`` maps week to reason."""

    def __init__(self, failures: Mapping[date, str]) -> None:
        self.failures = dict(failures)
        first, reason = next(iter(self.failures.items()))
        more = len(self.failures) - 1
        suffix = (
            f" (+{more} more week(s): {_listed(list(self.failures)[1:])})"
            if more
            else ""
        )
        super().__init__(f"week of {first}: {reason}{suffix}")


def _listed(weeks: list[date], cap: int = 10) -> str:
    shown = ", ".join(str(week) for week in weeks[:cap])
    return shown + (", ..." if len(weeks) > cap else "")


def week_of(naive: datetime) -> date:
    """The Sunday that keys the Session week a raw (naive) timestamp belongs to."""
    return week_sunday((naive - _LEAD).date())


def _naive(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y%m%d %H%M%S")


def ny_offset(sunday: date) -> timedelta:
    """New York's UTC offset for the Session that opens on ``sunday``."""
    utcoffset = open_of(sunday).astimezone(NY).utcoffset()
    assert utcoffset is not None
    return utcoffset


def rule_of(sunday: date, offset: timedelta) -> str | None:
    """The rule ``offset`` stands for in that week; None when the rules coincide."""
    ny = ny_offset(sunday)
    if ny == FIXED_OFFSET:
        return None
    return NEW_YORK_LOCAL if offset == ny else FIXED_UTC_MINUS_5


def calendar_offset(sunday: date) -> timedelta:
    """The raw-clock offset the calendar predicts for the Session opening ``sunday``.

    Before ``ERA_SWITCH`` the stamps are New York local time; from then they are
    Europe/Zurich local time minus six hours (UTC-5 in EU winter, UTC-4 in EU
    summer). Independent of any tick, so it can cross-check the inference.
    """
    if sunday < ERA_SWITCH:
        return ny_offset(sunday)
    utcoffset = open_of(sunday).astimezone(ZURICH).utcoffset()
    assert utcoffset is not None
    return utcoffset - timedelta(hours=6)


def calendar_rule(sunday: date) -> str | None:
    """The rule the calendar predicts for that week; None when the rules coincide."""
    return rule_of(sunday, calendar_offset(sunday))


def offset_of(sunday: date, rule: str) -> timedelta:
    return ny_offset(sunday) if rule == NEW_YORK_LOCAL else FIXED_OFFSET


@dataclass(frozen=True)
class WeekClock:
    """The raw-clock UTC offset chosen for each Session week of one Instrument."""

    offsets: Mapping[date, timedelta]
    inherited: frozenset[date] = frozenset()
    single_end: frozenset[date] = frozenset()
    calendar: frozenset[date] = frozenset()

    def to_utc(self, naive: datetime) -> datetime:
        return (naive - self.offsets[week_of(naive)]).replace(tzinfo=UTC)

    def minutes(self) -> dict[str, int]:
        return {
            week.isoformat(): int(offset / timedelta(minutes=1))
            for week, offset in sorted(self.offsets.items())
        }

    def rule_counts(self) -> dict[str, int]:
        """Weeks per rule; ``either`` where New York and UTC-5 coincide (winter)."""
        counts = {NEW_YORK_LOCAL: 0, FIXED_UTC_MINUS_5: 0, "either": 0}
        for week, offset in self.offsets.items():
            counts[rule_of(week, offset) or "either"] += 1
        return counts

    def provenance(self) -> dict[str, object]:
        """What was inferred (weeks listed by exception), for manifest and report."""
        return {
            "offsets": self.minutes(),
            "inherited": sorted(week.isoformat() for week in self.inherited),
            "single_end": sorted(week.isoformat() for week in self.single_end),
            "calendar": sorted(week.isoformat() for week in self.calendar),
        }

    def summary(self) -> dict[str, object]:
        return {"rule_counts": self.rule_counts(), **self.provenance()}


class WeekEvidence:
    """Accumulates, month by month, the ticks that identify each week's clock."""

    def __init__(self) -> None:
        # naive day -> [first stamp, last stamp, first Sunday stamp from 12:00]
        self._days: dict[str, list[str | None]] = {}

    def add_text(self, text: str) -> None:
        days = self._days
        current: str | None = None
        record: list[str | None] | None = None
        sunday = False
        for line in text.splitlines():
            day = line[:8]
            if day != current:
                current = day
                try:
                    sunday = datetime.strptime(day, "%Y%m%d").weekday() == 6
                except ValueError:
                    record = None  # parse_ticks reports the bad row
                    continue
                record = days.setdefault(day, [line[:15], None, None])
            if record is None:
                continue
            record[1] = line[:15]
            if sunday and record[2] is None and line[9:11] >= _SUNDAY_NOON:
                record[2] = line[:15]

    def _weeks(self) -> dict[date, list[str | None]]:
        """Per week: [first Sunday tick, last tick]; either may be missing."""
        weeks: dict[date, list[str | None]] = {}
        for day, (first, last, sunday_open) in sorted(self._days.items()):
            assert first is not None and last is not None
            for stamp in (first, last):  # every tick must find its week's offset
                weeks.setdefault(week_of(_naive(stamp)), [None, None])
            when = datetime.strptime(day, "%Y%m%d")
            if when.weekday() == 5:
                continue  # Saturday ticks are outside any Session
            if when.weekday() == 6 and sunday_open is None:
                continue  # only strays before Sunday noon: no week of their own
            entry = weeks.setdefault(week_sunday(when.date()), [None, None])
            if when.weekday() == 6:
                entry[0] = entry[0] or sunday_open
            entry[1] = last
        return weeks

    def infer(self) -> WeekClock:
        """The clock of every week seen; raises ``WeekClockError`` for the rest."""
        clock, failures = self.analyse()
        if failures:
            raise WeekClockError(failures)
        return clock

    def analyse(self) -> tuple[WeekClock, dict[date, str]]:
        """The clock of the weeks that could be decided, and why the others failed."""
        weeks = self._weeks()
        decided: dict[date, timedelta] = {}
        single_end: set[date] = set()
        undecided: dict[date, list[timedelta]] = {}
        from_calendar: set[date] = set()
        failures: dict[date, str] = {}
        for sunday, (first, last) in sorted(weeks.items()):
            alive, reason = _alive(sunday, first, last)
            fits = {o: _ends_fitting(sunday, first, last, o) for o in alive}
            if not alive:
                failures[sunday] = reason
            elif len(alive) == 1:
                decided[sunday] = alive[0]
            elif both := [o for o in alive if fits[o] == 2]:
                decided[sunday] = both[0]
            elif len(backed := [o for o in alive if fits[o]]) == 1:
                expected = calendar_offset(sunday)
                if backed[0] != expected and expected in alive:
                    decided[sunday] = expected  # one end is weak evidence: defer
                    from_calendar.add(sunday)
                else:
                    decided[sunday] = backed[0]
                    single_end.add(sunday)
            else:
                undecided[sunday] = alive
        inherited: set[date] = set()
        direct = dict(decided)  # inheritance never chains through inherited weeks
        for sunday, alive in undecided.items():
            offset, reason = _inherit(sunday, alive, direct, weeks)
            if offset is None and (expected := calendar_offset(sunday)) in alive:
                from_calendar.add(sunday)
                decided[sunday] = expected
            elif offset is None:
                failures[sunday] = reason
            else:
                inherited.add(sunday)
                decided[sunday] = offset
        return (
            WeekClock(
                decided,
                frozenset(inherited),
                frozenset(single_end),
                frozenset(from_calendar),
            ),
            dict(sorted(failures.items())),
        )


def _deltas(
    sunday: date, first: str | None, last: str | None, offset: timedelta
) -> tuple[timedelta | None, timedelta | None]:
    """Signed distance of the first/last tick from the open/close under ``offset``."""
    opened = open_of(sunday)
    closed = open_of(sunday + timedelta(days=5))
    return (
        (_naive(first) - offset).replace(tzinfo=UTC) - opened if first else None,
        (_naive(last) - offset).replace(tzinfo=UTC) - closed if last else None,
    )


def _alive(
    sunday: date, first: str | None, last: str | None
) -> tuple[list[timedelta], str]:
    """The candidate offsets no tick contradicts, and why the others died."""
    alive: list[timedelta] = []
    notes: list[str] = []
    for offset in dict.fromkeys((ny_offset(sunday), FIXED_OFFSET)):
        early, late = _deltas(sunday, first, last, offset)
        label = f"UTC{int(offset / timedelta(hours=1)):+d}"
        if early is not None and early < -TOLERANCE:
            minutes = -early / timedelta(minutes=1)
            notes.append(f"{label}: first tick {minutes:.0f} min before the open")
        elif late is not None and late > TOLERANCE:
            minutes = late / timedelta(minutes=1)
            notes.append(f"{label}: last tick {minutes:.0f} min after the close")
        else:
            alive.append(offset)
    return alive, "fits no raw-clock rule (" + "; ".join(notes) + ")"


def _ends_fitting(
    sunday: date, first: str | None, last: str | None, offset: timedelta
) -> int:
    """How many of the open and close ticks sit within tolerance under ``offset``."""
    return sum(
        delta is not None and abs(delta) <= TOLERANCE
        for delta in _deltas(sunday, first, last, offset)
    )


def _inherit(
    sunday: date,
    alive: list[timedelta],
    direct: Mapping[date, timedelta],
    weeks: Mapping[date, object],
) -> tuple[timedelta | None, str]:
    """The adjacent weeks' rule if they agree.

    An adjacent week that is present but undecided, or in winter where the rules
    coincide and say nothing, blocks: the rule may change at that boundary.
    """
    cannot = "cannot be classified (no clean open and close)"
    rules: set[str] = set()
    for step in (-7, 7):
        other = sunday + timedelta(days=step)
        if other in direct:
            if (rule := rule_of(other, direct[other])) is None:
                return None, f"{cannot} and the week of {other} is in winter (no rule)"
            rules.add(rule)
        elif other in weeks:
            return None, f"{cannot} and the week of {other} is not classified either"
    if not rules:
        return None, f"{cannot} and no adjacent week gives a rule to inherit"
    if len(rules) > 1:
        return None, f"{cannot} and the adjacent weeks follow different rules"
    offset = offset_of(sunday, next(iter(rules)))
    if offset not in alive:
        return None, "the adjacent weeks' rule is contradicted by this week's ticks"
    return offset, ""
