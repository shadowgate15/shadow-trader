"""Progress events: how long operations tell a front end they are working.

Library code reports phases, steps and tolerated warnings to a ``ProgressSink``
and knows nothing about rendering (the CLI draws them with ``rich``). The
default sink does nothing. Sinks are called per month, week or phase, never per
tick.
"""

from __future__ import annotations

from typing import Protocol


class ProgressSink(Protocol):
    def start(self, phase: str, total: int | None = None) -> None:
        """A phase begins; ``total`` steps are expected (None: unknown)."""

    def advance(self, detail: str = "", outcome: str | None = None) -> None:
        """One step of the current phase finished.

        ``detail`` names it (e.g. ``EURUSD 202401``); ``outcome`` classifies it
        for tallies (fetch: fetched, cached, absent, unsettled, rejected).
        """

    def warn(self, message: str) -> None:
        """A condition the run tolerates, surfaced as it happens."""

    def status(self, text: str) -> None:
        """Replace the one-line status of an open-ended run (live)."""

    def done(self) -> None:
        """The current phase ended."""


class NullProgress:
    """The default sink: ignores every event."""

    def start(self, phase: str, total: int | None = None) -> None:
        pass

    def advance(self, detail: str = "", outcome: str | None = None) -> None:
        pass

    def warn(self, message: str) -> None:
        pass

    def status(self, text: str) -> None:
        pass

    def done(self) -> None:
        pass


NULL_PROGRESS = NullProgress()
