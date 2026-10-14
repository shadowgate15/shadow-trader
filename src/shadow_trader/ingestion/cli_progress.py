"""The ``rich`` renderer of ingestion progress events (CLI layer only).

Everything goes to the console it is given (stderr in the CLI), so stdout keeps
only the result lines. On an interactive terminal it draws a live progress bar;
anywhere else (pipes, CI, ``TERM=dumb``) it writes plain lines, one per month,
and never control characters.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from types import TracebackType

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.status import Status

QUIET, NORMAL, VERBOSE = 0, 1, 2
WARNING_CAP = 20  # warnings shown at the default level before folding the rest
PLAIN_EVERY = 25  # plain mode: a line every this many quiet (cached) steps
STATUS_EVERY_SECONDS = 60.0  # plain mode: at most one live status line per minute
_NOTABLE = {"fetched", "absent", "unsettled", "rejected"}


class RichProgress:
    """A ``ProgressSink`` that draws on a ``rich`` Console."""

    def __init__(
        self,
        console: Console,
        level: int = NORMAL,
        *,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.console = console
        self.level = level
        self.phase = ""  # the phase now running (kept after a failure)
        self._monotonic = monotonic
        self._interactive = (
            level > QUIET and console.is_terminal and not console.is_dumb_terminal
        )
        self._bar: Progress | None = None
        self._task: TaskID | None = None
        self._status: Status | None = None
        self._total: int | None = None
        self._count = 0
        self._began = 0.0
        self._outcomes: Counter[str] = Counter()
        self._warned = 0
        self._last_status = float("-inf")

    def __enter__(self) -> RichProgress:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Stop any live display and fold away suppressed-warning counts."""
        if self._bar is not None:
            self._bar.stop()
            self._bar = None
        if self._status is not None:
            self._status.stop()
            self._status = None
        if self._warned > WARNING_CAP and self.level == NORMAL:
            self.console.print(
                f"... {self._warned - WARNING_CAP} further warning(s) not shown; "
                "run with -v to see them all"
            )
        self._warned = 0

    def _print(self, message: str) -> None:
        if self.level > QUIET:
            self.console.print(message, markup=False, highlight=False, soft_wrap=True)

    def start(self, phase: str, total: int | None = None) -> None:
        self.phase = phase
        self._total, self._count = total, 0
        self._outcomes = Counter()
        self._began = self._monotonic()
        if self.level == QUIET:
            return
        if self._interactive:
            if self._bar is None:
                self._bar = Progress(
                    SpinnerColumn(),
                    TextColumn("{task.description}"),
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                    TextColumn("{task.fields[detail]}"),
                    console=self.console,
                )
                self._bar.start()
            self._task = self._bar.add_task(phase, total=total, detail="")
        else:
            self._print(f"{phase}: {total} step(s)" if total else f"{phase}...")

    def advance(self, detail: str = "", outcome: str | None = None) -> None:
        self._count += 1
        if outcome is not None:
            self._outcomes[outcome] += 1
        if self.level == QUIET:
            return
        quiet_step = outcome == "cached" and self.level < VERBOSE
        if self._interactive and self._bar is not None and self._task is not None:
            self._bar.update(self._task, advance=1, detail=detail)
            if outcome is not None and not quiet_step:
                self.console.print(f"{outcome} {detail}", markup=False, highlight=False)
            return
        shown = outcome in _NOTABLE or self.level == VERBOSE or outcome is None
        if shown or self._count % PLAIN_EVERY == 0:
            of = f"/{self._total}" if self._total else ""
            label = f"{outcome} {detail}" if outcome else detail
            self._print(f"[{self._count}{of}] {label}")

    def warn(self, message: str) -> None:
        if self.level == QUIET:
            return
        self._warned += 1
        if self.level == NORMAL and self._warned > WARNING_CAP:
            return
        self._print(f"warning: {message}")

    def status(self, text: str) -> None:
        if self.level == QUIET:
            return
        if self._interactive:
            if self._status is None:
                self._status = self.console.status(text)
                self._status.start()
            else:
                self._status.update(text)
            return
        now = self._monotonic()
        if now - self._last_status >= STATUS_EVERY_SECONDS:
            self._last_status = now
            self._print(text)

    def done(self) -> None:
        elapsed = self._monotonic() - self._began
        if self._interactive and self._bar is not None and self._task is not None:
            if self._total is None:
                self._bar.update(self._task, total=1, completed=1)
            self._bar.refresh()
        elif self.level > QUIET:
            self._print(f"{self.phase}: done in {elapsed:.1f}s")
        if self._outcomes and self.level > QUIET:
            tally = ", ".join(f"{n} {kind}" for kind, n in self._outcomes.items())
            self._print(f"{self.phase}: {tally}")
