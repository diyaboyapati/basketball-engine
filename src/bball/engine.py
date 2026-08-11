from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .event_queue import ReorderBuffer
from .events import Event, period_length, period_start_elapsed, total_game_seconds
from .fenwick import DualFenwick
from .game_state import GameState, PlayerLine
from .possession_fsm import PossessionFSM, Transition
from .segment_tree import Run, RunTree

MAX_PERIODS = 8  # regulation plus four overtimes, sizes the fixed arrays


@dataclass(frozen=True, slots=True)
class Update:
    """What one ingested event did. Returned so callers can react."""

    event: Event
    transition: Transition
    score: dict[str, int]
    possession_ended: bool


class Engine:
    """Wires the reorder buffer, FSM, game state, and both trees together.

    One process() call per incoming event. Fan-out order is fixed:
    FSM first (it decides possession), then game state, then the trees.
    """

    def __init__(
        self,
        home: str = "HOME",
        away: str = "AWAY",
        lateness: int = 5,
        strict: bool = False,
        max_periods: int = MAX_PERIODS,
    ) -> None:
        self.home = home
        self.away = away
        self.size = total_game_seconds(max_periods) + 1  # +1 so clock 0:00 fits
        self.buffer = ReorderBuffer(lateness=lateness, strict=strict)
        self.fsm = PossessionFSM(home=home, away=away, strict=strict)
        self.state = GameState(home=home, away=away)
        self.points = DualFenwick(self.size, home=home, away=away)
        self.runs = RunTree(self.size, home=home, away=away)
        self.log: list[Event] = []
        self.listeners: list[Callable[[Update], None]] = []
        self._finished = False

    def on_update(self, fn: Callable[[Update], None]) -> None:
        """Register a callback. The agent uses this."""
        self.listeners.append(fn)

    # ---------- ingest ----------

    def process(self, event: Event) -> list[Update]:
        """Feed one event in. Returns updates for whatever the buffer released."""
        if self._finished:
            raise RuntimeError("engine already finished")
        return [self._apply(e) for e in self.buffer.push(event)]

    def process_all(self, events: Iterable[Event]) -> list[Update]:
        out: list[Update] = []
        for e in events:
            out.extend(self.process(e))
        return out

    def finish(self) -> list[Update]:
        """End of feed. Drain the buffer and close the last possession."""
        if self._finished:
            return []
        updates = [self._apply(e) for e in self.buffer.flush()]
        self.fsm.finish(self.state.elapsed)
        self._finished = True
        return updates

    def _apply(self, event: Event) -> Update:
        index = min(event.elapsed, self.size - 1)
        transition = self.fsm.apply(event)  # decides possession first
        self.state.apply(event)
        if event.points and event.team:
            self.points.add(event.team, index, event.points)
            self.runs.add(event.team, index, event.points)
        self.log.append(event)

        update = Update(
            event=event,
            transition=transition,
            score=self.state.score,
            possession_ended=transition.changed,
        )
        for fn in self.listeners:
            fn(update)
        return update

    # ---------- queries ----------

    def window_score(self, start: int, end: int) -> dict[str, int]:
        """Points by each team in [start, end) seconds. O(log n)."""
        start, end = self._clamp(start, end)
        return self.points.window(start, end)

    def window_minutes(self, start_minute: float, end_minute: float) -> dict[str, int]:
        return self.window_score(int(start_minute * 60), int(end_minute * 60))

    def best_run(
        self, team: str | None = None, start: int = 0, end: int | None = None
    ) -> Run:
        """Biggest scoring run in the window, and when it happened. O(log n)."""
        start, end = self._clamp(start, end)
        if team is None:
            return self.runs.best_run_overall(start, end)
        return self.runs.best_run(team, start, end)

    def best_run_minutes(
        self, start_minute: float, end_minute: float, team: str | None = None
    ) -> Run:
        return self.best_run(team, int(start_minute * 60), int(end_minute * 60))

    def margin(self, start: int = 0, end: int | None = None) -> int:
        start, end = self._clamp(start, end)
        return self.points.margin(start, end)

    def player_stats(self, name: str) -> PlayerLine | None:
        return self.state.players.get(name)

    def top_scorers(self, team: str | None = None, limit: int = 5) -> list[PlayerLine]:
        return self.state.top_scorers(team=team, limit=limit)

    def _clamp(self, start: int, end: int | None) -> tuple[int, int]:
        if end is None:
            end = self.size
        return max(0, start), min(self.size, end)

    # ---------- status ----------

    @property
    def score(self) -> dict[str, int]:
        return self.state.score

    @property
    def elapsed(self) -> int:
        return self.state.elapsed

    @property
    def clock(self) -> str:
        """Time remaining in the current period, as M:SS."""
        period = max(self.state.period, 1)
        into = self.state.elapsed - period_start_elapsed(period)
        remaining = max(0, period_length(period) - into)
        return f"{remaining // 60}:{remaining % 60:02d}"

    @property
    def possession(self) -> str | None:
        return self.fsm.state

    def opponent(self, team: str) -> str:
        return self.away if team == self.home else self.home

    def box_score(self) -> dict:
        box = self.state.box_score()
        run = self.best_run()
        box["possessions"] = self.fsm.count
        box["violations"] = len(self.fsm.violations)
        box["dropped_events"] = len(self.buffer.dropped)
        box["best_run"] = (
            {"team": run.team, "points": run.points, "start": run.start, "end": run.end}
            if run.points
            else None
        )
        return box