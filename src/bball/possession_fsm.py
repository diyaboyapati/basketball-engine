from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .events import Event, EventType


class Effect(Enum):
    """What an event does to possession, relative to the team that did it."""

    KEEP = "keep"
    TO_ACTOR = "to_actor"
    TO_OPPONENT = "to_opponent"
    DEAD = "dead"


# the whole model, in one table
TRANSITIONS: dict[EventType, Effect] = {
    EventType.PERIOD_START: Effect.KEEP,
    EventType.PERIOD_END: Effect.DEAD,
    EventType.JUMP_BALL: Effect.TO_ACTOR,
    EventType.MADE_2: Effect.TO_OPPONENT,
    EventType.MADE_3: Effect.TO_OPPONENT,
    EventType.MADE_FT: Effect.TO_OPPONENT,
    EventType.MISS_2: Effect.KEEP,
    EventType.MISS_3: Effect.KEEP,
    EventType.MISS_FT: Effect.KEEP,
    EventType.OFF_REBOUND: Effect.TO_ACTOR,
    EventType.DEF_REBOUND: Effect.TO_ACTOR,
    EventType.TURNOVER: Effect.TO_OPPONENT,
    EventType.STEAL: Effect.TO_ACTOR,
    EventType.ASSIST: Effect.KEEP,
    EventType.BLOCK: Effect.KEEP,
    EventType.FOUL: Effect.KEEP,
    EventType.TIMEOUT: Effect.KEEP,
}

# only the team holding the ball can do these
REQUIRES_BALL = {
    EventType.MADE_2,
    EventType.MADE_3,
    EventType.MISS_2,
    EventType.MISS_3,
    EventType.TURNOVER,
}


class IllegalTransitionError(Exception):
    """Raised in strict mode when an event contradicts the derived state."""


@dataclass(frozen=True, slots=True)
class Violation:
    event: Event
    state: str | None
    reason: str

    def __repr__(self) -> str:
        return f"<Violation {self.reason} state={self.state} {self.event!r}>"


@dataclass(frozen=True, slots=True)
class Transition:
    event: Event
    before: str | None
    after: str | None
    legal: bool

    @property
    def changed(self) -> bool:
        return self.before != self.after


@dataclass(slots=True)
class Possession:
    """One team's trip with the ball."""

    team: str
    start: int
    end: int | None = None

    @property
    def seconds(self) -> int:
        return 0 if self.end is None else self.end - self.start


class PossessionFSM:
    """Derives who has the ball from play-by-play.

    Play-by-play says what happened, not what state the game is in, so
    possession is an inference. An illegal transition means the inference
    broke: strict mode raises, lenient mode records it and resyncs.
    """

    def __init__(self, home: str = "HOME", away: str = "AWAY", strict: bool = False) -> None:
        self.home = home
        self.away = away
        self.strict = strict
        self.state: str | None = None
        self.violations: list[Violation] = []
        self.possessions: list[Possession] = []
        self._current: Possession | None = None

    def opponent(self, team: str) -> str:
        if team == self.home:
            return self.away
        if team == self.away:
            return self.home
        raise KeyError(f"unknown team {team!r}")

    def apply(self, event: Event) -> Transition:
        before = self.state
        legal = self._check(event)
        after = self._next_state(event, before)
        if after != before:
            self._close_possession(event.elapsed)
            self.state = after
            if after is not None:
                self._open_possession(after, event.elapsed)
        return Transition(event=event, before=before, after=after, legal=legal)

    def _check(self, event: Event) -> bool:
        """Does this event contradict what we believe the state to be?"""
        if event.type not in REQUIRES_BALL:
            return True
        if event.team is None:
            return self._violation(event, "event needs a team")
        if self.state is None:
            return self._violation(event, "no live possession")
        if event.team != self.state:
            return self._violation(event, f"{event.team} acted without the ball")
        return True

    def _violation(self, event: Event, reason: str) -> bool:
        if self.strict:
            raise IllegalTransitionError(f"{reason}: {event!r} state={self.state}")
        self.violations.append(Violation(event=event, state=self.state, reason=reason))
        return False

    def _next_state(self, event: Event, before: str | None) -> str | None:
        effect = TRANSITIONS[event.type]
        if effect is Effect.DEAD:
            return None
        if effect is Effect.KEEP:
            return before
        if event.team is None:
            return before
        # actor-relative, so a wrong state gets corrected instead of compounded
        if effect is Effect.TO_ACTOR:
            return event.team
        return self.opponent(event.team)

    def _open_possession(self, team: str, at: int) -> None:
        self._current = Possession(team=team, start=at)
        self.possessions.append(self._current)

    def _close_possession(self, at: int) -> None:
        if self._current is not None:
            self._current.end = at
            self._current = None

    def finish(self, at: int) -> None:
        """End of game. Close whatever possession is still open."""
        self._close_possession(at)
        self.state = None

    @property
    def count(self) -> int:
        return len(self.possessions)

    def count_for(self, team: str) -> int:
        return sum(1 for p in self.possessions if p.team == team)

    def reset(self) -> None:
        self.state = None
        self.violations.clear()
        self.possessions.clear()
        self._current = None