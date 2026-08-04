from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

REGULATION_PERIODS = 4
REGULATION_PERIOD_SECONDS = 12 * 60
OVERTIME_PERIOD_SECONDS = 5 * 60


class EventType(str, Enum):
    PERIOD_START = "period_start"
    PERIOD_END = "period_end"
    MADE_2 = "made_2"
    MADE_3 = "made_3"
    MADE_FT = "made_ft"
    MISS_2 = "miss_2"
    MISS_3 = "miss_3"
    MISS_FT = "miss_ft"
    OFF_REBOUND = "off_rebound"
    DEF_REBOUND = "def_rebound"
    TURNOVER = "turnover"
    STEAL = "steal"
    FOUL = "foul"
    ASSIST = "assist"
    BLOCK = "block"
    TIMEOUT = "timeout"
    JUMP_BALL = "jump_ball"


# points credited per event, absent means zero
POINTS = {
    EventType.MADE_2: 2,
    EventType.MADE_3: 3,
    EventType.MADE_FT: 1,
}

SHOT_ATTEMPTS = {
    EventType.MADE_2,
    EventType.MADE_3,
    EventType.MISS_2,
    EventType.MISS_3,
}


def period_length(period: int) -> int:
    """Seconds in a given period. Periods above 4 are overtime."""
    if period <= REGULATION_PERIODS:
        return REGULATION_PERIOD_SECONDS
    return OVERTIME_PERIOD_SECONDS


def period_start_elapsed(period: int) -> int:
    """Seconds elapsed in the game when a period tips off."""
    total = 0
    for p in range(1, period):
        total += period_length(p)
    return total


def total_game_seconds(periods: int = REGULATION_PERIODS) -> int:
    return period_start_elapsed(periods) + period_length(periods)


@dataclass(frozen=True, slots=True)
class Event:
    type: EventType
    period: int
    clock: int  # seconds REMAINING in the period, counts down
    seq: int  # feed-assigned sequence number, breaks clock ties
    team: str | None = None
    player: str | None = None
    elapsed: int = field(init=False)

    def __post_init__(self) -> None:
        if self.period < 1:
            raise ValueError(f"period must be >= 1, got {self.period}")
        length = period_length(self.period)
        if not 0 <= self.clock <= length:
            raise ValueError(f"clock {self.clock} outside period {self.period}")
        if self.seq < 0:
            raise ValueError(f"seq must be >= 0, got {self.seq}")
        # the whole point of this file: one axis, computed once
        absolute = period_start_elapsed(self.period) + (length - self.clock)
        object.__setattr__(self, "elapsed", absolute)

    @property
    def key(self) -> tuple[int, int]:
        """Total order for the reorder buffer. seq breaks clock ties."""
        return (self.elapsed, self.seq)

    @property
    def points(self) -> int:
        return POINTS.get(self.type, 0)

    @property
    def is_shot_attempt(self) -> bool:
        return self.type in SHOT_ATTEMPTS

    @property
    def is_made_shot(self) -> bool:
        return self.points > 0

    def __repr__(self) -> str:
        mins, secs = divmod(self.clock, 60)
        who = self.player or self.team or "-"
        return f"<Q{self.period} {mins}:{secs:02d} {self.type.value} {who} @{self.elapsed}s>"