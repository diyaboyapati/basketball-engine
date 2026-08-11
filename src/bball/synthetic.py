from __future__ import annotations

import random
from dataclasses import dataclass

from .events import (
    REGULATION_PERIODS,
    Event,
    EventType,
    period_length,
)

HOME_ROSTER = ["Ash", "Bell", "Cruz", "Diaz", "Ellis", "Foss", "Gray", "Hale"]
AWAY_ROSTER = ["Ito", "Jain", "Kerr", "Lowe", "Marsh", "Nunez", "Oke", "Pike"]

# how a possession ends, before shot outcome is rolled
OUTCOMES = [
    (EventType.MADE_2, 0.30),
    (EventType.MISS_2, 0.24),
    (EventType.MADE_3, 0.11),
    (EventType.MISS_3, 0.18),
    (EventType.TURNOVER, 0.11),
    (EventType.FOUL, 0.06),
]

ASSIST_RATE = 0.55
BLOCK_RATE = 0.08
STEAL_RATE = 0.45  # share of turnovers credited as a steal
OFF_REBOUND_RATE = 0.26
FT_MAKE_RATE = 0.77
MIN_POSSESSION = 4
MAX_POSSESSION = 22


@dataclass(slots=True)
class GameSpec:
    """Knobs for the generator. Defaults make a normal-looking game."""

    seed: int = 0
    home: str = "HOME"
    away: str = "AWAY"
    periods: int = REGULATION_PERIODS
    home_edge: float = 0.0  # shifts home scoring odds, roughly points per 100
    jitter: int = 0  # if set, shuffle output within this window
    lateness_seconds: int = 0  # if set, delay some events past their slot


class SyntheticGame:
    """Deterministic play-by-play generator. Same seed, same game, always."""

    def __init__(self, spec: GameSpec | None = None, **kwargs) -> None:
        self.spec = spec or GameSpec(**kwargs)
        self.rng = random.Random(self.spec.seed)
        self.rosters = {self.spec.home: HOME_ROSTER, self.spec.away: AWAY_ROSTER}
        self._seq = 0

    # ---------- helpers ----------

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq - 1

    def _event(self, etype: EventType, period: int, clock: int, team=None, player=None) -> Event:
        clock = max(0, min(clock, period_length(period)))
        return Event(
            type=etype,
            period=period,
            clock=clock,
            seq=self._next_seq(),
            team=team,
            player=player,
        )

    def _player(self, team: str) -> str:
        return self.rng.choice(self.rosters[team])

    def _outcome(self, team: str) -> EventType:
        """Weighted pick, nudged by home_edge."""
        weights = []
        for etype, w in OUTCOMES:
            edge = self.spec.home_edge / 100
            if team == self.spec.home and etype in (EventType.MADE_2, EventType.MADE_3):
                w *= 1 + edge
            elif team == self.spec.away and etype in (EventType.MADE_2, EventType.MADE_3):
                w *= 1 - edge
            weights.append(max(w, 0.001))
        return self.rng.choices([e for e, _ in OUTCOMES], weights=weights)[0]

    # ---------- generation ----------

    def events(self) -> list[Event]:
        """One full game, in true chronological order."""
        out: list[Event] = []
        offense = self.spec.home if self.rng.random() < 0.5 else self.spec.away

        for period in range(1, self.spec.periods + 1):
            length = period_length(period)
            out.append(self._event(EventType.PERIOD_START, period, length))
            if period == 1:
                out.append(self._event(EventType.JUMP_BALL, period, length, offense))

            clock = length
            while clock > MIN_POSSESSION:
                used = self.rng.randint(MIN_POSSESSION, MAX_POSSESSION)
                clock = max(0, clock - used)
                events, offense = self._possession(period, clock, offense)
                out.extend(events)

            out.append(self._event(EventType.PERIOD_END, period, 0))
            # next period, ball to whoever did not have it
            offense = self._other(offense)

        return self._maybe_disorder(out)

    def _other(self, team: str) -> str:
        return self.spec.away if team == self.spec.home else self.spec.home

    def _possession(self, period: int, clock: int, offense: str) -> tuple[list[Event], str]:
        """Emit one trip. Returns the events and who has the ball next."""
        defense = self._other(offense)
        shooter = self._player(offense)
        outcome = self._outcome(offense)
        out: list[Event] = []

        if outcome is EventType.TURNOVER:
            out.append(self._event(EventType.TURNOVER, period, clock, offense, shooter))
            if self.rng.random() < STEAL_RATE:
                out.append(
                    self._event(EventType.STEAL, period, clock, defense, self._player(defense))
                )
            return out, defense

        if outcome is EventType.FOUL:
            out.append(
                self._event(EventType.FOUL, period, clock, defense, self._player(defense))
            )
            return out + self._free_throws(period, clock, offense, shooter, 2), defense

        made = outcome in (EventType.MADE_2, EventType.MADE_3)
        if made:
            # assist is emitted first, it caused the basket
            if self.rng.random() < ASSIST_RATE:
                passer = self._player(offense)
                if passer != shooter:
                    out.append(self._event(EventType.ASSIST, period, clock, offense, passer))
            out.append(self._event(outcome, period, clock, offense, shooter))
            return out, defense

        out.append(self._event(outcome, period, clock, offense, shooter))
        if self.rng.random() < BLOCK_RATE:
            out.append(
                self._event(EventType.BLOCK, period, clock, defense, self._player(defense))
            )

        # somebody has to rebound a miss
        if self.rng.random() < OFF_REBOUND_RATE:
            out.append(
                self._event(
                    EventType.OFF_REBOUND, period, max(0, clock - 1), offense, self._player(offense)
                )
            )
            return out, offense
        out.append(
            self._event(
                EventType.DEF_REBOUND, period, max(0, clock - 1), defense, self._player(defense)
            )
        )
        return out, defense

    def _free_throws(self, period: int, clock: int, team: str, shooter: str, count: int) -> list[Event]:
        out: list[Event] = []
        for i in range(count):
            at = max(0, clock - i)
            made = self.rng.random() < FT_MAKE_RATE
            etype = EventType.MADE_FT if made else EventType.MISS_FT
            out.append(self._event(etype, period, at, team, shooter))
        return out

    # ---------- feed simulation ----------

    def _maybe_disorder(self, events: list[Event]) -> list[Event]:
        """Fake a real feed: roughly ordered, not exactly."""
        if self.spec.jitter <= 0:
            return events
        out = events[:]
        for i in range(0, len(out), self.spec.jitter):
            chunk = out[i : i + self.spec.jitter]
            self.rng.shuffle(chunk)
            out[i : i + self.spec.jitter] = chunk
        return out


def generate(seed: int = 0, **kwargs) -> list[Event]:
    """One game as a list of events."""
    return SyntheticGame(GameSpec(seed=seed, **kwargs)).events()