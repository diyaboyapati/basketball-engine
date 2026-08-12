from __future__ import annotations

from dataclasses import dataclass

from .engine import Engine, Update
from .events import period_length, period_start_elapsed

RUN_THRESHOLD = 8  # unanswered points that make a possession noteworthy
SWING_THRESHOLD = 6  # win probability points
MIN_GAP_SECONDS = 45  # do not comment twice in a row on the same sequence


def clock_label(elapsed: int) -> str:
    period = 1
    while elapsed >= period_start_elapsed(period) + period_length(period) and period < 8:
        period += 1
    into = elapsed - period_start_elapsed(period)
    left = max(0, period_length(period) - into)
    return f"Q{period} {left // 60}:{left % 60:02d}"


def win_probability(margin: int, elapsed: int, total: int) -> float:
    """Crude logistic on margin and time left. Enough to detect swings."""
    remaining = max(total - elapsed, 1)
    # a lead is worth more as the clock runs out
    weight = margin / (remaining**0.5)
    return 1 / (1 + 2.718281828 ** (-0.9 * weight))


# ---------- trigger ----------


def check_noteworthy(engine: Engine, update: Update) -> str | None:
    """Plain-English reason this possession matters, or None."""
    if not update.event.points:
        return None

    streak = engine.state.streak
    if streak.team and streak.points >= RUN_THRESHOLD:
        return f"{streak.team} is on a {streak.points}-0 run"

    total = engine.size
    before = win_probability(
        engine.state.margin - _signed(engine, update), update.event.elapsed, total
    )
    after = win_probability(engine.state.margin, update.event.elapsed, total)
    swing = abs(after - before) * 100
    if swing >= SWING_THRESHOLD:
        return f"win probability moved {swing:.0f} points"
    return None


def _signed(engine: Engine, update: Update) -> int:
    """This event's points, signed from the home team's view."""
    pts = update.event.points
    return pts if update.event.team == engine.home else -pts


# ---------- tools ----------
# the only way commentary reaches engine numbers, so every line is checkable


def get_score(engine: Engine) -> dict:
    return {
        "score": engine.score,
        "clock": clock_label(engine.elapsed),
        "margin": engine.state.margin,
        "leader": engine.state.leader,
    }


def get_best_run(
    engine: Engine,
    team: str | None = None,
    start_minute: float = 0.0,
    end_minute: float | None = None,
) -> dict:
    end = engine.size if end_minute is None else int(end_minute * 60)
    run = engine.best_run(team, int(start_minute * 60), end)
    if not run.points:
        return {"team": run.team, "points": 0}
    return {
        "team": run.team,
        "points": run.points,
        "from": clock_label(run.start),
        "to": clock_label(run.end - 1),
        "seconds": run.seconds,
    }


def get_player_stats(engine: Engine, name: str) -> dict:
    line = engine.player_stats(name)
    if line is None:
        return {"error": f"no stats for {name}"}
    return line.as_dict()


DISPATCH = {
    "get_score": get_score,
    "get_best_run": get_best_run,
    "get_player_stats": get_player_stats,
}


def run_tool(engine: Engine, name: str, args: dict) -> dict:
    fn = DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool {name}"}
    try:
        return fn(engine, **args)
    except TypeError as exc:
        return {"error": str(exc)}


# ---------- commentator ----------


@dataclass(slots=True)
class Line:
    elapsed: int
    reason: str
    text: str


class Commentator:
    """Watches the engine, writes one sentence when something matters."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.lines: list[Line] = []
        self._last_at = -MIN_GAP_SECONDS

    def observe(self, update: Update) -> None:
        """Engine callback. Registered via engine.on_update."""
        if update.event.elapsed - self._last_at < MIN_GAP_SECONDS:
            return
        reason = check_noteworthy(self.engine, update)
        if reason is None:
            return
        self._last_at = update.event.elapsed
        self.lines.append(
            Line(elapsed=update.event.elapsed, reason=reason, text=self._write(reason))
        )

    def _write(self, reason: str) -> str:
        score = run_tool(self.engine, "get_score", {})
        home, away = self.engine.home, self.engine.away
        text = (
            f"{reason} — {home} {score['score'][home]}, {away} {score['score'][away]} "
            f"with {score['clock']} on the clock"
        )
        run = run_tool(self.engine, "get_best_run", {})
        if run["points"]:
            text += f", after a {run['points']}-point {run['team']} run from {run['from']}"
        return text + "."

    def transcript(self) -> list[str]:
        return [f"{clock_label(l.elapsed)}  {l.text}" for l in self.lines]