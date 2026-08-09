from __future__ import annotations

from dataclasses import dataclass

from .events import Event, EventType


@dataclass(slots=True)
class PlayerLine:
    """One player's box score row."""

    name: str
    team: str
    points: int = 0
    fg_made: int = 0
    fg_att: int = 0
    fg3_made: int = 0
    fg3_att: int = 0
    ft_made: int = 0
    ft_att: int = 0
    off_reb: int = 0
    def_reb: int = 0
    assists: int = 0
    steals: int = 0
    blocks: int = 0
    turnovers: int = 0
    fouls: int = 0

    @property
    def rebounds(self) -> int:
        return self.off_reb + self.def_reb

    @property
    def fg_pct(self) -> float:
        return self.fg_made / self.fg_att if self.fg_att else 0.0

    @property
    def fg3_pct(self) -> float:
        return self.fg3_made / self.fg3_att if self.fg3_att else 0.0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "team": self.team,
            "points": self.points,
            "fg": f"{self.fg_made}/{self.fg_att}",
            "fg3": f"{self.fg3_made}/{self.fg3_att}",
            "ft": f"{self.ft_made}/{self.ft_att}",
            "reb": self.rebounds,
            "ast": self.assists,
            "stl": self.steals,
            "blk": self.blocks,
            "tov": self.turnovers,
            "pf": self.fouls,
        }


@dataclass(slots=True)
class TeamLine:
    """Team totals, plus the things only a team has."""

    name: str
    points: int = 0
    fg_made: int = 0
    fg_att: int = 0
    fg3_made: int = 0
    fg3_att: int = 0
    ft_made: int = 0
    ft_att: int = 0
    off_reb: int = 0
    def_reb: int = 0
    assists: int = 0
    steals: int = 0
    blocks: int = 0
    turnovers: int = 0
    fouls: int = 0
    timeouts: int = 0
    largest_lead: int = 0
    lead_changes: int = 0
    ties: int = 0

    @property
    def rebounds(self) -> int:
        return self.off_reb + self.def_reb

    @property
    def fg_pct(self) -> float:
        return self.fg_made / self.fg_att if self.fg_att else 0.0

    def as_dict(self) -> dict:
        return {
            "team": self.name,
            "points": self.points,
            "fg": f"{self.fg_made}/{self.fg_att}",
            "fg3": f"{self.fg3_made}/{self.fg3_att}",
            "ft": f"{self.ft_made}/{self.ft_att}",
            "reb": self.rebounds,
            "ast": self.assists,
            "stl": self.steals,
            "blk": self.blocks,
            "tov": self.turnovers,
            "pf": self.fouls,
            "largest_lead": self.largest_lead,
        }


@dataclass(slots=True)
class Streak:
    """Unanswered points by one team, reset when the other scores."""

    team: str | None = None
    points: int = 0
    start: int = 0
    end: int = 0

    def as_dict(self) -> dict:
        return {
            "team": self.team,
            "points": self.points,
            "start": self.start,
            "end": self.end,
        }


class GameState:
    """Incremental box scores, lead tracking, streaks. O(1) per event."""

    def __init__(self, home: str = "HOME", away: str = "AWAY") -> None:
        self.home = home
        self.away = away
        self.teams = {home: TeamLine(home), away: TeamLine(away)}
        self.players: dict[str, PlayerLine] = {}
        self.period = 0
        self.elapsed = 0
        self.events_seen = 0
        self.streak = Streak()
        self.best_streak = Streak()
        self._last_margin = 0
        self._last_leader: str | None = None

    # ---------- lookups ----------

    def opponent(self, team: str) -> str:
        return self.away if team == self.home else self.home

    def player(self, name: str, team: str) -> PlayerLine:
        line = self.players.get(name)
        if line is None:
            line = PlayerLine(name=name, team=team)
            self.players[name] = line
        return line

    @property
    def score(self) -> dict[str, int]:
        return {
            self.home: self.teams[self.home].points,
            self.away: self.teams[self.away].points,
        }

    @property
    def margin(self) -> int:
        """Home minus away."""
        return self.teams[self.home].points - self.teams[self.away].points

    @property
    def leader(self) -> str | None:
        m = self.margin
        if m > 0:
            return self.home
        if m < 0:
            return self.away
        return None

    def top_scorers(self, team: str | None = None, limit: int = 5) -> list[PlayerLine]:
        lines = [p for p in self.players.values() if team is None or p.team == team]
        # name breaks ties so replay output is stable
        lines.sort(key=lambda p: (-p.points, p.name))
        return lines[:limit]

    # ---------- ingest ----------

    def apply(self, event: Event) -> None:
        self.events_seen += 1
        self.elapsed = event.elapsed
        self.period = max(self.period, event.period)

        if event.type is EventType.PERIOD_START:
            return
        if event.type is EventType.PERIOD_END:
            return
        if event.team is None:
            return

        team = self.teams[event.team]
        line = self.player(event.player, event.team) if event.player else None
        self._tally(event, team, line)

        if event.points:
            self._score(event, team, line)

    def _tally(self, event: Event, team: TeamLine, line: PlayerLine | None) -> None:
        t = event.type
        if t is EventType.MADE_2:
            self._bump(team, line, "fg_made", "fg_att")
        elif t is EventType.MISS_2:
            self._bump(team, line, "fg_att")
        elif t is EventType.MADE_3:
            self._bump(team, line, "fg_made", "fg_att", "fg3_made", "fg3_att")
        elif t is EventType.MISS_3:
            self._bump(team, line, "fg_att", "fg3_att")
        elif t is EventType.MADE_FT:
            self._bump(team, line, "ft_made", "ft_att")
        elif t is EventType.MISS_FT:
            self._bump(team, line, "ft_att")
        elif t is EventType.OFF_REBOUND:
            self._bump(team, line, "off_reb")
        elif t is EventType.DEF_REBOUND:
            self._bump(team, line, "def_reb")
        elif t is EventType.ASSIST:
            self._bump(team, line, "assists")
        elif t is EventType.STEAL:
            self._bump(team, line, "steals")
        elif t is EventType.BLOCK:
            self._bump(team, line, "blocks")
        elif t is EventType.TURNOVER:
            self._bump(team, line, "turnovers")
        elif t is EventType.FOUL:
            self._bump(team, line, "fouls")
        elif t is EventType.TIMEOUT:
            team.timeouts += 1

    def _bump(self, team: TeamLine, line: PlayerLine | None, *fields: str) -> None:
        for f in fields:
            setattr(team, f, getattr(team, f) + 1)
            if line is not None:
                setattr(line, f, getattr(line, f) + 1)

    def _score(self, event: Event, team: TeamLine, line: PlayerLine | None) -> None:
        team.points += event.points
        if line is not None:
            line.points += event.points
        self._update_streak(event)
        self._update_lead()

    def _update_streak(self, event: Event) -> None:
        if self.streak.team != event.team:
            self.streak = Streak(team=event.team, points=0, start=event.elapsed)
        self.streak.points += event.points
        self.streak.end = event.elapsed
        if self.streak.points > self.best_streak.points:
            self.best_streak = Streak(
                team=self.streak.team,
                points=self.streak.points,
                start=self.streak.start,
                end=self.streak.end,
            )

    def _update_lead(self) -> None:
        margin = self.margin
        if margin > 0:
            self.teams[self.home].largest_lead = max(
                self.teams[self.home].largest_lead, margin
            )
        elif margin < 0:
            self.teams[self.away].largest_lead = max(
                self.teams[self.away].largest_lead, -margin
            )

        if margin == 0 and self._last_margin != 0:
            for t in self.teams.values():
                t.ties += 1

        # compare against whoever led last, so a tie in between still counts
        if margin != 0:
            leader = self.home if margin > 0 else self.away
            if self._last_leader is not None and leader != self._last_leader:
                for t in self.teams.values():
                    t.lead_changes += 1
            self._last_leader = leader

        self._last_margin = margin

    # ---------- output ----------

    def box_score(self) -> dict:
        return {
            "score": self.score,
            "period": self.period,
            "teams": [
                self.teams[self.home].as_dict(),
                self.teams[self.away].as_dict(),
            ],
            "players": [p.as_dict() for p in self.top_scorers(limit=len(self.players))],
            "lead_changes": self.teams[self.home].lead_changes,
            "ties": self.teams[self.home].ties,
            "best_streak": self.best_streak.as_dict(),
        }