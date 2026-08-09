from __future__ import annotations

import itertools

import pytest

from bball.events import Event, EventType
from bball.game_state import GameState
from bball.possession_fsm import (
    TRANSITIONS,
    Effect,
    IllegalTransitionError,
    PossessionFSM,
)

_seq = itertools.count()


def ev(elapsed: int, etype: EventType, team: str | None = None, player: str | None = None) -> Event:
    """Build an event at a given second of period 1."""
    return Event(
        type=etype,
        period=1,
        clock=720 - elapsed,
        seq=next(_seq),
        team=team,
        player=player,
    )


def tipoff(fsm: PossessionFSM, team: str = "HOME", at: int = 0) -> None:
    fsm.apply(ev(at, EventType.JUMP_BALL, team))


# ---------- FSM: the transition table ----------


def test_every_event_type_has_a_transition():
    for t in EventType:
        assert t in TRANSITIONS


def test_jump_ball_gives_ball_to_actor():
    fsm = PossessionFSM()
    assert fsm.state is None
    fsm.apply(ev(0, EventType.JUMP_BALL, "HOME"))
    assert fsm.state == "HOME"


def test_made_basket_gives_ball_to_opponent():
    fsm = PossessionFSM()
    tipoff(fsm, "HOME")
    fsm.apply(ev(10, EventType.MADE_2, "HOME"))
    assert fsm.state == "AWAY"
    fsm.apply(ev(20, EventType.MADE_3, "AWAY"))
    assert fsm.state == "HOME"


def test_miss_keeps_ball_live_until_the_rebound():
    fsm = PossessionFSM()
    tipoff(fsm, "HOME")
    fsm.apply(ev(10, EventType.MISS_2, "HOME"))
    assert fsm.state == "HOME"  # nobody has it yet, shot is in the air
    fsm.apply(ev(12, EventType.DEF_REBOUND, "AWAY"))
    assert fsm.state == "AWAY"


def test_offensive_rebound_keeps_ball_with_shooter_team():
    fsm = PossessionFSM()
    tipoff(fsm, "HOME")
    fsm.apply(ev(10, EventType.MISS_3, "HOME"))
    fsm.apply(ev(12, EventType.OFF_REBOUND, "HOME"))
    assert fsm.state == "HOME"


def test_turnover_and_steal_both_flip_to_the_defense():
    fsm = PossessionFSM()
    tipoff(fsm, "HOME")
    fsm.apply(ev(10, EventType.TURNOVER, "HOME"))
    assert fsm.state == "AWAY"

    fsm.apply(ev(20, EventType.TURNOVER, "AWAY"))
    assert fsm.state == "HOME"
    fsm.apply(ev(25, EventType.STEAL, "AWAY"))
    assert fsm.state == "AWAY"


def test_neutral_events_do_not_move_the_ball():
    fsm = PossessionFSM()
    tipoff(fsm, "HOME")
    for etype in (EventType.FOUL, EventType.TIMEOUT, EventType.ASSIST, EventType.BLOCK):
        fsm.apply(ev(30, etype, "AWAY"))
        assert fsm.state == "HOME"


def test_missed_free_throw_stays_live_and_made_one_does_not():
    fsm = PossessionFSM()
    tipoff(fsm, "HOME")
    fsm.apply(ev(10, EventType.MISS_FT, "HOME"))
    assert fsm.state == "HOME"
    fsm.apply(ev(11, EventType.MADE_FT, "HOME"))
    assert fsm.state == "AWAY"


def test_period_end_kills_possession():
    fsm = PossessionFSM()
    tipoff(fsm, "HOME")
    fsm.apply(ev(700, EventType.PERIOD_END))
    assert fsm.state is None


# ---------- FSM: self-correction and violations ----------


def test_actor_relative_rule_corrects_a_wrong_belief():
    fsm = PossessionFSM()
    tipoff(fsm, "HOME")
    assert fsm.state == "HOME"
    # a defensive rebound by AWAY is legal regardless of what we believed
    fsm.apply(ev(10, EventType.DEF_REBOUND, "AWAY"))
    assert fsm.state == "AWAY"
    assert fsm.violations == []


def test_shot_without_the_ball_is_a_violation():
    fsm = PossessionFSM()
    tipoff(fsm, "HOME")
    t = fsm.apply(ev(10, EventType.MISS_2, "AWAY"))
    assert t.legal is False
    assert len(fsm.violations) == 1
    assert "without the ball" in fsm.violations[0].reason


def test_shot_with_no_live_possession_is_a_violation():
    fsm = PossessionFSM()
    t = fsm.apply(ev(5, EventType.MADE_2, "HOME"))
    assert t.legal is False
    assert fsm.violations[0].reason == "no live possession"


def test_lenient_mode_keeps_going_after_a_violation():
    fsm = PossessionFSM()
    tipoff(fsm, "HOME")
    fsm.apply(ev(10, EventType.TURNOVER, "AWAY"))  # AWAY had no ball
    assert len(fsm.violations) == 1
    fsm.apply(ev(20, EventType.MADE_2, "HOME"))  # still processes normally
    assert fsm.state == "AWAY"


def test_strict_mode_raises_on_the_same_event():
    fsm = PossessionFSM(strict=True)
    tipoff(fsm, "HOME")
    with pytest.raises(IllegalTransitionError):
        fsm.apply(ev(10, EventType.TURNOVER, "AWAY"))


def test_neutral_events_never_trigger_violations():
    fsm = PossessionFSM(strict=True)
    tipoff(fsm, "HOME")
    # defense legitimately does all of these
    fsm.apply(ev(10, EventType.FOUL, "AWAY"))
    fsm.apply(ev(11, EventType.BLOCK, "AWAY"))
    fsm.apply(ev(12, EventType.STEAL, "AWAY"))
    assert fsm.violations == []


def test_unknown_team_rejected():
    fsm = PossessionFSM()
    with pytest.raises(KeyError):
        fsm.opponent("NOBODY")


# ---------- FSM: possession intervals ----------


def test_possessions_are_contiguous_and_ordered():
    fsm = PossessionFSM()
    tipoff(fsm, "HOME", at=0)
    fsm.apply(ev(14, EventType.MADE_2, "HOME"))
    fsm.apply(ev(31, EventType.TURNOVER, "AWAY"))
    fsm.apply(ev(48, EventType.MADE_3, "HOME"))
    fsm.finish(60)

    starts = [p.start for p in fsm.possessions]
    assert starts == sorted(starts)
    for a, b in zip(fsm.possessions, fsm.possessions[1:]):
        assert a.end == b.start
    assert fsm.possessions[-1].end == 60
    assert all(p.seconds >= 0 for p in fsm.possessions)


def test_possession_counts_per_team():
    fsm = PossessionFSM()
    tipoff(fsm, "HOME", at=0)
    fsm.apply(ev(10, EventType.MADE_2, "HOME"))  # -> AWAY
    fsm.apply(ev(20, EventType.MADE_2, "AWAY"))  # -> HOME
    fsm.apply(ev(30, EventType.TURNOVER, "HOME"))  # -> AWAY
    fsm.finish(40)

    assert fsm.count == 4
    assert fsm.count_for("HOME") == 2
    assert fsm.count_for("AWAY") == 2


def test_keep_events_do_not_open_new_possessions():
    fsm = PossessionFSM()
    tipoff(fsm, "HOME", at=0)
    before = fsm.count
    fsm.apply(ev(5, EventType.MISS_2, "HOME"))
    fsm.apply(ev(6, EventType.OFF_REBOUND, "HOME"))
    fsm.apply(ev(7, EventType.FOUL, "AWAY"))
    assert fsm.count == before


def test_reset_clears_everything():
    fsm = PossessionFSM()
    tipoff(fsm, "HOME")
    fsm.apply(ev(10, EventType.MISS_2, "AWAY"))
    fsm.reset()
    assert fsm.state is None
    assert fsm.count == 0
    assert fsm.violations == []


# ---------- GameState: tallies ----------


def test_three_pointer_counts_in_both_fg_and_fg3():
    gs = GameState()
    gs.apply(ev(10, EventType.MADE_3, "HOME", "Ash"))
    line = gs.players["Ash"]
    assert (line.points, line.fg_made, line.fg_att, line.fg3_made, line.fg3_att) == (3, 1, 1, 1, 1)
    assert gs.teams["HOME"].points == 3


def test_misses_count_attempts_only():
    gs = GameState()
    gs.apply(ev(10, EventType.MISS_2, "HOME", "Ash"))
    gs.apply(ev(12, EventType.MISS_3, "HOME", "Ash"))
    line = gs.players["Ash"]
    assert line.points == 0
    assert (line.fg_att, line.fg_made) == (2, 0)
    assert (line.fg3_att, line.fg3_made) == (1, 0)


def test_free_throws_stay_out_of_field_goals():
    gs = GameState()
    gs.apply(ev(10, EventType.MADE_FT, "HOME", "Ash"))
    gs.apply(ev(11, EventType.MISS_FT, "HOME", "Ash"))
    line = gs.players["Ash"]
    assert (line.ft_made, line.ft_att) == (1, 2)
    assert line.fg_att == 0
    assert line.points == 1


def test_rebounds_split_offensive_and_defensive():
    gs = GameState()
    gs.apply(ev(10, EventType.OFF_REBOUND, "HOME", "Ash"))
    gs.apply(ev(12, EventType.DEF_REBOUND, "HOME", "Ash"))
    gs.apply(ev(14, EventType.DEF_REBOUND, "HOME", "Ash"))
    line = gs.players["Ash"]
    assert (line.off_reb, line.def_reb, line.rebounds) == (1, 2, 3)


def test_players_created_lazily():
    gs = GameState()
    assert gs.players == {}
    gs.apply(ev(10, EventType.MADE_2, "HOME", "Ash"))
    assert list(gs.players) == ["Ash"]


def test_team_events_without_a_player_still_tally():
    gs = GameState()
    gs.apply(ev(10, EventType.TURNOVER, "HOME"))
    assert gs.teams["HOME"].turnovers == 1
    assert gs.players == {}


def test_team_totals_equal_sum_of_player_lines():
    gs = GameState()
    plays = [
        (10, EventType.MADE_3, "Ash"),
        (20, EventType.MADE_2, "Bell"),
        (30, EventType.MISS_2, "Ash"),
        (40, EventType.MADE_FT, "Bell"),
        (50, EventType.ASSIST, "Ash"),
        (60, EventType.TURNOVER, "Bell"),
    ]
    for at, etype, who in plays:
        gs.apply(ev(at, etype, "HOME", who))

    team = gs.teams["HOME"]
    lines = [p for p in gs.players.values() if p.team == "HOME"]
    assert team.points == sum(p.points for p in lines)
    assert team.fg_att == sum(p.fg_att for p in lines)
    assert team.assists == sum(p.assists for p in lines)
    assert team.turnovers == sum(p.turnovers for p in lines)


def test_percentages_handle_zero_attempts():
    gs = GameState()
    gs.apply(ev(10, EventType.ASSIST, "HOME", "Ash"))
    assert gs.players["Ash"].fg_pct == 0.0
    assert gs.teams["AWAY"].fg_pct == 0.0


# ---------- GameState: score, streaks, leads ----------


def test_score_margin_and_leader():
    gs = GameState()
    assert gs.leader is None
    gs.apply(ev(10, EventType.MADE_3, "HOME", "Ash"))
    assert gs.score == {"HOME": 3, "AWAY": 0}
    assert gs.margin == 3
    assert gs.leader == "HOME"

    gs.apply(ev(20, EventType.MADE_3, "AWAY", "Cruz"))
    assert gs.margin == 0
    assert gs.leader is None


def test_streak_accumulates_then_resets_on_opponent_score():
    gs = GameState()
    gs.apply(ev(10, EventType.MADE_2, "HOME", "Ash"))
    gs.apply(ev(20, EventType.MADE_3, "HOME", "Ash"))
    assert gs.streak.team == "HOME"
    assert gs.streak.points == 5
    assert (gs.streak.start, gs.streak.end) == (10, 20)

    gs.apply(ev(30, EventType.MADE_2, "AWAY", "Cruz"))
    assert gs.streak.team == "AWAY"
    assert gs.streak.points == 2


def test_best_streak_survives_the_reset():
    gs = GameState()
    for at in (10, 20, 30):
        gs.apply(ev(at, EventType.MADE_3, "HOME", "Ash"))
    gs.apply(ev(40, EventType.MADE_2, "AWAY", "Cruz"))
    assert gs.streak.points == 2
    assert gs.best_streak.team == "HOME"
    assert gs.best_streak.points == 9


def test_misses_do_not_break_a_streak():
    gs = GameState()
    gs.apply(ev(10, EventType.MADE_2, "HOME", "Ash"))
    gs.apply(ev(15, EventType.MISS_3, "AWAY", "Cruz"))
    gs.apply(ev(20, EventType.MADE_2, "HOME", "Ash"))
    assert gs.streak.team == "HOME"
    assert gs.streak.points == 4


def test_largest_lead_tracked_per_team():
    gs = GameState()
    gs.apply(ev(10, EventType.MADE_3, "HOME", "Ash"))
    gs.apply(ev(20, EventType.MADE_3, "HOME", "Ash"))
    gs.apply(ev(30, EventType.MADE_3, "AWAY", "Cruz"))
    gs.apply(ev(40, EventType.MADE_3, "AWAY", "Cruz"))
    gs.apply(ev(50, EventType.MADE_3, "AWAY", "Cruz"))
    assert gs.teams["HOME"].largest_lead == 6
    assert gs.teams["AWAY"].largest_lead == 3


def test_lead_change_counted_even_when_it_passes_through_a_tie():
    gs = GameState()
    gs.apply(ev(10, EventType.MADE_2, "HOME", "Ash"))  # +2
    gs.apply(ev(20, EventType.MADE_2, "AWAY", "Cruz"))  # tie
    gs.apply(ev(30, EventType.MADE_2, "AWAY", "Cruz"))  # -2, lead changed
    gs.apply(ev(40, EventType.MADE_3, "HOME", "Ash"))  # +1, changed back
    assert gs.teams["HOME"].lead_changes == 2
    assert gs.teams["HOME"].ties == 1


def test_extending_a_lead_is_not_a_lead_change():
    gs = GameState()
    for at in (10, 20, 30):
        gs.apply(ev(at, EventType.MADE_3, "HOME", "Ash"))
    assert gs.teams["HOME"].lead_changes == 0
    assert gs.teams["HOME"].ties == 0


# ---------- GameState: output ----------


def test_top_scorers_sorted_with_name_tiebreak():
    gs = GameState()
    gs.apply(ev(10, EventType.MADE_2, "HOME", "Bell"))
    gs.apply(ev(20, EventType.MADE_2, "HOME", "Bell"))
    gs.apply(ev(30, EventType.MADE_2, "HOME", "Ash"))
    gs.apply(ev(40, EventType.MADE_2, "HOME", "Ash"))
    gs.apply(ev(50, EventType.MADE_2, "AWAY", "Cruz"))

    names = [p.name for p in gs.top_scorers()]
    assert names[:2] == ["Ash", "Bell"]  # tied at 4, name breaks it
    assert [p.name for p in gs.top_scorers(team="AWAY")] == ["Cruz"]
    assert len(gs.top_scorers(limit=1)) == 1


def test_box_score_shape():
    gs = GameState()
    gs.apply(ev(10, EventType.MADE_3, "HOME", "Ash"))
    gs.apply(ev(20, EventType.MADE_2, "AWAY", "Cruz"))
    box = gs.box_score()

    assert box["score"] == {"HOME": 3, "AWAY": 2}
    assert box["period"] == 1
    assert len(box["teams"]) == 2
    assert {t["team"] for t in box["teams"]} == {"HOME", "AWAY"}
    assert len(box["players"]) == 2
    assert box["best_streak"]["team"] == "HOME"


def test_period_markers_advance_the_clock_only():
    gs = GameState()
    gs.apply(ev(0, EventType.PERIOD_START))
    gs.apply(ev(719, EventType.PERIOD_END))
    assert gs.score == {"HOME": 0, "AWAY": 0}
    assert gs.events_seen == 2
    assert gs.elapsed == 719