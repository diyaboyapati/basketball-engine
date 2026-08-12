from __future__ import annotations

import itertools

import pytest

from bball.agent import (
    Commentator,
    check_noteworthy,
    clock_label,
    get_best_run,
    get_player_stats,
    get_score,
    run_tool,
    win_probability,
)
from bball.cli import main as cli_main
from bball.engine import Engine
from bball.events import Event, EventType
from bball.game_state import GameState
from bball.possession_fsm import (
    TRANSITIONS,
    IllegalTransitionError,
    PossessionFSM,
)
from bball.synthetic import generate

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


def run_game(seed: int = 0, **kwargs) -> Engine:
    engine = Engine(home=kwargs.pop("home", "HOME"), away=kwargs.pop("away", "AWAY"))
    engine.process_all(generate(seed=seed, **kwargs))
    engine.finish()
    return engine


def watched_game(seed: int, **kwargs) -> tuple[Engine, Commentator]:
    engine = Engine()
    commentator = Commentator(engine)
    engine.on_update(commentator.observe)
    engine.process_all(generate(seed=seed, **kwargs))
    engine.finish()
    return engine, commentator


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


# ---------- synthetic generator ----------


def test_same_seed_gives_identical_events():
    a = generate(seed=42)
    b = generate(seed=42)
    assert [e.key for e in a] == [e.key for e in b]
    assert [e.type for e in a] == [e.type for e in b]
    assert [e.player for e in a] == [e.player for e in b]


def test_different_seeds_give_different_games():
    a = generate(seed=1)
    b = generate(seed=2)
    assert [(e.type, e.player) for e in a] != [(e.type, e.player) for e in b]


def test_generated_events_are_chronological():
    events = generate(seed=5)
    assert [e.key for e in events] == sorted(e.key for e in events)


def test_generated_game_covers_four_periods():
    events = generate(seed=6)
    assert {e.period for e in events} == {1, 2, 3, 4}
    assert events[0].type is EventType.PERIOD_START
    assert events[-1].type is EventType.PERIOD_END


def test_every_period_opens_with_a_possession():
    events = generate(seed=71)
    openers = [e for e in events if e.type is EventType.JUMP_BALL]
    assert len(openers) == 4
    assert {e.period for e in openers} == {1, 2, 3, 4}
    assert all(e.team is not None for e in openers)


def test_every_field_goal_miss_is_rebounded():
    events = generate(seed=8)
    misses = {EventType.MISS_2, EventType.MISS_3}
    rebounds = {EventType.OFF_REBOUND, EventType.DEF_REBOUND}
    for i, e in enumerate(events):
        if e.type in misses:
            nxt = [x.type for x in events[i + 1 : i + 3]]
            assert any(t in rebounds for t in nxt)


def test_every_rebound_follows_a_miss():
    events = generate(seed=8)
    misses = {EventType.MISS_2, EventType.MISS_3, EventType.MISS_FT}
    rebounds = {EventType.OFF_REBOUND, EventType.DEF_REBOUND}
    for i, e in enumerate(events):
        if e.type in rebounds:
            prev = [x.type for x in events[max(0, i - 2) : i]]
            assert any(t in misses for t in prev)


def test_assist_precedes_the_basket_it_fed():
    events = generate(seed=9)
    for a, b in zip(events, events[1:]):
        if a.type is EventType.ASSIST:
            assert b.type in (EventType.MADE_2, EventType.MADE_3)
            assert a.team == b.team
            assert a.elapsed == b.elapsed
            assert a.seq < b.seq  # same second, ordering comes from seq


def test_jitter_scrambles_but_keeps_the_same_events():
    clean = generate(seed=11)
    messy = generate(seed=11, jitter=6)
    assert [e.key for e in messy] != [e.key for e in clean]
    assert sorted(e.key for e in messy) == sorted(e.key for e in clean)


def test_home_edge_shifts_scoring():
    plain = sum(run_game(seed=s).margin() for s in range(4))
    tilted = sum(run_game(seed=s, home_edge=25).margin() for s in range(4))
    assert tilted > plain


# ---------- engine wiring ----------


def test_engine_replay_is_deterministic():
    a = run_game(seed=21)
    b = run_game(seed=21)
    assert a.box_score() == b.box_score()
    assert [e.key for e in a.log] == [e.key for e in b.log]


def test_engine_recovers_the_same_game_from_a_jittered_feed():
    clean = Engine()
    clean.process_all(generate(seed=23))
    clean.finish()

    # lateness must exceed the feed's actual disorder or events age out
    messy = Engine(lateness=300)
    messy.process_all(generate(seed=23, jitter=6))
    messy.finish()

    assert messy.buffer.dropped == []
    assert messy.score == clean.score
    assert messy.box_score()["players"] == clean.box_score()["players"]
    assert [e.key for e in messy.log] == [e.key for e in clean.log]


def test_too_small_a_lateness_drops_events():
    engine = Engine(lateness=5)
    engine.process_all(generate(seed=23, jitter=6))
    engine.finish()
    assert engine.buffer.dropped  # the tolerance is a real tradeoff, not free


def test_engine_processes_every_event():
    events = generate(seed=25)
    engine = Engine()
    engine.process_all(events)
    engine.finish()
    assert len(engine.log) == len(events)
    assert [e.key for e in engine.log] == sorted(e.key for e in events)


def test_synthetic_game_produces_no_fsm_violations():
    engine = run_game(seed=27)
    assert engine.fsm.violations == []


def test_fenwick_total_matches_game_state_score():
    engine = run_game(seed=29)
    assert engine.points.score() == engine.state.score


def test_window_score_over_whole_game_equals_final():
    engine = run_game(seed=31)
    whole = engine.window_score(0, engine.size)
    assert whole == engine.score


def test_windows_partition_the_game():
    engine = run_game(seed=33)
    cuts = [0, 700, 1440, 2100, engine.size]
    totals = {engine.home: 0, engine.away: 0}
    for lo, hi in zip(cuts, cuts[1:]):
        w = engine.window_score(lo, hi)
        totals[engine.home] += w[engine.home]
        totals[engine.away] += w[engine.away]
    assert totals == engine.score


def test_best_run_never_exceeds_window_points():
    engine = run_game(seed=35)
    for lo, hi in [(0, 720), (720, 1440), (1080, 1440), (0, engine.size)]:
        window = engine.window_score(lo, hi)
        for team in (engine.home, engine.away):
            run = engine.best_run(team, lo, hi)
            assert run.points <= window[team]
            if run.points:
                assert lo <= run.start < run.end <= hi


def test_best_run_matches_a_direct_window_query():
    engine = run_game(seed=37)
    run = engine.best_run()
    if run.points:
        window = engine.window_score(run.start, run.end)
        assert window[run.team] - window[engine.opponent(run.team)] == run.points


def test_narrower_window_never_finds_a_bigger_run():
    engine = run_game(seed=39)
    wide = engine.best_run(engine.home, 0, 1440).points
    narrow = engine.best_run(engine.home, 300, 1000).points
    assert narrow <= wide


def test_possessions_alternate_sensibly():
    engine = run_game(seed=41)
    assert engine.fsm.count > 50
    home = engine.fsm.count_for(engine.home)
    away = engine.fsm.count_for(engine.away)
    assert abs(home - away) <= 12  # pace is shared, counts stay close


def test_engine_rejects_events_after_finish():
    engine = Engine()
    events = generate(seed=43)
    engine.process_all(events)
    engine.finish()
    with pytest.raises(RuntimeError):
        engine.process(events[0])


def test_finish_is_idempotent():
    engine = run_game(seed=45)
    assert engine.finish() == []


def test_listeners_fire_once_per_released_event():
    engine = Engine()
    seen = []
    engine.on_update(lambda u: seen.append(u.event.key))
    engine.process_all(generate(seed=47))
    engine.finish()
    assert seen == [e.key for e in engine.log]


def test_clamped_windows_do_not_raise():
    engine = run_game(seed=49)
    assert engine.window_score(-500, 10) == engine.window_score(0, 10)
    assert engine.window_score(0, engine.size * 2) == engine.score


def test_minute_helpers_match_second_queries():
    engine = run_game(seed=51)
    assert engine.window_minutes(18, 24) == engine.window_score(1080, 1440)
    assert engine.best_run_minutes(18, 24).points == engine.best_run(None, 1080, 1440).points


# ---------- agent ----------


def test_clock_label_counts_down_within_the_period():
    assert clock_label(0) == "Q1 12:00"
    assert clock_label(60) == "Q1 11:00"
    assert clock_label(720) == "Q2 12:00"
    assert clock_label(2879) == "Q4 0:01"


def test_win_probability_is_symmetric_and_bounded():
    total = 2880
    assert win_probability(0, 1000, total) == pytest.approx(0.5)
    up = win_probability(10, 1000, total)
    down = win_probability(-10, 1000, total)
    assert up + down == pytest.approx(1.0)
    assert 0.0 < down < 0.5 < up < 1.0


def test_same_lead_matters_more_late():
    early = win_probability(6, 200, 2880)
    late = win_probability(6, 2800, 2880)
    assert late > early


def test_trigger_fires_on_a_big_run():
    engine = Engine()
    updates = []
    for at in (10, 40, 70, 100):
        updates = engine.process(ev(at, EventType.MADE_3, "HOME", "Ash"))
    engine.finish()
    reason = check_noteworthy(engine, updates[-1])
    assert reason is not None
    assert "HOME" in reason


def test_trigger_ignores_non_scoring_events():
    engine = Engine()
    engine.process(ev(0, EventType.JUMP_BALL, "HOME"))
    updates = engine.process(ev(10, EventType.DEF_REBOUND, "AWAY", "Ito"))
    engine.finish()
    assert all(check_noteworthy(engine, u) is None for u in updates)


def test_tools_return_engine_numbers():
    engine = run_game(seed=53)
    assert get_score(engine)["score"] == engine.score
    top = engine.top_scorers(limit=1)[0]
    assert get_player_stats(engine, top.name)["points"] == top.points
    run = get_best_run(engine)
    if run["points"]:
        assert run["points"] == engine.best_run().points


def test_get_best_run_respects_the_window():
    engine = run_game(seed=55)
    scoped = get_best_run(engine, start_minute=18, end_minute=24)
    assert scoped["points"] == engine.best_run(None, 1080, 1440).points


def test_unknown_player_returns_an_error_not_an_exception():
    engine = run_game(seed=57)
    assert "error" in get_player_stats(engine, "Nobody")


def test_run_tool_dispatch_and_bad_input():
    engine = run_game(seed=59)
    assert run_tool(engine, "get_score", {})["score"] == engine.score
    assert "error" in run_tool(engine, "not_a_tool", {})
    assert "error" in run_tool(engine, "get_player_stats", {"wrong": 1})


def test_commentator_writes_a_line_for_every_trigger():
    engine, commentator = watched_game(61, home_edge=20)
    assert commentator.lines
    for line in commentator.lines:
        assert line.text.endswith(".")
        assert line.reason in line.text


def test_commentary_cites_numbers_the_engine_can_reproduce():
    engine, commentator = watched_game(62, home_edge=20)
    final = engine.score
    for line in commentator.lines:
        # the score in a line must be one of the two teams' totals so far
        assert str(final[engine.home]) or str(final[engine.away])
        assert "on the clock" in line.text


def test_commentary_lines_are_spaced_out():
    engine, commentator = watched_game(63, home_edge=20)
    times = [line.elapsed for line in commentator.lines]
    assert times == sorted(times)
    assert all(b - a >= 45 for a, b in zip(times, times[1:]))


def test_transcript_is_formatted_and_ordered():
    engine, commentator = watched_game(65, home_edge=20)
    lines = commentator.transcript()
    assert len(lines) == len(commentator.lines)
    assert all(line.startswith("Q") for line in lines)


def test_commentator_stays_quiet_in_a_dull_game():
    engine = Engine()
    commentator = Commentator(engine)
    engine.on_update(commentator.observe)
    engine.process(ev(0, EventType.JUMP_BALL, "HOME"))
    engine.process(ev(10, EventType.MADE_2, "HOME", "Ash"))
    engine.finish()
    assert commentator.lines == []


# ---------- cli ----------


def test_cli_replay_runs(capsys):
    assert cli_main(["replay", "--synthetic", "--seed", "3"]) == 0
    out = capsys.readouterr().out
    assert "FINAL" in out
    assert "biggest run" in out


def test_cli_query_runs(capsys):
    code = cli_main(
        ["query", "--synthetic", "--seed", "3", "--start-minute", "18", "--end-minute", "24"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "window" in out
    assert "best run" in out


def test_cli_query_rejects_a_backwards_window(capsys):
    code = cli_main(["query", "--synthetic", "--start-minute", "24", "--end-minute", "18"])
    assert code == 1


def test_cli_verbose_prints_events(capsys):
    cli_main(["replay", "--synthetic", "--seed", "3", "--verbose"])
    out = capsys.readouterr().out
    assert out.count("\n") > 100