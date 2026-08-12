from __future__ import annotations

import argparse
import sys

from .agent import Commentator
from .engine import Engine
from .events import Event, period_length, period_start_elapsed
from .synthetic import generate


def fmt_clock(event: Event) -> str:
    m, s = divmod(event.clock, 60)
    return f"Q{event.period} {m}:{s:02d}"


def fmt_second(elapsed: int) -> str:
    """Absolute second to period clock, for run windows."""
    period = 1
    while elapsed >= period_start_elapsed(period) + period_length(period) and period < 8:
        period += 1
    into = elapsed - period_start_elapsed(period)
    remaining = max(0, period_length(period) - into)
    return f"Q{period} {remaining // 60}:{remaining % 60:02d}"


def build_engine(args) -> Engine:
    engine = Engine(home=args.home, away=args.away, lateness=args.lateness)
    events = generate(
        seed=args.seed,
        home=args.home,
        away=args.away,
        jitter=args.jitter,
        home_edge=args.home_edge,
    )
    engine.process_all(events)
    engine.finish()
    return engine


def print_box(engine: Engine) -> None:
    box = engine.box_score()
    home, away = engine.home, engine.away
    print()
    print(f"  FINAL   {home} {box['score'][home]}  -  {away} {box['score'][away]}")
    print(f"  {box['period']} periods, {box['possessions']} possessions")
    print()

    head = (
        f"  {'':10} {'PTS':>4} {'FG':>7} {'3PT':>7} "
        f"{'FT':>6} {'REB':>4} {'AST':>4} {'TOV':>4}"
    )
    for team in (home, away):
        print(f"  {team}")
        print(head)
        for p in engine.top_scorers(team=team, limit=5):
            d = p.as_dict()
            print(
                f"  {d['name']:10} {d['points']:>4} {d['fg']:>7} {d['fg3']:>7} "
                f"{d['ft']:>6} {d['reb']:>4} {d['ast']:>4} {d['tov']:>4}"
            )
        print()

    print(f"  lead changes {box['lead_changes']}   ties {box['ties']}")
    run = box["best_run"]
    if run:
        print(
            f"  biggest run  {run['team']} +{run['points']} "
            f"({fmt_second(run['start'])} to {fmt_second(run['end'] - 1)})"
        )
    if box["violations"]:
        print(f"  fsm violations {box['violations']}")
    if box["dropped_events"]:
        print(f"  dropped events {box['dropped_events']}")
    print()


def cmd_replay(args) -> int:
    engine = Engine(home=args.home, away=args.away, lateness=args.lateness)

    commentator = None
    if not args.no_agent:
        commentator = Commentator(engine)
        engine.on_update(commentator.observe)

    events = generate(
        seed=args.seed,
        home=args.home,
        away=args.away,
        jitter=args.jitter,
        home_edge=args.home_edge,
    )

    for event in events:
        for update in engine.process(event):
            if args.verbose:
                e = update.event
                who = e.player or e.team or ""
                score = update.score
                print(
                    f"  {fmt_clock(e):10} {e.type.value:12} {who:8} "
                    f"{score[engine.home]}-{score[engine.away]}"
                )
    engine.finish()

    print_box(engine)

    if commentator and commentator.lines:
        print("  COMMENTARY")
        for line in commentator.transcript():
            print(f"  - {line}")
        print()
    return 0


def cmd_query(args) -> int:
    engine = build_engine(args)
    start = int(args.start_minute * 60)
    end = int(args.end_minute * 60)
    if start >= end:
        print("start-minute must be less than end-minute", file=sys.stderr)
        return 1

    window = engine.window_score(start, end)
    home, away = engine.home, engine.away
    print()
    print(
        f"  window  minute {args.start_minute:g} to {args.end_minute:g}  "
        f"({fmt_second(start)} to {fmt_second(max(start, end - 1))})"
    )
    print(
        f"  points  {home} {window[home]}  -  {away} {window[away]}   "
        f"margin {engine.margin(start, end):+d}"
    )

    for team in (home, away):
        run = engine.best_run(team, start, end)
        if run.points:
            print(
                f"  best run  {team} +{run.points} over {run.seconds}s "
                f"({fmt_second(run.start)} to {fmt_second(run.end - 1)})"
            )
        else:
            print(f"  best run  {team} none, outscored throughout")

    print(f"  final   {home} {engine.score[home]}  -  {away} {engine.score[away]}")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bball", description="basketball play-by-play engine"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--synthetic", action="store_true", help="use the generated game")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--home", default="HOME")
        p.add_argument("--away", default="AWAY")
        p.add_argument("--lateness", type=int, default=5)
        p.add_argument("--jitter", type=int, default=0, help="scramble the feed within N events")
        p.add_argument("--home-edge", type=float, default=0.0)

    replay = sub.add_parser("replay", help="run a full game and print the box score")
    common(replay)
    replay.add_argument("--verbose", action="store_true", help="print every event")
    replay.add_argument("--no-agent", action="store_true")
    replay.set_defaults(func=cmd_replay)

    query = sub.add_parser("query", help="range query over a window of the game")
    common(query)
    query.add_argument("--start-minute", type=float, required=True)
    query.add_argument("--end-minute", type=float, required=True)
    query.set_defaults(func=cmd_query)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())