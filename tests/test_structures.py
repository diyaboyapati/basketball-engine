from __future__ import annotations

import random

import pytest

from bball.events import Event, EventType, period_length, period_start_elapsed
from bball.event_queue import LateEventError, MinHeap, ReorderBuffer, ordered


def make_event(period: int, clock: int, seq: int, etype=EventType.MADE_2) -> Event:
    return Event(type=etype, period=period, clock=clock, seq=seq, team="HOME")


def random_events(n: int, rng: random.Random) -> list[Event]:
    """n events in true chronological order, seq assigned in that order."""
    out = []
    for i in range(n):
        period = rng.randint(1, 4)
        clock = rng.randint(0, period_length(period))
        out.append(make_event(period, clock, 0))
    out.sort(key=lambda e: e.elapsed)
    return [make_event(e.period, e.clock, i) for i, e in enumerate(out)]


# ---------- Event axis ----------


def test_elapsed_counts_up_while_clock_counts_down():
    tipoff = make_event(1, 720, 0)
    later = make_event(1, 600, 1)
    assert tipoff.elapsed == 0
    assert later.elapsed == 120


def test_period_boundaries_are_contiguous():
    end_q1 = make_event(1, 0, 0)
    start_q2 = make_event(2, 720, 1)
    assert end_q1.elapsed == start_q2.elapsed == 720


def test_overtime_is_five_minutes():
    assert period_length(5) == 300
    assert period_start_elapsed(5) == 2880
    ot = make_event(5, 300, 0)
    assert ot.elapsed == 2880


def test_invalid_clock_rejected():
    with pytest.raises(ValueError):
        make_event(1, 800, 0)
    with pytest.raises(ValueError):
        make_event(1, -1, 0)


def test_events_are_frozen():
    e = make_event(1, 720, 0)
    with pytest.raises(Exception):
        e.clock = 600


def test_key_uses_seq_as_tiebreaker():
    a = make_event(1, 600, 5)
    b = make_event(1, 600, 6)
    assert a.elapsed == b.elapsed
    assert a.key < b.key


# ---------- MinHeap ----------


def test_heap_pops_in_key_order():
    rng = random.Random(1)
    events = random_events(200, rng)
    shuffled = events[:]
    rng.shuffle(shuffled)

    heap = MinHeap()
    for e in shuffled:
        heap.push(e)

    popped = [heap.pop() for _ in range(len(heap))]
    assert [e.key for e in popped] == [e.key for e in events]


def test_heap_matches_sorted_under_interleaved_ops():
    rng = random.Random(7)
    events = random_events(300, rng)
    rng.shuffle(events)

    heap = MinHeap()
    model: list[Event] = []
    popped_heap, popped_model = [], []

    for e in events:
        heap.push(e)
        model.append(e)
        if rng.random() < 0.4:
            model.sort(key=lambda x: x.key)
            popped_model.append(model.pop(0))
            popped_heap.append(heap.pop())

    assert [e.key for e in popped_heap] == [e.key for e in popped_model]
    assert len(heap) == len(model)


def test_heap_peek_does_not_remove():
    heap = MinHeap()
    heap.push(make_event(2, 100, 1))
    heap.push(make_event(1, 100, 0))
    top = heap.peek()
    assert heap.peek() is top
    assert len(heap) == 2
    assert heap.pop() is top


def test_empty_heap_raises():
    heap = MinHeap()
    with pytest.raises(IndexError):
        heap.pop()
    with pytest.raises(IndexError):
        heap.peek()


# ---------- ReorderBuffer ----------


def jitter(events: list[Event], rng: random.Random, window: int) -> list[Event]:
    """Shuffle within a sliding window to fake a roughly-ordered feed."""
    out = events[:]
    for i in range(0, len(out), window):
        chunk = out[i : i + window]
        rng.shuffle(chunk)
        out[i : i + window] = chunk
    return out


def test_buffer_restores_order_from_jittered_feed():
    rng = random.Random(11)
    truth = random_events(400, rng)
    feed = jitter(truth, rng, window=6)

    result = ordered(feed, lateness=120)
    assert [e.key for e in result] == [e.key for e in truth]


def test_buffer_output_is_always_monotonic():
    rng = random.Random(13)
    truth = random_events(300, rng)
    feed = jitter(truth, rng, window=10)

    result = ordered(feed, lateness=60)
    keys = [e.key for e in result]
    assert keys == sorted(keys)


def test_buffer_holds_until_watermark_advances():
    buf = ReorderBuffer(lateness=30)
    released = buf.push(make_event(1, 720, 0))  # elapsed 0
    assert released == []
    assert len(buf) == 1

    released = buf.push(make_event(1, 700, 1))  # elapsed 20, wm still < 0
    assert released == []

    released = buf.push(make_event(1, 660, 2))  # elapsed 60, wm = 30
    assert [e.elapsed for e in released] == [0, 20]
    assert len(buf) == 1


def test_watermark_never_moves_backward():
    buf = ReorderBuffer(lateness=10)
    buf.push(make_event(1, 600, 0))  # elapsed 120, wm = 110
    high = buf.watermark
    buf.push(make_event(1, 610, 1))  # elapsed 110, earlier arrival
    assert buf.watermark == high


def test_late_event_dropped_in_lenient_mode():
    buf = ReorderBuffer(lateness=0)
    buf.push(make_event(1, 600, 0))  # elapsed 120, wm = 120
    released = buf.push(make_event(1, 700, 1))  # elapsed 20, way behind
    assert released == []
    assert len(buf.dropped) == 1


def test_late_event_raises_in_strict_mode():
    buf = ReorderBuffer(lateness=0, strict=True)
    buf.push(make_event(1, 600, 0))
    with pytest.raises(LateEventError):
        buf.push(make_event(1, 700, 1))


def test_flush_releases_everything_in_order():
    rng = random.Random(17)
    truth = random_events(50, rng)
    buf = ReorderBuffer(lateness=10_000)  # nothing ages out mid-feed

    released = []
    for e in jitter(truth, rng, window=8):
        released.extend(buf.push(e))
    assert released == []

    released.extend(buf.flush())
    assert [e.key for e in released] == [e.key for e in truth]
    assert len(buf) == 0


def test_no_events_lost_or_duplicated():
    rng = random.Random(19)
    truth = random_events(500, rng)
    feed = jitter(truth, rng, window=5)

    buf = ReorderBuffer(lateness=90)
    out = []
    for e in feed:
        out.extend(buf.push(e))
    out.extend(buf.flush())

    assert len(out) + len(buf.dropped) == len(truth)
    assert {e.seq for e in out} | {e.seq for e in buf.dropped} == {
        e.seq for e in truth
    }


def test_lateness_zero_still_orders_ties_at_same_second():
    events = [make_event(1, 600, 2), make_event(1, 600, 1), make_event(1, 600, 0)]
    result = ordered(events, lateness=0)
    assert [e.seq for e in result] == [0, 1, 2]

from bball.fenwick import DualFenwick, Fenwick
from bball.segment_tree import IDENTITY, MaxSubarrayTree, RunTree


# ---------- brute-force ways ----------


def brute_range_sum(values: list[int], start: int, end: int) -> int:
    return sum(values[start:end])


def brute_max_subarray(values: list[int], start: int, end: int):
    """Kadane over a slice. Returns (best, start, end_inclusive)."""
    window = values[start:end]
    if not window:
        return None
    best = window[0]
    best_lo = best_hi = start
    cur = window[0]
    cur_lo = start
    for i in range(1, len(window)):
        idx = start + i
        if cur + window[i] >= window[i]:
            cur += window[i]
        else:
            cur = window[i]
            cur_lo = idx
        if cur > best:
            best, best_lo, best_hi = cur, cur_lo, idx
    return best, best_lo, best_hi


# ---------- Fenwick ----------


def test_fenwick_empty_reads_zero():
    f = Fenwick(10)
    assert f.total == 0
    assert f.prefix(10) == 0
    assert f.range_sum(3, 7) == 0
    assert f.to_list() == [0] * 10


def test_fenwick_single_point():
    f = Fenwick(8)
    f.add(3, 5)
    assert f.at(3) == 5
    assert f.prefix(3) == 0
    assert f.prefix(4) == 5
    assert f.range_sum(3, 4) == 5
    assert f.range_sum(0, 3) == 0
    assert f.total == 5


def test_fenwick_matches_list_model():
    rng = random.Random(101)
    n = 200
    f = Fenwick(n)
    model = [0] * n

    for _ in range(1000):
        i = rng.randrange(n)
        d = rng.randint(-5, 5)
        f.add(i, d)
        model[i] += d

        start = rng.randrange(n)
        end = rng.randint(start, n)
        assert f.range_sum(start, end) == brute_range_sum(model, start, end)

    assert f.to_list() == model
    assert f.total == sum(model)


def test_fenwick_prefix_covers_all_boundaries():
    rng = random.Random(103)
    n = 64
    f = Fenwick(n)
    model = [rng.randint(0, 3) for _ in range(n)]
    for i, v in enumerate(model):
        f.add(i, v)

    for end in range(n + 1):
        assert f.prefix(end) == sum(model[:end])


def test_fenwick_empty_and_reversed_ranges():
    f = Fenwick(10)
    f.add(5, 7)
    assert f.range_sum(5, 5) == 0
    assert f.range_sum(8, 3) == 0
    assert f.prefix(0) == 0


def test_fenwick_rejects_out_of_range():
    f = Fenwick(10)
    with pytest.raises(IndexError):
        f.add(10, 1)
    with pytest.raises(IndexError):
        f.add(-1, 1)
    with pytest.raises(IndexError):
        f.prefix(11)


# ---------- DualFenwick ----------


def test_dual_fenwick_keeps_teams_separate():
    d = DualFenwick(2880)
    d.add("HOME", 100, 3)
    d.add("AWAY", 100, 2)
    d.add("HOME", 500, 2)

    assert d.score() == {"HOME": 5, "AWAY": 2}
    assert d.window(0, 200) == {"HOME": 3, "AWAY": 2}
    assert d.window(200, 2880) == {"HOME": 2, "AWAY": 0}


def test_dual_fenwick_margin_sign():
    d = DualFenwick(2880)
    d.add("AWAY", 50, 6)
    d.add("HOME", 60, 2)
    assert d.margin() == -4
    assert d.margin(55, 2880) == 2


def test_dual_fenwick_unknown_team():
    d = DualFenwick(100)
    with pytest.raises(KeyError):
        d.add("NOBODY", 0, 2)


# ---------- MaxSubarrayTree ----------


def test_segment_tree_matches_kadane():
    rng = random.Random(201)
    n = 120
    tree = MaxSubarrayTree(n)
    model = [0] * n

    for _ in range(400):
        i = rng.randrange(n)
        d = rng.randint(-4, 4)
        tree.add(i, d)
        model[i] += d

        start = rng.randrange(n)
        end = rng.randint(start + 1, n)
        node = tree.query(start, end)
        expected = brute_max_subarray(model, start, end)
        assert node.best == expected[0]
        assert sum(model[node.best_start : node.best_end + 1]) == node.best


def test_segment_tree_all_negative_picks_least_bad():
    tree = MaxSubarrayTree(5)
    for i, v in enumerate([-3, -1, -4, -2, -5]):
        tree.set(i, v)
    node = tree.query(0, 5)
    assert node.best == -1
    assert node.best_start == node.best_end == 1


def test_segment_tree_all_positive_takes_whole_range():
    tree = MaxSubarrayTree(6)
    for i in range(6):
        tree.set(i, 2)
    node = tree.query(0, 6)
    assert node.best == 12
    assert node.best_start == 0
    assert node.best_end == 5


def test_segment_tree_finds_straddling_run():
    # the answer crosses the midpoint, which prefix subtraction cannot recover
    values = [5, -1, -1, 4, 4, -1, -1, 5]
    tree = MaxSubarrayTree(len(values))
    for i, v in enumerate(values):
        tree.set(i, v)

    node = tree.query(2, 6)
    assert node.best == 8
    assert node.best_start == 3
    assert node.best_end == 4


def test_segment_tree_prefix_subtraction_would_be_wrong():
    values = [9, -9, 4, 4, -9, 9]
    tree = MaxSubarrayTree(len(values))
    for i, v in enumerate(values):
        tree.set(i, v)

    whole = tree.query(0, 6).best
    left = tree.query(0, 2).best
    inner = tree.query(2, 4).best
    assert inner == 8
    assert whole - left != inner  # no inverse, so subtraction is meaningless


def test_segment_tree_total_matches_sum():
    rng = random.Random(203)
    n = 50
    tree = MaxSubarrayTree(n)
    model = [rng.randint(-5, 5) for _ in range(n)]
    for i, v in enumerate(model):
        tree.set(i, v)

    for _ in range(100):
        start = rng.randrange(n)
        end = rng.randint(start + 1, n)
        assert tree.query(start, end).total == sum(model[start:end])


def test_segment_tree_single_element_ranges():
    tree = MaxSubarrayTree(10)
    tree.set(4, 7)
    node = tree.query(4, 5)
    assert node.best == node.total == 7
    assert node.best_start == node.best_end == 4


def test_segment_tree_empty_range_is_identity():
    tree = MaxSubarrayTree(10)
    assert tree.query(3, 3) is IDENTITY


def test_segment_tree_rejects_out_of_range():
    tree = MaxSubarrayTree(10)
    with pytest.raises(IndexError):
        tree.add(10, 1)
    with pytest.raises(IndexError):
        tree.query(0, 11)


# ---------- RunTree ----------


def test_run_tree_finds_scoring_run():
    rt = RunTree(2880)
    for sec in (100, 120, 140, 160):
        rt.add("HOME", sec, 3)

    run = rt.best_run("HOME")
    assert run.points == 12
    assert run.start == 100
    assert run.end == 161
    assert run.seconds == 61


def test_run_tree_opponent_scoring_breaks_the_run():
    rt = RunTree(2880)
    rt.add("HOME", 10, 5)
    rt.add("AWAY", 20, 20)
    rt.add("HOME", 30, 6)

    assert rt.best_run("HOME").points == 6
    assert rt.best_run("HOME").start == 30
    assert rt.best_run("AWAY").points == 20


def test_run_tree_negation_is_symmetric():
    rng = random.Random(207)
    rt = RunTree(600)
    for _ in range(80):
        team = rng.choice(["HOME", "AWAY"])
        rt.add(team, rng.randrange(600), rng.choice([1, 2, 3]))

    for start, end in [(0, 600), (100, 400), (250, 260)]:
        h = rt.best_run("HOME", start, end)
        a = rt.best_run("AWAY", start, end)
        assert h.points >= 0 and a.points >= 0
        # both teams cannot own a net run over the same exact window
        assert not (h.points > 0 and a.points > 0 and h.start == a.start and h.end == a.end)


def test_run_tree_no_run_returns_zero():
    rt = RunTree(600)
    rt.add("AWAY", 50, 10)
    run = rt.best_run("HOME")
    assert run.points == 0
    assert run.seconds == 0


def test_run_tree_window_restricts_the_answer():
    rt = RunTree(2880)
    for sec in (100, 110, 120):
        rt.add("HOME", sec, 3)
    for sec in (2000, 2010):
        rt.add("HOME", sec, 2)

    assert rt.best_run("HOME", 0, 2880).points == 13
    assert rt.best_run("HOME", 1500, 2880).points == 4
    assert rt.best_run("HOME", 1500, 2880).start == 2000


def test_run_tree_overall_picks_bigger_side():
    rt = RunTree(2880)
    rt.add("HOME", 100, 4)
    rt.add("AWAY", 500, 9)
    best = rt.best_run_overall()
    assert best.team == "AWAY"
    assert best.points == 9