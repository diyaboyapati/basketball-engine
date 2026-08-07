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