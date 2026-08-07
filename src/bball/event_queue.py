from __future__ import annotations

from .events import Event


class MinHeap:
    """Binary min-heap over Events, ordered by (elapsed, seq)."""

    def __init__(self) -> None:
        self._items: list[Event] = []

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def peek(self) -> Event:
        if not self._items:
            raise IndexError("peek from empty heap")
        return self._items[0]

    def push(self, event: Event) -> None:
        self._items.append(event)
        self._sift_up(len(self._items) - 1)

    def pop(self) -> Event:
        if not self._items:
            raise IndexError("pop from empty heap")
        top = self._items[0]
        last = self._items.pop()
        if self._items:
            self._items[0] = last
            self._sift_down(0)
        return top

    def _sift_up(self, i: int) -> None:
        item = self._items[i]
        while i > 0:
            parent = (i - 1) // 2
            if self._items[parent].key <= item.key:
                break
            self._items[i] = self._items[parent]
            i = parent
        self._items[i] = item

    def _sift_down(self, i: int) -> None:
        n = len(self._items)
        item = self._items[i]
        while True:
            left = 2 * i + 1
            right = left + 1
            smallest = i
            best = item.key
            if left < n and self._items[left].key < best:
                smallest = left
                best = self._items[left].key
            if right < n and self._items[right].key < best:
                smallest = right
                best = self._items[right].key
            if smallest == i:
                break
            self._items[i] = self._items[smallest]
            i = smallest
        self._items[i] = item


class LateEventError(Exception):
    """Raised when an event arrives already behind the watermark."""


class ReorderBuffer:
    """Holds events until the watermark proves nothing earlier can arrive.

    A live feed is roughly ordered but not exactly. We cannot sort a stream
    that has not finished arriving, so we hold events until we have seen
    something `lateness` seconds later, then release them in key order.
    The watermark asserts: nothing strictly before this second will arrive.
    """

    def __init__(self, lateness: int = 5, strict: bool = False) -> None:
        if lateness < 0:
            raise ValueError("lateness must be >= 0")
        self.lateness = lateness
        self.strict = strict
        self._heap = MinHeap()
        self._max_seen = -1  # highest elapsed observed on input
        self._watermark = -1  # everything strictly before this is released
        self.dropped: list[Event] = []

    def __len__(self) -> int:
        return len(self._heap)

    @property
    def watermark(self) -> int:
        return self._watermark

    def push(self, event: Event) -> list[Event]:
        """Ingest one event, return whatever became releasable."""
        if event.elapsed < self._watermark:
            # too late, its slot in the output order already passed
            if self.strict:
                raise LateEventError(
                    f"{event!r} arrived behind watermark {self._watermark}"
                )
            self.dropped.append(event)
            return []
        self._heap.push(event)
        self._max_seen = max(self._max_seen, event.elapsed)
        self._advance()
        return self._drain_ready()

    def _advance(self) -> None:
        candidate = self._max_seen - self.lateness
        # watermark never moves backward
        if candidate > self._watermark:
            self._watermark = candidate

    def _drain_ready(self) -> list[Event]:
        out: list[Event] = []
        while self._heap and self._heap.peek().elapsed < self._watermark:
            out.append(self._heap.pop())
        return out

    def flush(self) -> list[Event]:
        """End of feed. Release everything still held, in order."""
        out: list[Event] = []
        while self._heap:
            out.append(self._heap.pop())
        if out:
            self._watermark = out[-1].elapsed + 1
        return out


def ordered(events: list[Event], lateness: int = 5) -> list[Event]:
    """Run a whole batch through the buffer. Mostly for tests."""
    buf = ReorderBuffer(lateness=lateness)
    out: list[Event] = []
    for e in events:
        out.extend(buf.push(e))
    out.extend(buf.flush())
    return out