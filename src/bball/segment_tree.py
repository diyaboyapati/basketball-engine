from __future__ import annotations

from dataclasses import dataclass

NEG = -(1 << 60)  # stands in for -infinity, stays an int


@dataclass(frozen=True, slots=True)
class Node:
    """Everything needed to merge this range with a neighbor."""

    total: int
    pref: int  # best sum starting at the left edge
    pref_end: int
    suf: int  # best sum ending at the right edge
    suf_start: int
    best: int  # best sum anywhere inside
    best_start: int
    best_end: int


IDENTITY = Node(0, NEG, -1, NEG, -1, NEG, -1, -1)


def leaf(value: int, index: int) -> Node:
    return Node(value, value, index, value, index, value, index, index)


def merge(a: Node, b: Node) -> Node:
    """Combine adjacent ranges. Ties prefer the leftmost window."""
    if a is IDENTITY:
        return b
    if b is IDENTITY:
        return a

    total = a.total + b.total

    # best prefix either stops inside a, or swallows a and continues into b
    if a.pref >= a.total + b.pref:
        pref, pref_end = a.pref, a.pref_end
    else:
        pref, pref_end = a.total + b.pref, b.pref_end

    # best suffix either starts inside b, or swallows b and reaches back into a
    if b.suf > b.total + a.suf:
        suf, suf_start = b.suf, b.suf_start
    else:
        suf, suf_start = b.total + a.suf, a.suf_start

    # best is left-only, right-only, or straddling the seam
    best, best_start, best_end = a.best, a.best_start, a.best_end
    cross = a.suf + b.pref
    if cross > best:
        best, best_start, best_end = cross, a.suf_start, b.pref_end
    if b.best > best:
        best, best_start, best_end = b.best, b.best_start, b.best_end

    return Node(total, pref, pref_end, suf, suf_start, best, best_start, best_end)


class MaxSubarrayTree:
    """Point update, max-subarray-in-range query, both O(log n).

    Cannot be a Fenwick tree: max-subarray has no inverse, so knowing the
    answer for [0,b) and [0,a) tells you nothing about [a,b). The real run
    may straddle a, and subtraction cannot recover it. Segment trees merge
    partial answers upward instead of subtracting them.
    """

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("size must be > 0")
        self.size = size
        self._values = [0] * size
        self._nodes: list[Node] = [IDENTITY] * (4 * size)
        self._build(1, 0, size)

    def _build(self, node: int, lo: int, hi: int) -> None:
        if hi - lo == 1:
            self._nodes[node] = leaf(self._values[lo], lo)
            return
        mid = (lo + hi) // 2
        self._build(2 * node, lo, mid)
        self._build(2 * node + 1, mid, hi)
        self._nodes[node] = merge(self._nodes[2 * node], self._nodes[2 * node + 1])

    def add(self, index: int, delta: int) -> None:
        """Add delta to one slot."""
        if not 0 <= index < self.size:
            raise IndexError(f"index {index} out of range for size {self.size}")
        self._values[index] += delta
        self._update(1, 0, self.size, index)

    def set(self, index: int, value: int) -> None:
        if not 0 <= index < self.size:
            raise IndexError(f"index {index} out of range for size {self.size}")
        self._values[index] = value
        self._update(1, 0, self.size, index)

    def _update(self, node: int, lo: int, hi: int, index: int) -> None:
        if hi - lo == 1:
            self._nodes[node] = leaf(self._values[lo], lo)
            return
        mid = (lo + hi) // 2
        if index < mid:
            self._update(2 * node, lo, mid, index)
        else:
            self._update(2 * node + 1, mid, hi, index)
        self._nodes[node] = merge(self._nodes[2 * node], self._nodes[2 * node + 1])

    def query(self, start: int, end: int) -> Node:
        """Merged node for [start, end). IDENTITY if the range is empty."""
        if start < 0 or end > self.size:
            raise IndexError(f"range [{start},{end}) out of range")
        if start >= end:
            return IDENTITY
        return self._query(1, 0, self.size, start, end)

    def _query(self, node: int, lo: int, hi: int, start: int, end: int) -> Node:
        if start <= lo and hi <= end:
            return self._nodes[node]
        mid = (lo + hi) // 2
        if end <= mid:
            return self._query(2 * node, lo, mid, start, end)
        if start >= mid:
            return self._query(2 * node + 1, mid, hi, start, end)
        left = self._query(2 * node, lo, mid, start, end)
        right = self._query(2 * node + 1, mid, hi, start, end)
        return merge(left, right)

    def at(self, index: int) -> int:
        return self._values[index]

    def to_list(self) -> list[int]:
        return self._values[:]


@dataclass(frozen=True, slots=True)
class Run:
    """A scoring run: who, how many, and the window it happened in."""

    team: str
    points: int
    start: int  # inclusive second
    end: int  # exclusive second

    @property
    def seconds(self) -> int:
        return self.end - self.start

    def __repr__(self) -> str:
        return f"<Run {self.team} +{self.points} [{self.start},{self.end})>"


class RunTree:
    """Best scoring run for either team, over any window.

    Stores the signed margin per second in one tree and its negation in
    another, so both teams' runs are a plain max-subarray query.
    """

    def __init__(self, size: int, home: str = "HOME", away: str = "AWAY") -> None:
        self.size = size
        self.home = home
        self.away = away
        self._trees = {home: MaxSubarrayTree(size), away: MaxSubarrayTree(size)}

    def add(self, team: str, index: int, points: int) -> None:
        if team not in self._trees:
            raise KeyError(f"unknown team {team!r}, expected {self.home} or {self.away}")
        other = self.away if team == self.home else self.home
        self._trees[team].add(index, points)
        self._trees[other].add(index, -points)

    def best_run(self, team: str, start: int = 0, end: int | None = None) -> Run:
        """Largest margin this team built inside [start, end)."""
        if end is None:
            end = self.size
        node = self._trees[team].query(start, end)
        if node is IDENTITY or node.best <= 0:
            return Run(team, 0, start, start)
        return Run(team, node.best, node.best_start, node.best_end + 1)

    def best_run_overall(self, start: int = 0, end: int | None = None) -> Run:
        """Bigger of the two teams' runs. Home wins exact ties."""
        h = self.best_run(self.home, start, end)
        a = self.best_run(self.away, start, end)
        return h if h.points >= a.points else a