from __future__ import annotations


class Fenwick:
    """Binary indexed tree. Point update, prefix sum, both O(log n).

    Internally 1-indexed and the public API is 0-indexed to match Python.
    tree[i] stores the sum of the (i & -i) slots ending at i.
    """

    def __init__(self, size: int) -> None:
        if size < 0:
            raise ValueError("size must be >= 0")
        self.size = size
        self._tree = [0] * (size + 1)
        self._total = 0

    def __len__(self) -> int:
        return self.size

    def add(self, index: int, delta: int) -> None:
        """Add delta at index."""
        if not 0 <= index < self.size:
            raise IndexError(f"index {index} out of range for size {self.size}")
        self._total += delta
        i = index + 1
        while i <= self.size:
            self._tree[i] += delta
            i += i & -i  # jump to the next node that covers i

    def prefix(self, end: int) -> int:
        """Sum of [0, end). end == size gives the whole array."""
        if end <= 0:
            return 0
        if end > self.size:
            raise IndexError(f"end {end} out of range for size {self.size}")
        total = 0
        i = end
        while i > 0:
            total += self._tree[i]
            i -= i & -i  # strip the lowest set bit, walk left
        return total

    def range_sum(self, start: int, end: int) -> int:
        """Sum of [start, end). Valid only because addition is invertible."""
        if start < 0 or end > self.size:
            raise IndexError(f"range [{start},{end}) out of range")
        if start >= end:
            return 0
        return self.prefix(end) - self.prefix(start)

    def at(self, index: int) -> int:
        """Value of a single slot."""
        return self.range_sum(index, index + 1)

    @property
    def total(self) -> int:
        """Running total, kept incrementally so it stays O(1)."""
        return self._total

    def to_list(self) -> list[int]:
        """Rebuild the underlying array. O(n log n), for tests and debugging."""
        return [self.at(i) for i in range(self.size)]


class DualFenwick:
    """One Fenwick per team over the same second-by-second axis."""

    def __init__(self, size: int, home: str = "HOME", away: str = "AWAY") -> None:
        self.size = size
        self.home = home
        self.away = away
        self._trees = {home: Fenwick(size), away: Fenwick(size)}

    def _tree(self, team: str) -> Fenwick:
        try:
            return self._trees[team]
        except KeyError:
            raise KeyError(f"unknown team {team!r}, expected {self.home} or {self.away}")

    def add(self, team: str, index: int, points: int) -> None:
        self._tree(team).add(index, points)

    def prefix(self, team: str, end: int) -> int:
        return self._tree(team).prefix(end)

    def range_sum(self, team: str, start: int, end: int) -> int:
        return self._tree(team).range_sum(start, end)

    def window(self, start: int, end: int) -> dict[str, int]:
        """Both teams' points in [start, end)."""
        return {
            self.home: self.range_sum(self.home, start, end),
            self.away: self.range_sum(self.away, start, end),
        }

    def score(self) -> dict[str, int]:
        """Current full-game score."""
        return {
            self.home: self._trees[self.home].total,
            self.away: self._trees[self.away].total,
        }

    def margin(self, start: int = 0, end: int | None = None) -> int:
        """Home minus away over a window. Positive means home outscored away."""
        if end is None:
            end = self.size
        w = self.window(start, end)
        return w[self.home] - w[self.away]