"""Group items joined by qualifying pairs into clusters (union find).

Shared by the word overlap and embedding similarity signals: both produce pairs
of reviews with a similarity at or above a threshold, and both report each
connected group as one flag.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Cluster:
    members: tuple[int, ...]
    """Item indices, in ascending order."""
    similarities: tuple[float, ...]
    """Similarity of every qualifying pair inside the cluster."""

    @property
    def peak(self) -> float:
        return max(self.similarities)

    @property
    def mean(self) -> float:
        return sum(self.similarities) / len(self.similarities)

    @property
    def pair_count(self) -> int:
        return len(self.similarities)


def union_clusters(n: int, pairs: Iterable[tuple[int, int, float]]) -> list[Cluster]:
    """Return clusters of size two or more, ordered by their smallest member index."""
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = list(pairs)
    for i, j, _ in pairs:
        parent[find(i)] = find(j)

    members: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        members[find(i)].append(i)
    sims: dict[int, list[float]] = defaultdict(list)
    for i, j, s in pairs:
        sims[find(i)].append(s)

    out = [
        Cluster(members=tuple(sorted(idx)), similarities=tuple(sims[root]))
        for root, idx in members.items()
        if len(idx) >= 2
    ]
    out.sort(key=lambda c: c.members[0])
    return out
