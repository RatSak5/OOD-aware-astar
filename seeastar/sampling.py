"""Candidate-generation strategies from the original SeeA* paper."""

from __future__ import annotations

from heapq import nsmallest
from math import ceil, isfinite, sqrt
import random
from typing import Mapping, Protocol, Sequence

import numpy as np

from algorithms.movement import Coord


def lowest_f_candidate(
    candidates: Sequence[Coord], f_values: Mapping[Coord, float]
) -> Coord:
    """Choose the minimum-f state from a sampled candidate set.

    Coordinates provide a deterministic tie-break when two f-values are equal.
    Candidate construction and expansion ranking are kept as separate steps,
    matching Algorithm 1 and the strategy-specific algorithms in the paper.
    """

    if not candidates:
        raise ValueError("Cannot choose a SeeA* expansion from an empty candidate set.")
    missing = [node for node in candidates if node not in f_values]
    if missing:
        raise KeyError(f"Candidates are missing f-values: {missing!r}.")
    return min(candidates, key=lambda node: (f_values[node], node))


class CandidateSampler(Protocol):
    """Interface for stateful or stateless SeeA* candidate strategies."""

    name: str

    def select(
        self,
        open_nodes: Sequence[Coord],
        f_values: Mapping[Coord, float],
        depths: Mapping[Coord, int],
    ) -> list[Coord]:
        """Return a duplicate-free candidate subset of OPEN."""

    def on_add(self, node: Coord) -> None:
        """Notify a sampler that a state entered OPEN."""

    def on_remove(self, node: Coord) -> None:
        """Notify a sampler that a state left OPEN."""

    def assert_open_consistent(self, open_nodes: Sequence[Coord]) -> None:
        """Validate any stateful sampler bookkeeping against OPEN."""


class UniformSampler:
    """Paper Algorithm 2: uniform K-subset, followed by minimum-f selection."""

    name = "uniform"

    def __init__(self, k: int, seed: int = 0) -> None:
        if k < 1:
            raise ValueError("K must be at least one.")
        self.k = int(k)
        self._rng = random.Random(seed)

    def select(
        self,
        open_nodes: Sequence[Coord],
        f_values: Mapping[Coord, float],
        depths: Mapping[Coord, int],
    ) -> list[Coord]:
        del f_values, depths
        population = list(open_nodes)
        if len(population) <= self.k:
            return population
        return self._rng.sample(population, self.k)

    def on_add(self, node: Coord) -> None:
        del node

    def on_remove(self, node: Coord) -> None:
        del node

    def assert_open_consistent(self, open_nodes: Sequence[Coord]) -> None:
        del open_nodes


class UCTLikeSampler:
    """Paper Algorithm 4 using its depth-adjusted score to form the subset."""

    name = "uct"

    def __init__(self, k: int, cb: float = 0.35) -> None:
        if k < 1:
            raise ValueError("K must be at least one.")
        if not isfinite(cb) or cb < 0.0:
            raise ValueError("cb must be finite and non-negative.")
        self.k = int(k)
        self.cb = float(cb)

    def exploration_score(
        self,
        node: Coord,
        f_values: Mapping[Coord, float],
        depths: Mapping[Coord, int],
        maximum_depth: int,
    ) -> float:
        """Evaluate Eq. (4): ``E=f-cb*sqrt(dmax/(1+d))``."""

        return float(
            f_values[node]
            - self.cb * sqrt(maximum_depth / (1.0 + depths[node]))
        )

    def select(
        self,
        open_nodes: Sequence[Coord],
        f_values: Mapping[Coord, float],
        depths: Mapping[Coord, int],
    ) -> list[Coord]:
        population = list(open_nodes)
        if len(population) <= self.k:
            return population
        maximum_depth = max(depths[node] for node in population)
        return nsmallest(
            self.k,
            population,
            key=lambda node: (
                self.exploration_score(node, f_values, depths, maximum_depth),
                f_values[node],
                node,
            ),
        )

    def on_add(self, node: Coord) -> None:
        del node

    def on_remove(self, node: Coord) -> None:
        del node

    def assert_open_consistent(self, open_nodes: Sequence[Coord]) -> None:
        del open_nodes


class CompetitiveClusterSampler:
    """Paper Algorithms 3/5 with online competitive clustering.

    The grid adaptation uses normalized cell coordinates as ``fh(node)``.
    A state's cluster assignment is retained if graph search later reopens it.
    """

    name = "cluster"

    def __init__(
        self,
        k: int,
        shape: tuple[int, int],
        *,
        num_clusters: int = 5,
        eta: float = 0.15,
        seed: int = 0,
    ) -> None:
        if k < 1:
            raise ValueError("K must be at least one.")
        if num_clusters < 1:
            raise ValueError("num_clusters must be positive.")
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError("shape must contain positive height and width.")
        if not isfinite(eta) or not 0.0 <= eta <= 1.0:
            raise ValueError("eta must be finite and in [0, 1].")
        self.k = int(k)
        self.shape = (int(shape[0]), int(shape[1]))
        self.num_clusters = int(num_clusters)
        self.eta = float(eta)
        self._rng = random.Random(seed)
        self.centers = np.asarray(
            [
                (self._rng.random(), self._rng.random())
                for _ in range(self.num_clusters)
            ],
            dtype=np.float64,
        )
        self._members: list[set[Coord]] = [set() for _ in range(self.num_clusters)]
        self._assignments: dict[Coord, int] = {}

    @property
    def cluster_members(self) -> tuple[frozenset[Coord], ...]:
        """Current OPEN members by cluster, exposed read-only for diagnostics."""

        return tuple(frozenset(members) for members in self._members)

    def _feature(self, node: Coord) -> np.ndarray:
        height, width = self.shape
        row, col = node
        if not (0 <= row < height and 0 <= col < width):
            raise ValueError(f"Cluster node {node!r} lies outside map shape {self.shape}.")
        return np.asarray(
            (row / max(height - 1, 1), col / max(width - 1, 1)),
            dtype=np.float64,
        )

    def on_add(self, node: Coord) -> None:
        """Assign a new OPEN state and apply the competitive center update."""

        if node in self._assignments:
            self._members[self._assignments[node]].add(node)
            return
        feature = self._feature(node)
        distances = np.sum((self.centers - feature) ** 2, axis=1)
        cluster_index = int(np.argmin(distances))
        self._assignments[node] = cluster_index
        self._members[cluster_index].add(node)
        self.centers[cluster_index] += self.eta * (feature - self.centers[cluster_index])

    def on_remove(self, node: Coord) -> None:
        """Remove an expanded state from its active cluster."""

        cluster_index = self._assignments.get(node)
        if cluster_index is None or node not in self._members[cluster_index]:
            raise RuntimeError(f"Cluster sampler has no active membership for {node}.")
        self._members[cluster_index].remove(node)

    def assert_open_consistent(self, open_nodes: Sequence[Coord]) -> None:
        expected = set(open_nodes)
        actual = set().union(*self._members)
        if actual != expected:
            raise RuntimeError("Cluster memberships do not match the current OPEN set.")
        if sum(len(members) for members in self._members) != len(actual):
            raise RuntimeError("A state appears in more than one active cluster.")

    def select(
        self,
        open_nodes: Sequence[Coord],
        f_values: Mapping[Coord, float],
        depths: Mapping[Coord, int],
    ) -> list[Coord]:
        del f_values, depths
        population = list(open_nodes)
        if not population:
            return []
        # The supplementary clustering pseudocode omits the small-OPEN guard
        # shown for uniform/UCT. The implementation note records this safeguard.
        if len(population) <= self.k:
            return population
        per_cluster = ceil(self.k / self.num_clusters)
        candidates: set[Coord] = set()
        for members in self._members:
            cluster_population = sorted(members)
            if len(cluster_population) <= per_cluster:
                chosen = cluster_population
            else:
                chosen = self._rng.sample(cluster_population, per_cluster)
            candidates.update(chosen)
        return sorted(candidates)
