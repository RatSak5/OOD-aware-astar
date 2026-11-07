"""Reproducible grid-world maps for the D1-D6 distributions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt

import numpy as np

from algorithms.movement import Coord, DEFAULT_COST_CONFIG, GroundTruthCostConfig, get_neighbors, validate_map

DEFAULT_DENSITIES: dict[str, float] = {
    "D1": 0.10,
    "D2": 0.25,
    "D3": 0.40,
    "D4": 0.28,
    "D5": 0.35,
    "D6": 0.30,
}
DISTRIBUTIONS = tuple(DEFAULT_DENSITIES)


@dataclass(frozen=True, slots=True)
class MapGenerationConfig:
    """All parameters needed to reproduce one generated map.

    D1-D3 use a fixed-count uniform random obstacle process. D4 samples compact
    clusters. D5 carves a connected grid of corridors. D6 randomly selects a
    random, clustered, corridor, or room-and-connector structure. A provided
    ``obstacle_density`` overrides the distribution default; for D5 the value
    determines corridor spacing unless ``corridor_spacing`` is explicitly set.
    """

    height: int = 32
    width: int = 32
    distribution: str = "D1"
    seed: int = 0
    obstacle_density: float | None = None
    cluster_size: int = 7
    cluster_count: int = 6
    corridor_width: int = 2
    corridor_spacing: int | None = None
    max_attempts: int = 128
    allow_unreachable: bool = False

    def __post_init__(self) -> None:
        if self.height < 2 or self.width < 2:
            raise ValueError("Map height and width must both be at least 2.")
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(f"distribution must be one of {DISTRIBUTIONS}.")
        if not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer.")
        if self.obstacle_density is not None and not 0.0 <= self.obstacle_density < 1.0:
            raise ValueError("obstacle_density must be in [0, 1).")
        if self.cluster_size < 1 or self.cluster_count < 1:
            raise ValueError("cluster_size and cluster_count must be positive.")
        if self.corridor_width < 1:
            raise ValueError("corridor_width must be positive.")
        if self.corridor_spacing is not None and self.corridor_spacing < self.corridor_width:
            raise ValueError("corridor_spacing must be at least corridor_width.")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive.")

    @property
    def density(self) -> float:
        """Resolved obstacle density for this distribution."""

        if self.obstacle_density is not None:
            return self.obstacle_density
        return DEFAULT_DENSITIES[self.distribution]


@dataclass(frozen=True, slots=True)
class GridMap:
    """One map and its endpoint and generation metadata."""

    grid: np.ndarray
    start: Coord
    goal: Coord
    distribution: str
    seed: int
    structure: str = ""

    def __post_init__(self) -> None:
        array = validate_map(self.grid).astype(np.uint8, copy=True)
        object.__setattr__(self, "grid", array)
        if self.distribution not in DISTRIBUTIONS:
            raise ValueError(f"distribution must be one of {DISTRIBUTIONS}.")
        if self.start == self.goal:
            raise ValueError("Start and goal must be distinct cells.")
        for label, coordinate in (("start", self.start), ("goal", self.goal)):
            if len(coordinate) != 2:
                raise ValueError(f"{label} must be a (row, column) coordinate.")
            row, col = coordinate
            if (
                isinstance(row, (bool, np.bool_))
                or isinstance(col, (bool, np.bool_))
                or not isinstance(row, (int, np.integer))
                or not isinstance(col, (int, np.integer))
            ):
                raise ValueError(f"{label} coordinates must be integers.")
            row, col = int(row), int(col)
            if not (0 <= row < array.shape[0] and 0 <= col < array.shape[1]):
                raise ValueError(f"{label} coordinate is outside the map.")
            if array[row, col] != 0:
                raise ValueError(f"{label} cell must be traversable.")
        if not 0 <= self.seed < 2**64:
            raise ValueError("seed must be an unsigned 64-bit integer.")


def _random_obstacles(shape: tuple[int, int], density: float, rng: np.random.Generator) -> np.ndarray:
    height, width = shape
    total = height * width
    obstacle_count = min(int(round(total * density)), total - 2)
    grid = np.zeros(shape, dtype=np.uint8)
    if obstacle_count:
        chosen = rng.choice(total, size=obstacle_count, replace=False)
        grid.flat[chosen] = 1
    return grid


def _clustered_obstacles(
    shape: tuple[int, int], density: float, cluster_size: int, cluster_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = shape
    total = height * width
    obstacle_count = min(int(round(total * density)), total - 2)
    grid = np.zeros(shape, dtype=np.uint8)
    if obstacle_count == 0:
        return grid

    candidates: set[tuple[int, int]] = set()
    half = cluster_size // 2
    maximum_centers = max(total * 2, cluster_count)
    attempts = 0
    centers_to_place = cluster_count
    while len(candidates) < obstacle_count and attempts < maximum_centers:
        for _ in range(centers_to_place):
            center_row = int(rng.integers(0, height))
            center_col = int(rng.integers(0, width))
            for row in range(max(0, center_row - half), min(height, center_row + half + 1)):
                for col in range(max(0, center_col - half), min(width, center_col + half + 1)):
                    candidates.add((row, col))
        attempts += centers_to_place
        centers_to_place = 1

    # The fallback matters for very small cluster sizes and extreme densities.
    candidate_array = np.array(sorted(candidates), dtype=np.int32)
    selected_indices = rng.choice(len(candidate_array), size=obstacle_count, replace=False)
    selected = candidate_array[selected_indices]
    grid[selected[:, 0], selected[:, 1]] = 1
    return grid


def _corridor_obstacles(
    shape: tuple[int, int], density: float, corridor_width: int,
    corridor_spacing: int | None, rng: np.random.Generator,
) -> np.ndarray:
    height, width = shape
    if density <= 0:
        return np.zeros(shape, dtype=np.uint8)
    if corridor_spacing is None:
        free_fraction = 1.0 - sqrt(density)
        derived = int(round(corridor_width / max(free_fraction, 1e-6)))
        spacing = max(corridor_width, derived)
    else:
        spacing = corridor_spacing
    grid = np.ones(shape, dtype=np.uint8)

    def carve_rows(center: int) -> None:
        low = max(0, center - corridor_width // 2)
        high = min(height, low + corridor_width)
        low = max(0, high - corridor_width)
        grid[low:high, :] = 0

    def carve_columns(center: int) -> None:
        low = max(0, center - corridor_width // 2)
        high = min(width, low + corridor_width)
        low = max(0, high - corridor_width)
        grid[:, low:high] = 0

    jitter = max(0, spacing // 4)

    def sample_lane_centers(limit: int) -> list[int]:
        offset = int(rng.integers(0, spacing))
        if offset >= limit:
            offset = int(rng.integers(0, limit))
        centers: list[int] = []
        center = offset
        while center < limit:
            centers.append(center)
            variation = int(rng.integers(-jitter, jitter + 1)) if jitter else 0
            center += max(corridor_width, spacing + variation)
        return centers

    row_centers = sample_lane_centers(height)
    col_centers = sample_lane_centers(width)
    for center in row_centers:
        carve_rows(center)
    for center in col_centers:
        carve_columns(center)
    if not row_centers:
        carve_rows(int(rng.integers(0, height)))
    if not col_centers:
        carve_columns(int(rng.integers(0, width)))
    return grid


def _rooms_and_connectors(
    shape: tuple[int, int], density: float, corridor_width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = shape
    if density <= 0:
        return np.zeros(shape, dtype=np.uint8)
    grid = np.ones(shape, dtype=np.uint8)
    room_count = max(3, min(8, (height * width) // 64))
    centers: list[Coord] = []
    for _ in range(room_count):
        room_h = int(rng.integers(2, max(3, min(height, 7))))
        room_w = int(rng.integers(2, max(3, min(width, 7))))
        top = int(rng.integers(0, max(1, height - room_h + 1)))
        left = int(rng.integers(0, max(1, width - room_w + 1)))
        grid[top : min(height, top + room_h), left : min(width, left + room_w)] = 0
        centers.append((min(height - 1, top + room_h // 2), min(width - 1, left + room_w // 2)))

    radius = corridor_width // 2
    for source, destination in zip(centers, centers[1:]):
        row, col = source
        goal_row, goal_col = destination
        while col != goal_col:
            low = max(0, col - radius)
            high = min(width, low + corridor_width)
            low = max(0, high - corridor_width)
            grid[row, low:high] = 0
            col += 1 if goal_col > col else -1
        while row != goal_row:
            low = max(0, row - radius)
            high = min(height, low + corridor_width)
            low = max(0, high - corridor_width)
            grid[low:high, col] = 0
            row += 1 if goal_row > row else -1

    target_free_count = max(2, int(round(height * width * (1.0 - density))))
    open_cells = set(map(tuple, np.argwhere(grid == 0).tolist()))
    frontier: list[Coord] = []
    frontier_set: set[Coord] = set()
    for row, col in sorted(open_cells):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            candidate = (row + dr, col + dc)
            if (
                0 <= candidate[0] < height
                and 0 <= candidate[1] < width
                and grid[candidate] == 1
                and candidate not in frontier_set
            ):
                frontier.append(candidate)
                frontier_set.add(candidate)
    while len(open_cells) < target_free_count:
        if not frontier:
            break
        # Expand rooms along their connected boundary until the requested
        # traversable fraction is reached. A list plus set avoids rescanning
        # the full open region at every cell added.
        selected_index = int(rng.integers(0, len(frontier)))
        row, col = frontier[selected_index]
        frontier[selected_index] = frontier[-1]
        frontier.pop()
        frontier_set.remove((row, col))
        grid[row, col] = 0
        open_cells.add((row, col))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            candidate = (row + dr, col + dc)
            if (
                0 <= candidate[0] < height
                and 0 <= candidate[1] < width
                and grid[candidate] == 1
                and candidate not in frontier_set
            ):
                frontier.append(candidate)
                frontier_set.add(candidate)
    return grid


def _make_obstacles(
    config: MapGenerationConfig, rng: np.random.Generator
) -> tuple[np.ndarray, str]:
    shape = (config.height, config.width)
    if config.distribution in {"D1", "D2", "D3"}:
        return _random_obstacles(shape, config.density, rng), "random"
    if config.distribution == "D4":
        return (
            _clustered_obstacles(
                shape, config.density, config.cluster_size, config.cluster_count, rng
            ),
            "clustered",
        )
    if config.distribution == "D5":
        return (
            _corridor_obstacles(
                shape, config.density, config.corridor_width, config.corridor_spacing, rng
            ),
            "corridors",
        )

    structure = str(rng.choice(("random", "clustered", "corridors", "rooms")))
    if structure == "random":
        grid = _random_obstacles(shape, config.density, rng)
    elif structure == "clustered":
        grid = _clustered_obstacles(
            shape, config.density, config.cluster_size, config.cluster_count, rng
        )
    elif structure == "corridors":
        grid = _corridor_obstacles(
            shape, config.density, config.corridor_width, config.corridor_spacing, rng
        )
    else:
        grid = _rooms_and_connectors(
            shape, config.density, config.corridor_width, rng
        )
    return grid, structure


def _reachable_from(
    grid: np.ndarray,
    start: Coord,
    cost_config: GroundTruthCostConfig,
) -> set[Coord]:
    seen = {start}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in get_neighbors(grid, current, cost_config):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def generate_map(
    config: MapGenerationConfig,
    cost_config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
) -> GridMap:
    """Generate one reproducible map and traversable, distinct endpoints.

    With ``allow_unreachable=False``, a complete obstacle map is redrawn when
    randomly selected endpoints are disconnected. The returned root seed and
    full ``MapGenerationConfig`` identify how to reproduce this sample.
    """

    if not isinstance(config, MapGenerationConfig):
        raise TypeError("config must be a MapGenerationConfig instance.")
    rng = np.random.default_rng(config.seed)
    for _ in range(config.max_attempts):
        grid, structure = _make_obstacles(config, rng)
        free_cells = np.argwhere(grid == 0)
        if len(free_cells) < 2:
            continue
        chosen = rng.choice(len(free_cells), size=2, replace=False)
        start = tuple(int(v) for v in free_cells[chosen[0]])
        goal = tuple(int(v) for v in free_cells[chosen[1]])
        if not config.allow_unreachable and goal not in _reachable_from(grid, start, cost_config):
            continue
        return GridMap(
            grid=grid,
            start=start,
            goal=goal,
            distribution=config.distribution,
            seed=config.seed,
            structure=structure,
        )
    raise RuntimeError(
        f"Could not generate a map with reachable distinct endpoints for seed "
        f"{config.seed} after {config.max_attempts} attempts."
    )
