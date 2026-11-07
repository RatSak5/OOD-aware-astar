from __future__ import annotations

import unittest

import numpy as np

from algorithms.movement import (
    DEFAULT_COST_CONFIG,
    GroundTruthCostConfig,
    get_neighbors,
    get_transition_cost,
)
from data.ground_truth import compute_ground_truth
from data.map_generator import MapGenerationConfig, generate_map


class NeighborGenerationTests(unittest.TestCase):
    def test_open_center_has_eight_neighbors(self) -> None:
        grid = np.zeros((3, 3), dtype=np.uint8)
        self.assertEqual(len(get_neighbors(grid, (1, 1))), 8)

    def test_corner_boundary_neighbors(self) -> None:
        grid = np.zeros((3, 3), dtype=np.uint8)
        self.assertEqual(get_neighbors(grid, (0, 0)), [(1, 0), (0, 1), (1, 1)])

    def test_obstacles_are_not_neighbors(self) -> None:
        grid = np.zeros((3, 3), dtype=np.uint8)
        grid[1, 0] = 1
        self.assertNotIn((1, 0), get_neighbors(grid, (1, 1)))

    def test_diagonal_corner_cutting_is_blocked_if_either_side_is_blocked(self) -> None:
        for blocked_side in ((0, 1), (1, 0)):
            with self.subTest(blocked_side=blocked_side):
                grid = np.zeros((2, 2), dtype=np.uint8)
                grid[blocked_side] = 1
                self.assertNotIn((1, 1), get_neighbors(grid, (0, 0)))
                self.assertTrue(np.isinf(get_transition_cost(grid, (0, 0), (1, 1))))

    def test_legal_diagonal_has_unit_cost(self) -> None:
        grid = np.zeros((2, 2), dtype=np.uint8)
        self.assertEqual(get_transition_cost(grid, (0, 0), (1, 1)), 1.0)
        self.assertEqual(get_transition_cost(grid, (0, 0), (0, 1)), 1.0)

    def test_invalid_transitions_have_infinite_cost(self) -> None:
        grid = np.zeros((3, 3), dtype=np.uint8)
        grid[0, 1] = 1
        self.assertTrue(np.isinf(get_transition_cost(grid, (1, 1), (0, 1))))
        self.assertTrue(np.isinf(get_transition_cost(grid, (1, 1), (0, 0))))
        self.assertTrue(np.isinf(get_transition_cost(grid, (1, 1), (1, 1))))

    def test_corner_cutting_cannot_be_disabled(self) -> None:
        with self.assertRaises(ValueError):
            GroundTruthCostConfig(prevent_diagonal_corner_cutting=False)


class MapGenerationTests(unittest.TestCase):
    def test_seed_reproduces_map_endpoints_and_structure(self) -> None:
        config = MapGenerationConfig(height=18, width=20, distribution="D4", seed=9102)
        first = generate_map(config)
        second = generate_map(config)
        np.testing.assert_array_equal(first.grid, second.grid)
        self.assertEqual(first.start, second.start)
        self.assertEqual(first.goal, second.goal)
        self.assertEqual(first.structure, second.structure)
        self.assertEqual(first.seed, config.seed)

    def test_corridor_distribution_varies_layouts_by_seed(self) -> None:
        first = generate_map(
            MapGenerationConfig(height=24, width=24, distribution="D5", seed=31)
        )
        second = generate_map(
            MapGenerationConfig(height=24, width=24, distribution="D5", seed=32)
        )
        self.assertFalse(np.array_equal(first.grid, second.grid))

    def test_endpoints_are_distinct_and_passable_for_all_distributions(self) -> None:
        for distribution in ("D1", "D2", "D3", "D4", "D5", "D6"):
            with self.subTest(distribution=distribution):
                generated = generate_map(
                    MapGenerationConfig(height=18, width=18, distribution=distribution, seed=177)
                )
                self.assertNotEqual(generated.start, generated.goal)
                self.assertEqual(generated.grid[generated.start], 0)
                self.assertEqual(generated.grid[generated.goal], 0)
                self.assertEqual(generated.grid.dtype, np.uint8)
                self.assertIsNotNone(
                    compute_ground_truth(
                        generated.grid, generated.start, generated.goal
                    ).optimal_path
                )

    def test_unreachable_pairs_are_kept_only_when_explicitly_enabled(self) -> None:
        config = MapGenerationConfig(
            height=6,
            width=6,
            distribution="D3",
            seed=0,
            obstacle_density=0.78,
            allow_unreachable=True,
        )
        generated = generate_map(config)
        ground_truth = compute_ground_truth(
            generated.grid, generated.start, generated.goal
        )
        self.assertIsNone(ground_truth.optimal_path)

        regenerated = generate_map(
            MapGenerationConfig(
                height=6,
                width=6,
                distribution="D3",
                seed=0,
                obstacle_density=0.78,
                allow_unreachable=False,
            )
        )
        regenerated_truth = compute_ground_truth(
            regenerated.grid, regenerated.start, regenerated.goal
        )
        self.assertIsNotNone(regenerated_truth.optimal_path)
        self.assertFalse(np.array_equal(generated.grid, regenerated.grid))

    def test_default_ground_truth_cost_config_is_unit_move_count(self) -> None:
        self.assertEqual(DEFAULT_COST_CONFIG.model, "unit_move_count")
        self.assertTrue(DEFAULT_COST_CONFIG.prevent_diagonal_corner_cutting)


if __name__ == "__main__":
    unittest.main()
