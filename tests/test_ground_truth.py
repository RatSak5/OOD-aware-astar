from __future__ import annotations

import unittest

import numpy as np

from algorithms.dijkstra import dijkstra
from algorithms.movement import DEFAULT_COST_CONFIG, get_neighbors, get_transition_cost
from data.ground_truth import compute_ground_truth


class DijkstraTests(unittest.TestCase):
    def test_open_grid_uses_unit_diagonal_moves(self) -> None:
        grid = np.zeros((3, 3), dtype=np.uint8)
        result = compute_ground_truth(grid, (0, 0), (2, 2))
        self.assertEqual(result.optimal_path_cost, 2.0)
        self.assertEqual(result.distance_map[0, 0], 2.0)
        self.assertTrue(result.reachable_mask.all())
        self.assertEqual(len(result.optimal_path or ()), 3)

    def test_obstacles_require_a_valid_detour(self) -> None:
        grid = np.asarray([[0, 1, 0], [0, 0, 0], [0, 1, 0]], dtype=np.uint8)
        result = compute_ground_truth(grid, (0, 0), (0, 2))
        self.assertEqual(result.optimal_path_cost, 4.0)
        self.assertIsNotNone(result.optimal_path)
        path = result.optimal_path or ()
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (0, 2))
        for source, destination in zip(path, path[1:]):
            self.assertIn(destination, get_neighbors(grid, source))
        measured_cost = sum(
            get_transition_cost(grid, source, destination)
            for source, destination in zip(path, path[1:])
        )
        self.assertEqual(measured_cost, result.optimal_path_cost)

    def test_unreachable_cells_have_infinite_distance_and_false_mask(self) -> None:
        grid = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        result = compute_ground_truth(grid, (0, 0), (0, 2))
        self.assertIsNone(result.optimal_path)
        self.assertTrue(np.isinf(result.optimal_path_cost))
        self.assertFalse(result.reachable_mask[2, 2])
        self.assertTrue(np.isinf(result.distance_map[2, 2]))
        self.assertEqual(result.distance_map[0, 2], 0.0)

    def test_dijkstra_distance_map_is_float64(self) -> None:
        grid = np.zeros((4, 4), dtype=np.uint8)
        result = dijkstra(grid, (0, 0), DEFAULT_COST_CONFIG)
        self.assertEqual(result.distances.dtype, np.float64)
        self.assertEqual(result.distances[3, 3], 3.0)

    def test_start_and_goal_must_be_traversable(self) -> None:
        grid = np.zeros((2, 2), dtype=np.uint8)
        grid[0, 0] = 1
        with self.assertRaises(ValueError):
            compute_ground_truth(grid, (0, 0), (1, 1))


if __name__ == "__main__":
    unittest.main()
