from __future__ import annotations

import unittest

import numpy as np

from algorithms.astar import astar, chebyshev_heuristic, zero_heuristic
from algorithms.dijkstra import dijkstra
from algorithms.movement import DEFAULT_COST_CONFIG, get_neighbors, get_transition_cost
from data.ground_truth import compute_ground_truth
from data.map_generator import MapGenerationConfig, generate_map


class AStarCorrectnessTests(unittest.TestCase):
    def test_zero_heuristic_matches_dijkstra(self) -> None:
        grid = np.asarray(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        start, goal = (3, 1), (0, 5)
        expected = compute_ground_truth(grid, start, goal)
        result = astar(grid, start, goal, heuristic=zero_heuristic)
        self.assertTrue(result.found)
        self.assertEqual(result.found_ground_truth_cost, expected.optimal_path_cost)

    def test_chebyshev_astar_matches_dijkstra_on_all_distributions(self) -> None:
        for index, distribution in enumerate(("D1", "D2", "D3", "D4", "D5", "D6")):
            with self.subTest(distribution=distribution):
                generated = generate_map(
                    MapGenerationConfig(
                        height=18,
                        width=19,
                        distribution=distribution,
                        seed=6200 + index,
                    )
                )
                expected = compute_ground_truth(
                    generated.grid, generated.start, generated.goal
                )
                result = astar(generated.grid, generated.start, generated.goal)
                zero_result = astar(
                    generated.grid,
                    generated.start,
                    generated.goal,
                    heuristic=zero_heuristic,
                )
                self.assertTrue(result.found)
                self.assertTrue(zero_result.found)
                self.assertAlmostEqual(
                    result.found_ground_truth_cost,
                    expected.optimal_path_cost,
                    delta=1e-12,
                )
                self.assertAlmostEqual(
                    zero_result.found_ground_truth_cost,
                    expected.optimal_path_cost,
                    delta=1e-12,
                )
                self.assert_valid_path(generated.grid, result.path or ())

    def test_chebyshev_heuristic_is_admissible_and_consistent(self) -> None:
        grid = np.asarray(
            [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 1],
                [1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=np.uint8,
        )
        goal = (3, 4)
        dijkstra_result = dijkstra(grid, goal)
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                node = (row, col)
                if grid[node] != 0 or not dijkstra_result.reachable_mask[node]:
                    continue
                h_node = chebyshev_heuristic(node, goal)
                self.assertLessEqual(h_node, dijkstra_result.distances[node] + 1e-12)
                for neighbor in get_neighbors(grid, node):
                    h_neighbor = chebyshev_heuristic(neighbor, goal)
                    edge_cost = get_transition_cost(grid, node, neighbor)
                    self.assertLessEqual(h_node, edge_cost + h_neighbor + 1e-12)

    def test_path_cost_matches_shared_transition_cost(self) -> None:
        grid = np.zeros((7, 8), dtype=np.uint8)
        grid[2:6, 3] = 1
        result = astar(grid, (6, 0), (0, 7))
        self.assertTrue(result.found)
        path = result.path or ()
        self.assertEqual(path[0], (6, 0))
        self.assertEqual(path[-1], (0, 7))
        self.assert_valid_path(grid, path)
        path_cost = sum(
            get_transition_cost(grid, source, destination, DEFAULT_COST_CONFIG)
            for source, destination in zip(path, path[1:])
        )
        self.assertEqual(path_cost, result.found_ground_truth_cost)

    def test_unreachable_goal_returns_no_path(self) -> None:
        grid = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        result = astar(grid, (0, 0), (0, 2))
        self.assertFalse(result.found)
        self.assertIsNone(result.path)
        self.assertTrue(np.isinf(result.found_ground_truth_cost))
        self.assertEqual(result.final_open_size, 0)

    def test_start_equal_goal_returns_zero_cost(self) -> None:
        grid = np.zeros((2, 2), dtype=np.uint8)
        result = astar(grid, (1, 1), (1, 1))
        self.assertTrue(result.found)
        self.assertEqual(result.path, ((1, 1),))
        self.assertEqual(result.found_ground_truth_cost, 0.0)
        self.assertEqual(result.expanded_nodes, 0)

    def test_custom_heuristic_callable_is_supported(self) -> None:
        grid = np.zeros((4, 4), dtype=np.uint8)
        predictions = np.arange(16, dtype=np.float64).reshape((4, 4)) / 10.0
        predictor = lambda node, goal: predictions[node]  # noqa: E731
        result = astar(grid, (0, 0), (3, 3), heuristic=predictor)
        self.assertTrue(result.found)
        self.assertEqual(result.heuristic_name, "<lambda>")

    def test_non_finite_heuristic_is_rejected(self) -> None:
        grid = np.zeros((2, 2), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            astar(grid, (0, 0), (1, 1), heuristic=lambda node, goal: np.nan)

    def test_search_metrics_are_populated(self) -> None:
        grid = np.zeros((8, 8), dtype=np.uint8)
        result = astar(grid, (0, 0), (7, 7))
        self.assertGreater(result.generated_nodes, 0)
        self.assertGreaterEqual(result.expanded_nodes, 0)
        self.assertEqual(result.reexpansions, 0)
        self.assertGreaterEqual(result.max_open_size, result.final_open_size)
        self.assertGreaterEqual(result.final_open_size, 0)
        self.assertTrue(result.found)
        # The valid OPEN removal for the goal is counted as expanded, but the
        # search returns before adding the goal to CLOSED.
        self.assertEqual(result.expanded_nodes, result.closed_size + 1)
        self.assertGreaterEqual(result.search_time_seconds, 0.0)

    def test_better_paths_reopen_closed_states_and_update_parents(self) -> None:
        grid = np.zeros((6, 6), dtype=np.uint8)
        scores = np.asarray(
            [
                [75, 50, 54, 71, 46, 62],
                [66, 18, 4, 24, 22, 69],
                [73, 0, 39, 65, 10, 63],
                [9, 37, 65, 24, 27, 22],
                [57, 20, 79, 35, 38, 40],
                [46, 44, 40, 79, 64, 10000],
            ],
            dtype=np.float64,
        )
        # Deliberately inconsistent values exercise CLOSED reopening; this
        # does not certify this arbitrary heuristic as admissible.
        heuristic = lambda node, goal: scores[node]  # noqa: E731
        result = astar(grid, (0, 0), (5, 5), heuristic=heuristic)
        self.assertTrue(result.found)
        self.assertGreater(result.reexpansions, 0)
        self.assertAlmostEqual(result.found_ground_truth_cost, 5.0, delta=1e-12)
        self.assert_valid_path(grid, result.path or ())

    @staticmethod
    def assert_valid_path(grid: np.ndarray, path: tuple[tuple[int, int], ...]) -> None:
        if not path:
            raise AssertionError("Expected a non-empty path.")
        for source, destination in zip(path, path[1:]):
            if destination not in get_neighbors(grid, source):
                raise AssertionError(f"Invalid path transition {source} -> {destination}.")


if __name__ == "__main__":
    unittest.main()
