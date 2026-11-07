"""Synthetic correctness tests for uniform Neural-guidance SeeA*."""

from __future__ import annotations

import unittest

import numpy as np

from algorithms.movement import DEFAULT_COST_CONFIG, get_neighbors, is_legal_transition
from seeastar.neural_guidance import (
    GuidanceSeeAStarConfig,
    neural_guidance_heuristic,
    run_guidance_uniform_seeastar,
    run_neural_guidance_seeastar,
)
from seeastar.sampling import UniformSampler


class GuidanceAccumulationTests(unittest.TestCase):
    def test_64x64_map_with_overlapping_rectangles_searches_legally(self) -> None:
        grid = np.zeros((64, 64), dtype=np.uint8)
        first = (slice(20, 34), slice(18, 34))
        second = (slice(28, 42), slice(28, 44))
        grid[first] = 1
        grid[second] = 1
        overlap = np.zeros_like(grid)
        overlap[first] += 1
        overlap[second] += 1
        self.assertGreater(int(np.count_nonzero(overlap > 1)), 0)

        phi = np.full((64, 64), 0.05, dtype=np.float64)
        result = run_neural_guidance_seeastar(
            grid, (4, 4), (59, 59), phi, k=16, seed=23
        )
        self.assertTrue(result.found)
        self.assertEqual(result.guidance_map.shape, (64, 64))
        self.assertEqual(result.found_ground_truth_cost, len(result.path) - 1)
        self.assertAlmostEqual(result.found_guidance_cost, 0.05 * (len(result.path) - 1))
        for source, destination in zip(result.path, result.path[1:]):
            self.assertIn(destination, get_neighbors(grid, source, DEFAULT_COST_CONFIG))

    def test_start_is_zero_start_phi_is_not_charged_successor_phi_is(self) -> None:
        grid = np.zeros((1, 2), dtype=np.uint8)
        phi = np.asarray([[99.0, 3.5]])
        result = run_neural_guidance_seeastar(
            grid, (0, 0), (0, 1), phi, k=4, seed=7
        )
        self.assertEqual(result.accumulated_guidance_cost_map[0, 0], 0.0)
        self.assertEqual(result.found_guidance_cost, 3.5)
        self.assertEqual(result.found_ground_truth_cost, 1.0)

    def test_multi_step_guidance_cost_accumulates_destination_values(self) -> None:
        grid = np.zeros((1, 4), dtype=np.uint8)
        phi = np.asarray([[12.0, 0.25, 1.5, 2.25]])
        result = run_neural_guidance_seeastar(
            grid, (0, 0), (0, 3), phi, k=8
        )
        self.assertEqual(result.path, ((0, 0), (0, 1), (0, 2), (0, 3)))
        np.testing.assert_allclose(
            result.accumulated_guidance_cost_map[0], [0.0, 0.25, 1.75, 4.0]
        )
        self.assertEqual(result.found_guidance_cost, 4.0)
        self.assertEqual(result.found_ground_truth_cost, 3.0)

    def test_open_node_strict_improvement_updates_guidance_parent_and_priority(self) -> None:
        grid = np.zeros((4, 5), dtype=np.uint8)
        phi = np.asarray(
            [
                [4, 3, 2, 1, 1],
                [0, 0, 0, 0, 4],
                [3, 4, 2, 3, 4],
                [3, 3, 2, 2, 4],
            ],
            dtype=np.float64,
        )
        result = run_neural_guidance_seeastar(
            grid, (0, 0), (3, 4), phi, k=50, capture_trace=True
        )
        updates = [
            item
            for item in result.update_trace
            if item.state == "open" and item.updated and item.candidate_g < item.previous_g
        ]
        self.assertTrue(updates)
        update = updates[0]
        self.assertEqual(update.source, (2, 2))
        self.assertEqual(update.destination, (3, 2))
        self.assertEqual(update.previous_g, 5.0)
        self.assertEqual(update.candidate_g, 4.0)
        self.assertEqual(result.parent_map[update.destination], update.source)
        self.assertEqual(result.accumulated_guidance_cost_map[update.destination], 4.0)
        self.assertEqual(
            result.priority_map[update.destination],
            result.accumulated_guidance_cost_map[update.destination]
            + result.heuristic_map[update.destination],
        )

    def test_equal_cost_alternative_preserves_existing_parent(self) -> None:
        grid = np.zeros((4, 5), dtype=np.uint8)
        phi = np.asarray(
            [
                [4, 3, 2, 1, 1],
                [0, 0, 0, 0, 4],
                [3, 4, 2, 3, 4],
                [3, 3, 2, 2, 4],
            ],
            dtype=np.float64,
        )
        result = run_neural_guidance_seeastar(
            grid, (0, 0), (3, 4), phi, k=50, capture_trace=True
        )
        equal = [
            item
            for item in result.update_trace
            if item.state == "open"
            and not item.updated
            and item.candidate_g == item.previous_g
            and item.previous_parent is not None
            and item.source != item.previous_parent
        ]
        self.assertTrue(equal)
        self.assertEqual(equal[0].destination, (0, 1))
        self.assertEqual(equal[0].previous_parent, (0, 0))
        self.assertEqual(result.parent_map[equal[0].destination], equal[0].previous_parent)

    def test_closed_nodes_are_excluded_and_never_reopened(self) -> None:
        grid = np.zeros((4, 5), dtype=np.uint8)
        result = run_neural_guidance_seeastar(
            grid, (0, 0), (3, 4), np.ones(grid.shape), k=20, capture_trace=True
        )
        excluded = [item for item in result.update_trace if item.state == "closed_excluded"]
        self.assertTrue(excluded)
        self.assertTrue(all(not item.updated for item in excluded))
        self.assertEqual(result.reexpansions, 0)
        self.assertEqual(len(result.expanded_order), len(set(result.expanded_order)))
        self.assertTrue(all(result.closed_mask[node] for node in result.expanded_order))

    def test_chebyshev_plus_euclidean_heuristic(self) -> None:
        self.assertAlmostEqual(
            neural_guidance_heuristic((1, 2), (4, 6)), 4.0 + 0.001 * 5.0
        )
        self.assertAlmostEqual(
            neural_guidance_heuristic((2, 2), (3, 3)),
            1.0 + 0.001 * np.sqrt(2.0),
        )
        result = run_neural_guidance_seeastar(
            np.zeros((4, 6)), (0, 0), (3, 5), np.zeros((4, 6))
        )
        self.assertAlmostEqual(result.heuristic_map[1, 1], 4.0 + 0.001 * np.sqrt(20.0))


class GuidanceValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = np.zeros((2, 3), dtype=np.uint8)

    def test_phi_shape_must_be_two_dimensional_and_match_map(self) -> None:
        for phi in (np.zeros((6,)), np.zeros((1, 2, 3)), np.zeros((3, 2))):
            with self.subTest(shape=phi.shape), self.assertRaises(ValueError):
                run_neural_guidance_seeastar(self.grid, (0, 0), (1, 2), phi)

    def test_phi_values_must_be_finite(self) -> None:
        for value in (np.nan, np.inf, -np.inf):
            phi = np.zeros(self.grid.shape)
            phi[0, 1] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                run_neural_guidance_seeastar(self.grid, (0, 0), (1, 2), phi)

    def test_phi_values_must_be_nonnegative(self) -> None:
        phi = np.zeros(self.grid.shape)
        phi[1, 1] = -0.001
        with self.assertRaises(ValueError):
            run_neural_guidance_seeastar(self.grid, (0, 0), (1, 2), phi)

    def test_phi_values_above_one_are_preserved_without_clipping(self) -> None:
        phi = np.asarray([[4.0, 2.5, 1.5], [7.0, 3.0, 6.0]])
        result = run_neural_guidance_seeastar(
            self.grid, (0, 0), (1, 2), phi, k=8
        )
        np.testing.assert_array_equal(result.guidance_map, phi)
        self.assertTrue(result.found)
        self.assertGreater(result.found_guidance_cost, 1.0)


class UniformSamplingTests(unittest.TestCase):
    def test_draw_is_a_duplicate_free_subset_of_eligible_open(self) -> None:
        grid = np.zeros((5, 5), dtype=np.uint8)
        result = run_neural_guidance_seeastar(
            grid,
            (2, 2),
            (0, 0),
            np.zeros(grid.shape),
            k=3,
            seed=37,
            capture_trace=True,
        )
        self.assertTrue(result.selection_trace)
        for item in result.selection_trace:
            self.assertEqual(len(item.sampled_candidates), len(set(item.sampled_candidates)))
            self.assertTrue(set(item.sampled_candidates).issubset(item.eligible_open))
            self.assertEqual(len(item.sampled_candidates), min(3, len(item.eligible_open)))

    def test_uniform_draw_is_independent_of_f_values(self) -> None:
        nodes = [(row, column) for row in range(3) for column in range(4)]
        low_first = {node: float(index) for index, node in enumerate(nodes)}
        high_first = {node: float(-index) for index, node in enumerate(nodes)}
        depths = {node: 0 for node in nodes}
        first = UniformSampler(5, seed=223).select(nodes, low_first, depths)
        second = UniformSampler(5, seed=223).select(nodes, high_first, depths)
        self.assertEqual(first, second)

    def test_minimum_current_f_is_selected_within_sample_and_expanded(self) -> None:
        grid = np.zeros((5, 6), dtype=np.uint8)
        result = run_neural_guidance_seeastar(
            grid,
            (0, 0),
            (4, 5),
            np.linspace(0.0, 2.0, grid.size).reshape(grid.shape),
            k=3,
            seed=41,
            capture_trace=True,
        )
        non_goal_traces = [item for item in result.selection_trace if item.expanded is not None]
        self.assertTrue(non_goal_traces)
        for item in non_goal_traces:
            expected = min(item.sampled_f_values, key=lambda pair: (pair[1], pair[0]))[0]
            self.assertEqual(item.selected, expected)
            self.assertEqual(item.expanded, item.selected)
        self.assertEqual(
            tuple(item.expanded for item in non_goal_traces), result.expanded_order
        )

    def test_configured_k_is_preserved_and_effective_k_uses_open_size(self) -> None:
        grid = np.zeros((3, 4), dtype=np.uint8)
        result = run_neural_guidance_seeastar(
            grid, (0, 0), (2, 3), np.zeros(grid.shape), k=17, seed=3
        )
        self.assertEqual(result.configured_k, 17)
        self.assertTrue(any(size < result.configured_k for size in result.eligible_open_sizes))
        self.assertEqual(
            result.candidate_sizes,
            tuple(min(result.configured_k, size) for size in result.eligible_open_sizes),
        )

    def test_sampling_and_expansion_are_reproducible_for_same_seed(self) -> None:
        grid = np.zeros((6, 7), dtype=np.uint8)
        phi = np.arange(grid.size, dtype=np.float64).reshape(grid.shape) / 7.0
        first = run_neural_guidance_seeastar(
            grid, (0, 0), (5, 6), phi, k=4, seed=201, capture_trace=True
        )
        second = run_neural_guidance_seeastar(
            grid, (0, 0), (5, 6), phi, k=4, seed=201, capture_trace=True
        )
        self.assertEqual(first.selection_trace, second.selection_trace)
        self.assertEqual(first.expanded_order, second.expanded_order)
        self.assertEqual(first.path, second.path)

    def test_same_open_population_gets_same_draw_when_f_values_change(self) -> None:
        grid = np.zeros((4, 5), dtype=np.uint8)
        phi_a = np.zeros(grid.shape)
        phi_b = np.arange(grid.size, dtype=np.float64).reshape(grid.shape) + 10.0
        first = run_neural_guidance_seeastar(
            grid, (1, 2), (0, 0), phi_a, k=2, seed=77, capture_trace=True
        )
        second = run_neural_guidance_seeastar(
            grid, (1, 2), (0, 0), phi_b, k=2, seed=77, capture_trace=True
        )
        self.assertEqual(first.selection_trace[1].eligible_open, second.selection_trace[1].eligible_open)
        self.assertEqual(first.selection_trace[1].sampled_candidates, second.selection_trace[1].sampled_candidates)


class PathReportingTests(unittest.TestCase):
    def test_path_reconstructs_legal_transitions_and_reports_both_costs(self) -> None:
        grid = np.asarray(
            [[0, 0, 0, 1], [0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.uint8
        )
        phi = np.asarray(
            [[8.0, 0.5, 3.0, 100.0], [0.2, 1.25, 0.75, 4.0], [9.0, 2.0, 0.4, 1.5]]
        )
        result = run_neural_guidance_seeastar(
            grid, (0, 0), (2, 3), phi, k=5, seed=83
        )
        self.assertTrue(result.found)
        self.assertEqual(result.path[0], (0, 0))
        self.assertEqual(result.path[-1], (2, 3))
        self.assertTrue(
            all(
                is_legal_transition(grid, source, destination, DEFAULT_COST_CONFIG)
                for source, destination in zip(result.path, result.path[1:])
            )
        )
        self.assertEqual(
            result.found_ground_truth_cost,
            sum(
                1.0
                for source, destination in zip(result.path, result.path[1:])
                if destination in get_neighbors(grid, source)
            ),
        )
        self.assertEqual(
            result.found_guidance_cost,
            sum(float(phi[node]) for node in result.path[1:]),
        )
        self.assertNotEqual(result.found_ground_truth_cost, result.found_guidance_cost)


class ConfigApiTests(unittest.TestCase):
    def test_corrected_api_is_exported_from_the_packages(self) -> None:
        from adaptive import run_ood_adaptive_neural_guidance_seeastar
        from seeastar import run_neural_guidance_seeastar as exported_fixed_api

        self.assertIs(exported_fixed_api, run_neural_guidance_seeastar)
        self.assertTrue(callable(run_ood_adaptive_neural_guidance_seeastar))

    def test_config_and_low_level_entry_point_use_the_shared_core(self) -> None:
        grid = np.zeros((1, 3), dtype=np.uint8)
        phi = np.asarray([[5.0, 2.0, 3.0]])
        low_level = run_guidance_uniform_seeastar(
            grid,
            (0, 0),
            (0, 2),
            phi,
            GuidanceSeeAStarConfig(k=4, seed=12),
        )
        public = run_neural_guidance_seeastar(
            grid, (0, 0), (0, 2), phi, k=4, seed=12
        )
        self.assertEqual(low_level.path, public.path)
        self.assertEqual(low_level.expanded_order, public.expanded_order)
        self.assertEqual(low_level.found_guidance_cost, public.found_guidance_cost)


if __name__ == "__main__":
    unittest.main()
