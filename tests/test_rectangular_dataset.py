"""Focused checks for the isolated 64x64 rectangular dataset generator."""

from __future__ import annotations

import unittest

import numpy as np

from algorithms.movement import DEFAULT_COST_CONFIG, get_neighbors
from data.ground_truth import compute_ground_truth
from data.rectangular_dataset import (
    BASE_SEED,
    DENSITY_INTERVALS_PERCENT,
    MAP_SHAPE,
    RectangleGenerationConfig,
    CandidateRejected,
    canonical_inputs,
    generate_candidate,
    sample_seed_for,
    split_group_seed,
)


class RectangularDatasetTests(unittest.TestCase):
    def test_candidate_is_reproducible_and_has_shared_rectangular_geometry(self) -> None:
        config = RectangleGenerationConfig()
        for distribution in DENSITY_INTERVALS_PERCENT:
            split = "training" if distribution in {"D1", "D2", "D3"} else "ood_pilot"
            seed = sample_seed_for(split_group_seed(BASE_SEED, split, distribution), 0)
            first = second = None
            for attempt_index in range(100):
                try:
                    first = generate_candidate(distribution, seed, attempt_index, config)
                    second = generate_candidate(distribution, seed, attempt_index, config)
                    break
                except CandidateRejected:
                    continue
            self.assertIsNotNone(first, distribution)
            self.assertIsNotNone(second, distribution)
            assert first is not None and second is not None
            self.assertEqual(first.layout_hash, second.layout_hash)
            self.assertTrue(np.array_equal(first.grid, second.grid))
            self.assertEqual(first.start, second.start)
            self.assertEqual(first.goal, second.goal)
            self.assertEqual(first.grid.shape, MAP_SHAPE)
            low, high = DENSITY_INTERVALS_PERCENT[distribution]
            self.assertGreaterEqual(first.actual_density * 100, low)
            self.assertLessEqual(first.actual_density * 100, high)
            self.assertGreaterEqual(first.start_goal_separation, 24)
            for top, left, height, width in first.rectangles:
                self.assertGreaterEqual(height, 2)
                self.assertLessEqual(height, 14)
                self.assertGreaterEqual(width, 2)
                self.assertLessEqual(width, 14)
                self.assertGreaterEqual(top, 0)
                self.assertGreaterEqual(left, 0)
                self.assertLessEqual(top + height, 64)
                self.assertLessEqual(left + width, 64)

    def test_endpoint_and_label_generation_use_the_shared_legal_neighbors(self) -> None:
        config = RectangleGenerationConfig()
        seed = sample_seed_for(split_group_seed(BASE_SEED, "training", "D1"), 1)
        candidate = None
        for attempt_index in range(100):
            try:
                candidate = generate_candidate("D1", seed, attempt_index, config)
                break
            except CandidateRejected:
                continue
        self.assertIsNotNone(candidate)
        assert candidate is not None
        truth = compute_ground_truth(
            candidate.grid, candidate.start, candidate.goal, DEFAULT_COST_CONFIG
        )
        self.assertIsNotNone(truth.optimal_path)
        self.assertTrue(truth.reachable_mask[candidate.start])
        self.assertTrue(np.isinf(truth.distance_map[~truth.reachable_mask]).all())
        self.assertEqual(truth.distance_map.dtype, np.float64)
        for source, destination in zip(truth.optimal_path, truth.optimal_path[1:]):
            self.assertIn(destination, get_neighbors(candidate.grid, source))

    def test_three_logical_channels_have_exact_order_and_one_hot_endpoints(self) -> None:
        grid = np.zeros((64, 64), dtype=np.uint8)
        grid[5:9, 6:11] = 1
        channels = canonical_inputs(grid, (0, 0), (63, 63))
        self.assertEqual(channels.shape, (3, 64, 64))
        self.assertTrue(np.array_equal(channels[0], grid))
        self.assertEqual(int(channels[1].sum()), 1)
        self.assertEqual(int(channels[2].sum()), 1)
        self.assertEqual(channels[1, 0, 0], 1)
        self.assertEqual(channels[2, 63, 63], 1)


if __name__ == "__main__":
    unittest.main()
