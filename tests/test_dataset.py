from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data.dataset import (
    assert_no_map_leakage,
    generate_standard_splits,
    load_dataset_npz,
    map_fingerprint,
    save_dataset_npz,
)
from data.map_generator import MapGenerationConfig


class DatasetTests(unittest.TestCase):
    def test_standard_splits_are_independent_and_distribution_safe(self) -> None:
        splits = generate_standard_splits(
            3,
            3,
            3,
            base_seed=20261106,
            map_config=MapGenerationConfig(height=16, width=16),
        )
        self.assertEqual(
            [sample.grid_map.distribution for sample in splits["train"]],
            ["D1", "D2", "D3"],
        )
        self.assertEqual(
            [sample.grid_map.distribution for sample in splits["validation"]],
            ["D1", "D2", "D3"],
        )
        self.assertEqual(
            [sample.grid_map.distribution for sample in splits["test"]],
            ["D4", "D5", "D6"],
        )
        assert_no_map_leakage(splits)
        train_hashes = {map_fingerprint(sample.grid_map.grid) for sample in splits["train"]}
        validation_hashes = {
            map_fingerprint(sample.grid_map.grid) for sample in splits["validation"]
        }
        test_hashes = {map_fingerprint(sample.grid_map.grid) for sample in splits["test"]}
        self.assertFalse(train_hashes & validation_hashes)
        self.assertFalse(train_hashes & test_hashes)
        self.assertFalse(validation_hashes & test_hashes)

    def test_standard_splits_are_reproducible_from_base_seed(self) -> None:
        kwargs = dict(
            train_count=2,
            validation_count=2,
            ood_test_count=2,
            base_seed=413,
            map_config=MapGenerationConfig(height=16, width=16),
        )
        first = generate_standard_splits(**kwargs)
        second = generate_standard_splits(**kwargs)
        for split_name in first:
            for sample_a, sample_b in zip(first[split_name], second[split_name]):
                np.testing.assert_array_equal(sample_a.grid_map.grid, sample_b.grid_map.grid)
                self.assertEqual(sample_a.grid_map.start, sample_b.grid_map.start)
                self.assertEqual(sample_a.grid_map.goal, sample_b.grid_map.goal)
                self.assertEqual(sample_a.grid_map.seed, sample_b.grid_map.seed)

    def test_leakage_check_rejects_same_map_with_new_endpoints(self) -> None:
        splits = generate_standard_splits(
            1,
            1,
            0,
            base_seed=85,
            map_config=MapGenerationConfig(height=12, width=12),
        )
        first = splits["train"][0]
        other = splits["validation"][0]
        duplicated_map = replace(
            other.grid_map,
            grid=first.grid_map.grid,
            start=first.grid_map.start,
            goal=first.grid_map.goal,
        )
        duplicated_sample = replace(other, grid_map=duplicated_map)
        with self.assertRaisesRegex(ValueError, "Duplicate map layout"):
            assert_no_map_leakage({"train": [first], "validation": [duplicated_sample]})

    def test_npz_round_trip_preserves_required_fields(self) -> None:
        splits = generate_standard_splits(
            1,
            0,
            0,
            base_seed=86,
            map_config=MapGenerationConfig(height=12, width=14),
        )
        sample = splits["train"][0]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            output = Path(directory) / "train.npz"
            save_dataset_npz(output, [sample])
            loaded = load_dataset_npz(output)

        np.testing.assert_array_equal(loaded["maps"][0], sample.grid_map.grid)
        np.testing.assert_array_equal(loaded["starts"][0], sample.grid_map.start)
        np.testing.assert_array_equal(loaded["goals"][0], sample.grid_map.goal)
        np.testing.assert_array_equal(loaded["distance_maps"][0], sample.ground_truth.distance_map)
        np.testing.assert_array_equal(
            loaded["reachability_masks"][0], sample.ground_truth.reachable_mask
        )
        self.assertEqual(float(loaded["optimal_path_costs"][0]), sample.ground_truth.optimal_path_cost)
        self.assertEqual(loaded["distribution_labels"][0], sample.grid_map.distribution)
        self.assertEqual(int(loaded["seeds"][0]), sample.grid_map.seed)
        self.assertEqual(loaded["cost_model_config"]["model"], "unit_move_count")
        self.assertEqual(loaded["generation_configs"][0]["distribution"], "D1")
        self.assertEqual(int(loaded["schema_version"]), 1)


if __name__ == "__main__":
    unittest.main()
