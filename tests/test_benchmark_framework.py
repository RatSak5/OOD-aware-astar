"""Tests for Prompt 8 data preparation, statistics, and passive instrumentation."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmarks.config import BenchmarkConfig
from benchmarks.data import (
    SPLIT_FILENAMES,
    _map_structure_metrics,
    load_dataset_samples_npz,
    load_evaluation_splits,
    load_training_splits,
    prepare_datasets,
)
from benchmarks.neural_trace import run_instrumented_neural_astar
from benchmarks.reliability import heuristic_error_metrics, spearman_rank_correlation
from benchmarks.runner import (
    _all_search_arms,
    _result_counters,
    _search_row,
    _task_for_arm,
    expected_search_count,
    experiment_arms,
)
from benchmarks.runner import _result_counters
from benchmarks.search_worker import run_search_task
from benchmarks.statistics import (
    average_seeds_by_map,
    holm_adjust,
    paired_map_differences,
    paired_rank_biserial,
    paired_sign_flip_test,
)
from data.dataset import make_dataset_sample
from data.map_generator import MapGenerationConfig
from algorithms.movement import get_neighbors


class BenchmarkConfigTests(unittest.TestCase):
    def test_approved_configuration_includes_k_five_and_seed_pairs(self) -> None:
        config = BenchmarkConfig()
        self.assertEqual(
            (config.train_count, config.validation_count, config.id_holdout_count, config.ood_test_count),
            (300, 90, 90, 90),
        )
        self.assertEqual((config.height, config.width), (32, 32))
        self.assertEqual(config.base_seed, 20261106)
        self.assertEqual(config.sampling_seeds, (11, 23, 47, 89, 131))
        self.assertEqual(config.primary_fixed_ks, (4, 5, 10, 300))
        guidance = config.to_dict()["neural_guidance_search"]
        self.assertEqual(guidance["fixed_method_identifier"], "neural_guidance_fixed")
        self.assertEqual(guidance["adaptive_method_identifier"], "neural_guidance_adaptive")
        self.assertEqual(guidance["fixed_k_values"], [4, 5, 10, 300])
        self.assertEqual(guidance["adaptive_k_bounds"], [4, 300])

    def test_configuration_round_trips_without_manual_edits(self) -> None:
        config = BenchmarkConfig()
        with tempfile.TemporaryDirectory() as directory:
            path = config.save(Path(directory) / "benchmark.json")
            restored = BenchmarkConfig.load(path)
        self.assertEqual(restored, config)


class BenchmarkDatasetTests(unittest.TestCase):
    def test_vectorized_structure_stats_match_shared_legal_neighbors(self) -> None:
        sample = make_dataset_sample(
            MapGenerationConfig(height=12, width=13, distribution="D5", seed=5544)
        )
        metrics = _map_structure_metrics(sample)
        degrees = [
            len(get_neighbors(sample.grid_map.grid, (int(row), int(column)), sample.cost_config))
            for row, column in np.argwhere(sample.grid_map.grid == 0)
        ]
        self.assertAlmostEqual(metrics["free_legal_degree_mean"], float(np.mean(degrees)))
        self.assertAlmostEqual(
            metrics["free_cells_degree_le_3_fraction"],
            float(np.mean(np.asarray(degrees) <= 3)),
        )
        self.assertGreater(metrics["full_span_corridor_fraction"], 0.0)

    def test_generation_round_trip_leakage_statistics_and_ood_gate(self) -> None:
        config = replace(
            BenchmarkConfig(),
            height=10,
            width=10,
            train_count=6,
            validation_count=6,
            id_holdout_count=6,
            ood_test_count=6,
            base_seed=17001,
            output_directory="unused-in-this-test",
        )
        with tempfile.TemporaryDirectory() as directory:
            generated = prepare_datasets(config, directory)
            manifest = generated["manifest"]
            self.assertEqual(manifest["split_counts"], {
                "train": 6,
                "validation": 6,
                "id_holdout": 6,
                "ood_test": 6,
            })
            self.assertTrue(manifest["split_fingerprints_disjoint"])
            self.assertEqual(manifest["unique_obstacle_layouts"], 24)
            for split in ("train", "validation", "id_holdout"):
                counts = manifest["distribution_counts"][split]
                self.assertEqual([counts[name] for name in ("D1", "D2", "D3")], [2, 2, 2])
                self.assertEqual([counts[name] for name in ("D4", "D5", "D6")], [0, 0, 0])
            ood_counts = manifest["distribution_counts"]["ood_test"]
            self.assertEqual([ood_counts[name] for name in ("D1", "D2", "D3")], [0, 0, 0])
            self.assertEqual([ood_counts[name] for name in ("D4", "D5", "D6")], [2, 2, 2])
            self.assertEqual(set(manifest["distribution_statistics"]["by_distribution"]),
                             {"D1", "D2", "D3", "D4", "D5", "D6"})

            train, validation = load_training_splits(directory)
            self.assertEqual(len(train), 6)
            self.assertEqual(len(validation), 6)
            with self.assertRaises(PermissionError):
                load_evaluation_splits(
                    directory,
                    model_ready_manifest=Path(directory) / "model_ready_manifest.json",
                )
            ready = Path(directory) / "model_ready_manifest.json"
            ready.write_text(
                json.dumps({"training_complete": True, "d95_calibrated": True}),
                encoding="utf-8",
            )
            id_holdout, ood_test = load_evaluation_splits(
                directory,
                model_ready_manifest=ready,
            )
            self.assertEqual(len(id_holdout), 6)
            self.assertEqual(len(ood_test), 6)
            self.assertEqual(Path(directory, SPLIT_FILENAMES["ood_test"]).name, "ood_test_sealed.npz")

    def test_npz_loader_preserves_distance_precision_and_path(self) -> None:
        sample = make_dataset_sample(
            MapGenerationConfig(height=10, width=11, distribution="D2", seed=9981)
        )
        from data.dataset import save_dataset_npz

        with tempfile.TemporaryDirectory() as directory:
            path = save_dataset_npz(Path(directory) / "single.npz", [sample])
            (restored,) = load_dataset_samples_npz(path)
        self.assertEqual(restored.grid_map.start, sample.grid_map.start)
        self.assertEqual(restored.grid_map.goal, sample.grid_map.goal)
        self.assertEqual(restored.ground_truth.optimal_path, sample.ground_truth.optimal_path)
        self.assertEqual(restored.ground_truth.distance_map.dtype, np.float64)
        np.testing.assert_array_equal(restored.ground_truth.distance_map, sample.ground_truth.distance_map)


class BenchmarkStatisticsTests(unittest.TestCase):
    def test_seed_averages_before_paired_map_inference(self) -> None:
        rows = [
            {"map_id": "a", "method": "fixed", "metric": 8.0},
            {"map_id": "a", "method": "fixed", "metric": 10.0},
            {"map_id": "a", "method": "adaptive", "metric": 6.0},
            {"map_id": "a", "method": "adaptive", "metric": 8.0},
            {"map_id": "b", "method": "fixed", "metric": 4.0},
            {"map_id": "b", "method": "adaptive", "metric": 3.0},
        ]
        self.assertEqual(average_seeds_by_map(rows, metric="metric", method="fixed"), {"a": 9.0, "b": 4.0})
        map_ids, differences = paired_map_differences(
            rows, metric="metric", method_a="adaptive", method_b="fixed"
        )
        self.assertEqual(map_ids, ["a", "b"])
        np.testing.assert_array_equal(differences, [-2.0, -1.0])
        self.assertEqual(paired_rank_biserial(differences), -1.0)
        self.assertLessEqual(paired_sign_flip_test(differences, repetitions=1000, seed=3), 1.0)
        self.assertEqual(holm_adjust([0.01, 0.03, 0.04]), [0.03, 0.06, 0.06])
        self.assertAlmostEqual(spearman_rank_correlation([1, 2, 3], [3, 2, 1]), -1.0)

    def test_predeclared_search_orchestration_count_and_uniform_adaptive_separation(self) -> None:
        config = BenchmarkConfig()
        arms = experiment_arms(config)
        self.assertEqual(len(arms), 37)
        conventional_adaptive = [
            arm for arm in arms if str(arm["method_family"]).startswith("adaptive")
        ]
        self.assertTrue(conventional_adaptive)
        self.assertTrue(all(arm["search_implementation"] == "uniform_fixed" for arm in conventional_adaptive))
        self.assertFalse(any("cluster" in arm["method"] or "uct" in arm["method"] for arm in conventional_adaptive))
        primary_h1 = [arm for arm in arms if arm["heuristic"] == "H1" and arm["k_min"] == 4 and arm["k_max"] == 300]
        fixed_ks = {arm["configured_k"] for arm in primary_h1 if arm["method_family"] == "fixed_k_uniform"}
        self.assertEqual(fixed_ks, {4, 5, 10, 300})
        self.assertEqual(expected_search_count(config, full=True)["total_searches"], 34_560)
        self.assertEqual(expected_search_count(config, full=False)["total_searches"], 1_458)

    def test_raw_heuristic_error_uses_valid_edges_and_unit_costs(self) -> None:
        sample = make_dataset_sample(
            MapGenerationConfig(height=8, width=9, distribution="D1", seed=8877)
        )
        perfect = sample.ground_truth.distance_map.copy()
        result = heuristic_error_metrics(sample, perfect)
        self.assertEqual(result["mae"], 0.0)
        self.assertEqual(result["rmse"], 0.0)
        self.assertEqual(result["overestimation_rate"], 0.0)
        self.assertEqual(result["consistency_violation_rate"], 0.0)
        high = perfect.copy()
        high[sample.grid_map.start] += 10.0
        degraded = heuristic_error_metrics(sample, high)
        self.assertGreater(degraded["maximum_overestimation_magnitude"], 0.0)
        self.assertGreater(degraded["consistency_violation_rate"], 0.0)


class NeuralAStarInstrumentationTests(unittest.TestCase):
    def test_passive_counters_preserve_search_outputs(self) -> None:
        grid = np.zeros((7, 9), dtype=np.uint8)
        grid[2, 3:7] = 1
        grid[2, 5] = 0
        phi = np.linspace(0.05, 0.95, grid.size, dtype=np.float64).reshape(grid.shape)
        result, trace = run_instrumented_neural_astar(
            grid, (0, 0), (6, 8), phi
        )
        self.assertTrue(result.found)
        self.assertTrue(result.path)
        self.assertGreaterEqual(trace.generated_nodes, len(result.selected_order))
        self.assertGreaterEqual(trace.maximum_open_size, 1)
        self.assertEqual(len(trace.open_sizes), len(result.selected_order))
        self.assertGreaterEqual(trace.maximum_open_size, result.final_open_size)


class BenchmarkSearchDispatchTests(unittest.TestCase):
    def test_workers_use_shared_costs_and_keep_neural_guidance_distinct(self) -> None:
        sample = make_dataset_sample(
            MapGenerationConfig(height=10, width=11, distribution="D1", seed=38122)
        )
        grid = sample.grid_map.grid
        heuristic = np.where(
            np.isfinite(sample.ground_truth.distance_map),
            sample.ground_truth.distance_map,
            0.0,
        )
        base = {
            "grid": grid,
            "start": sample.grid_map.start,
            "goal": sample.grid_map.goal,
            "heuristic_map": heuristic,
            "seed": 23,
            "k": 4,
        }
        for method in ("dijkstra", "astar_zero", "astar_chebyshev", "astar_H1"):
            response = run_search_task({**base, "method": method})
            self.assertEqual(response["status"], "completed")
            self.assertTrue(response["found"])
            self.assertAlmostEqual(
                response["found_ground_truth_cost"], sample.ground_truth.optimal_path_cost
            )

        uniform = run_search_task({**base, "method": "uniform_fixed", "k": 4})
        self.assertTrue(uniform["found"])
        counters = _result_counters({**uniform, "configured_k": 4})
        self.assertGreater(counters["sampling_operations"], 0)
        self.assertLessEqual(counters["maximum_effective_k"], 4)
        self.assertTrue(all(size <= eligible for size, eligible in zip(
            json.loads(counters["candidate_sizes_json"]),
            json.loads(counters["eligible_open_sizes_json"]),
        )))

        for strategy in ("uniform", "cluster", "uct"):
            original = run_search_task({
                **base,
                "method": f"seeastar_original_{strategy}",
                "k": 4,
            })
            self.assertTrue(original["found"])
            self.assertEqual(original["search_result"].sampling_strategy, strategy)

        phi = np.full(grid.shape, 0.2, dtype=np.float64)
        neural = run_search_task({
            "method": "neural_astar",
            "grid": grid,
            "start": sample.grid_map.start,
            "goal": sample.grid_map.goal,
            "phi": phi,
            "instrument_neural": True,
        })
        self.assertTrue(neural["found"])
        self.assertEqual(neural["neural_trace"].open_sizes[0], 1)
        self.assertAlmostEqual(neural["found_ground_truth_cost"], len(neural["path"]) - 1)
        self.assertAlmostEqual(neural["search_result"].goal_guidance_cost, 0.2 * (len(neural["path"]) - 1))

    def test_neural_guidance_arms_use_phi_and_corrected_fixed_and_adaptive_apis(self) -> None:
        import adaptive.ood_adaptive_seeastar as adaptive_module
        import benchmarks.search_worker as worker_module
        from seeastar.neural_guidance import run_neural_guidance_seeastar

        sample = make_dataset_sample(
            MapGenerationConfig(height=9, width=10, distribution="D2", seed=38123)
        )
        config = BenchmarkConfig()
        arms = experiment_arms(config)
        fixed_arm = next(
            arm for arm in arms if arm["method"] == "neural_guidance_fixed_k5"
        )
        adaptive_arm = next(
            arm
            for arm in arms
            if arm["method"] == "neural_guidance_adaptive_k_bounds4_300"
        )
        phi = np.full(sample.grid_map.grid.shape, 0.35, dtype=np.float64)
        maps = {
            "phi": phi,
            "H1": np.full(phi.shape, 11.0),
            "H2": np.full(phi.shape, 22.0),
            "H3": np.full(phi.shape, 33.0),
        }
        fixed_task = _task_for_arm(
            sample,
            maps,
            fixed_arm,
            seed=19,
            decision=None,
            instrument_neural=True,
        )
        adaptive_task = _task_for_arm(
            sample,
            maps,
            adaptive_arm,
            seed=19,
            decision=None,
            instrument_neural=True,
            mahalanobis_distance=2.5,
            d95=4.0,
        )
        np.testing.assert_array_equal(fixed_task["phi"], phi)
        np.testing.assert_array_equal(adaptive_task["phi"], phi)
        self.assertNotIn("heuristic_map", fixed_task)
        self.assertNotIn("heuristic_map", adaptive_task)
        self.assertEqual(adaptive_task["mahalanobis_distance"], 2.5)
        self.assertEqual(adaptive_task["d95"], 4.0)
        self.assertEqual((adaptive_task["k_min"], adaptive_task["k_max"]), (4, 300))
        self.assertEqual(adaptive_task["seed"], 19)
        fixed_task["capture_guidance_trace"] = True
        adaptive_task["capture_guidance_trace"] = True

        with (
            patch.object(
                worker_module,
                "run_neural_guidance_seeastar",
                wraps=run_neural_guidance_seeastar,
            ) as fixed_spy,
            patch.object(
                worker_module,
                "run_ood_adaptive_neural_guidance_seeastar",
                wraps=adaptive_module.run_ood_adaptive_neural_guidance_seeastar,
            ) as adaptive_spy,
        ):
            fixed = run_search_task(fixed_task)
            adaptive = run_search_task(adaptive_task)

        fixed_spy.assert_called_once()
        adaptive_spy.assert_called_once()
        adaptive_args = adaptive_spy.call_args.args
        adaptive_kwargs = adaptive_spy.call_args.kwargs
        self.assertEqual(adaptive_args[4], 2.5)
        self.assertEqual(adaptive_args[5], 4.0)
        self.assertEqual(adaptive_kwargs["k_min"], 4)
        self.assertEqual(adaptive_kwargs["k_max"], 300)
        self.assertEqual(adaptive_kwargs["seed"], 19)
        self.assertTrue(fixed["found"])
        self.assertTrue(adaptive["found"])
        np.testing.assert_array_equal(fixed["search_result"].guidance_map, phi)
        np.testing.assert_array_equal(adaptive["search_result"].guidance_map, phi)
        self.assertEqual(fixed["search_result"].configured_k, 5)
        self.assertEqual(adaptive["selected_k"], adaptive["adaptive_decision"]["selected_k"])
        self.assertIn("found_ground_truth_cost", fixed)
        self.assertIn("found_guidance_cost", fixed)
        self.assertIn("found_ground_truth_cost", adaptive)
        self.assertIn("found_guidance_cost", adaptive)
        self.assertNotEqual(fixed["found_ground_truth_cost"], fixed["found_guidance_cost"])
        self.assertTrue(fixed["search_result"].selection_trace)
        for item in fixed["search_result"].selection_trace:
            if item.expanded is not None:
                self.assertEqual(item.selected, item.expanded)
        trace_rows = json.loads(_result_counters(fixed)["guidance_selection_trace_json"])
        self.assertTrue(trace_rows)
        self.assertTrue(
            all(item["selected"] == item["expanded"] for item in trace_rows if item["expanded"] is not None)
        )

        inference_row = {
            "map_fingerprint": "synthetic",
            "mahalanobis_distance": 2.5,
            "smooth_ood_score": 1.0 - np.exp(-2.5 / 4.0),
            "d95": 4.0,
            "neural_astar_cnn_inference_seconds": 0.01,
            "ood_feature_cnn_inference_seconds": 0.02,
            "ood_detector_inference_seconds": 0.003,
        }
        fixed_row = _search_row(
            sample,
            inference_row,
            fixed,
            method=fixed_arm["method"],
            method_family=fixed_arm["method_family"],
            heuristic=None,
            seed=19,
            configured_k=5,
            adaptive_decision=None,
            worker_roundtrip_seconds=0.0,
            worker_restarted=False,
            loading_seconds={},
            inference_reuse_count=1,
        )
        adaptive_row = _search_row(
            sample,
            inference_row,
            adaptive,
            method=adaptive_arm["method"],
            method_family=adaptive_arm["method_family"],
            heuristic=None,
            seed=19,
            configured_k=adaptive["selected_k"],
            adaptive_decision=None,
            worker_roundtrip_seconds=0.0,
            worker_restarted=False,
            loading_seconds={},
            inference_reuse_count=1,
        )
        for row in (fixed_row, adaptive_row):
            self.assertIsNotNone(row["found_ground_truth_cost"])
            self.assertIsNotNone(row["found_guidance_accumulated_cost"])
            self.assertNotEqual(
                row["found_ground_truth_cost"], row["found_guidance_accumulated_cost"]
            )

        all_arms = _all_search_arms(config)
        identifiers = {arm["method"]: arm for arm in all_arms}
        self.assertEqual(identifiers["neural_astar"]["search_implementation"], "neural_astar")
        self.assertIn("original_seeastar_uniform_H1_k16", identifiers)
        self.assertIn("uniform_H1_adaptive_smooth_smallk_bounds4_300", identifiers)
        self.assertNotEqual(
            identifiers["uniform_H1_adaptive_smooth_smallk_bounds4_300"]["search_implementation"],
            "neural_guidance_adaptive",
        )


if __name__ == "__main__":
    unittest.main()
