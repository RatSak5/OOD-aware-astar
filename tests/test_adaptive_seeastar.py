"""Tests for continuous-score adaptive K and uniform-only SeeA* experiments."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from adaptive.ood_adaptive_seeastar import (
    AdaptiveKPolicy,
    clipped_ood_score,
    compute_adaptive_k,
    compute_neural_guidance_adaptive_k,
    run_legacy_unit_cost_adaptive_seeastar,
    run_ood_adaptive_neural_guidance_seeastar,
    run_ood_adaptive_seeastar,
)
from algorithms.movement import DEFAULT_COST_CONFIG
from data.dataset import make_dataset_sample, map_fingerprint
from data.map_generator import MapGenerationConfig
from experiments.adaptive_seeastar import (
    AdaptiveSeeAStarExperimentConfig,
    NeuralGuidanceSeeAStarExperimentConfig,
    run_adaptive_seeastar_experiment,
    run_neural_guidance_seeastar_experiment,
)
from ood.feature_extractor import OODFeatureConfig, OODFeatureExtractor
from ood.mahalanobis import (
    CalibrationError,
    MahalanobisConfig,
    MahalanobisDetector,
    calibrate_d95,
    fit_mahalanobis_detector,
)
from seeastar.sampling import UniformSampler
import adaptive.ood_adaptive_seeastar as adaptive_module
import seeastar.original as seeastar_module
from seeastar.neural_guidance import run_neural_guidance_seeastar

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None


class AdaptiveKMathTests(unittest.TestCase):
    def test_smooth_score_zero_selects_k_max(self) -> None:
        decision = compute_adaptive_k(0.0, 3.0, AdaptiveKPolicy(2, 12))
        self.assertEqual(decision.smooth_ood_score, 0.0)
        self.assertEqual(decision.policy_score, 0.0)
        self.assertEqual(decision.selected_k, 12)

    def test_smooth_score_one_half_uses_floor_rule(self) -> None:
        d95 = 7.0
        distance = -np.log1p(-0.5) * d95
        decision = compute_adaptive_k(distance, d95, AdaptiveKPolicy(2, 12))
        self.assertAlmostEqual(decision.smooth_ood_score, 0.5)
        self.assertAlmostEqual(decision.policy_score, 0.5)
        self.assertEqual(decision.selected_k, 7)

    def test_exact_score_one_is_mathematical_boundary(self) -> None:
        # The smooth normalizer never emits exactly 1 for finite D_M. Exercise
        # the exact endpoint directly at the policy boundary to verify the
        # specified floor equation maps s=1 to K_min.
        with patch.object(
            adaptive_module,
            "normalized_ood_score",
            return_value=np.float64(1.0),
        ):
            decision = compute_adaptive_k(100.0, 1.0, AdaptiveKPolicy(2, 12))
        self.assertEqual(decision.policy_score, 1.0)
        self.assertEqual(decision.selected_k, 2)

    def test_proposed_k_is_monotone_and_bounded_across_full_score_interval(self) -> None:
        policy = AdaptiveKPolicy(k_min=2, k_max=12)
        scores = np.linspace(0.0, np.nextafter(1.0, 0.0), 1001)
        decisions = [
            compute_adaptive_k(
                float(-np.log1p(-score)),
                1.0,
                policy,
            )
            for score in scores
        ]
        selected = [decision.selected_k for decision in decisions]
        self.assertTrue(all(left >= right for left, right in zip(selected, selected[1:])))
        self.assertTrue(all(policy.k_min <= value <= policy.k_max for value in selected))
        self.assertEqual(selected[0], policy.k_max)

    def test_proposed_k_is_monotone_and_obeys_boundaries(self) -> None:
        policy = AdaptiveKPolicy(k_min=2, k_max=12)
        distances = np.linspace(0.0, 10.0, 101)
        decisions = [compute_adaptive_k(float(value), 10.0, policy) for value in distances]
        selected = [decision.selected_k for decision in decisions]
        self.assertEqual(selected[0], policy.k_max)
        self.assertTrue(all(left >= right for left, right in zip(selected, selected[1:])))
        self.assertTrue(all(policy.k_min <= value <= policy.k_max for value in selected))
        self.assertEqual(selected[-1], 6)  # smooth s(D95)=1-exp(-1)
        self.assertEqual(
            selected[-1],
            policy.k_max
            - int(np.floor((1.0 - np.exp(-1.0)) * (policy.k_max - policy.k_min))),
        )

    def test_reverse_direction_is_monotone_increasing(self) -> None:
        policy = AdaptiveKPolicy(2, 12, direction="higher_ood_larger_k")
        selected = [compute_adaptive_k(float(d), 10.0, policy).selected_k for d in range(11)]
        self.assertEqual(selected[0], 2)
        self.assertTrue(all(left <= right for left, right in zip(selected, selected[1:])))
        self.assertTrue(all(2 <= value <= 12 for value in selected))

    def test_clipped_normalization_ablation_is_distinct_and_bounded(self) -> None:
        smooth = compute_adaptive_k(
            10.0, 10.0, AdaptiveKPolicy(2, 12, normalization="smooth")
        )
        clipped = compute_adaptive_k(
            10.0, 10.0, AdaptiveKPolicy(2, 12, normalization="clipped")
        )
        self.assertAlmostEqual(smooth.smooth_ood_score, 1.0 - np.exp(-1.0))
        self.assertEqual(clipped.policy_score, 1.0)
        self.assertEqual(clipped.selected_k, 2)
        self.assertEqual(clipped_ood_score(100.0, 10.0), 1.0)

    def test_zero_very_large_and_equal_k_bounds(self) -> None:
        policy = AdaptiveKPolicy(3, 9)
        zero = compute_adaptive_k(0.0, 2.0, policy)
        self.assertEqual(zero.smooth_ood_score, 0.0)
        self.assertEqual(zero.selected_k, 9)
        huge = compute_adaptive_k(1e308, 2.0, policy)
        self.assertGreater(huge.smooth_ood_score, 0.99)
        self.assertEqual(huge.selected_k, 4)
        self.assertLess(huge.smooth_ood_score, 1.0)
        self.assertEqual(huge.selected_k, policy.k_min + 1)
        fixed = compute_adaptive_k(3.0, 2.0, AdaptiveKPolicy(5, 5))
        self.assertEqual(fixed.selected_k, 5)

    def test_near_zero_d95_is_a_calibration_error(self) -> None:
        policy = AdaptiveKPolicy(1, 4)
        for d95 in (0.0, 1e-15):
            with self.subTest(d95=d95), self.assertRaises(CalibrationError):
                compute_adaptive_k(1.0, d95, policy)
        with self.assertRaises(CalibrationError):
            clipped_ood_score(1.0, 0.0)

    def test_invalid_bounds_and_scores_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AdaptiveKPolicy(0, 5)
        with self.assertRaises(ValueError):
            AdaptiveKPolicy(5, 4)
        with self.assertRaises(ValueError):
            compute_adaptive_k(-1.0, 1.0, AdaptiveKPolicy())


class UniformSamplingFoundationTests(unittest.TestCase):
    def test_uniform_without_replacement_remains_the_only_candidate_sampler(self) -> None:
        nodes = [(0, column) for column in range(5)]
        f_values = {node: float(node[1]) for node in nodes}
        depths = {node: 0 for node in nodes}
        sampler = UniformSampler(3, seed=19)
        selected = sampler.select(nodes, f_values, depths)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(set(selected)), 3)
        self.assertTrue(set(selected).issubset(nodes))


class NeuralGuidanceAdaptiveKTests(unittest.TestCase):
    def test_zero_equal_to_d95_tiny_and_large_finite_distances(self) -> None:
        zero = compute_neural_guidance_adaptive_k(0.0, 2.0)
        self.assertEqual(zero.ood_score, 0.0)
        self.assertEqual(zero.selected_k, 16)

        equal = compute_neural_guidance_adaptive_k(2.0, 2.0)
        expected_score = 1.0 - np.exp(-1.0)
        self.assertAlmostEqual(equal.ood_score, expected_score)
        self.assertEqual(equal.selected_k, 9)

        small = compute_neural_guidance_adaptive_k(1e-12, 2.0)
        self.assertGreater(small.ood_score, 0.0)
        self.assertEqual(small.selected_k, 16)

        large = compute_neural_guidance_adaptive_k(1e308, 1.0)
        self.assertTrue(np.isfinite(large.ood_score))
        self.assertGreaterEqual(large.ood_score, 0.0)
        self.assertLess(large.ood_score, 1.0)
        self.assertEqual(large.selected_k, 5)

    def test_invalid_distance_and_calibration_values_are_rejected(self) -> None:
        for distance in (-1.0, np.nan, np.inf, -np.inf):
            with self.subTest(distance=distance), self.assertRaises(ValueError):
                compute_neural_guidance_adaptive_k(distance, 1.0)
        for d95 in (0.0, -1.0, np.nan, np.inf, -np.inf):
            with self.subTest(d95=d95), self.assertRaises(CalibrationError):
                compute_neural_guidance_adaptive_k(1.0, d95)

    def test_exact_primary_formula_and_configurable_bounds(self) -> None:
        distance, d95 = 5.0, 8.0
        decision = compute_neural_guidance_adaptive_k(distance, d95)
        score = 1.0 - np.exp(-distance / d95)
        expected_k = 16 - int(np.floor(score * (16 - 4)))
        self.assertAlmostEqual(decision.ood_score, score)
        self.assertEqual(decision.selected_k, expected_k)
        self.assertEqual(
            compute_neural_guidance_adaptive_k(0.0, 8.0, k_min=2, k_max=8).selected_k,
            8,
        )

    def test_adaptive_wrapper_calculates_k_once_and_holds_it_fixed(self) -> None:
        grid = np.zeros((5, 6), dtype=np.uint8)
        phi = np.arange(grid.size, dtype=np.float64).reshape(grid.shape) / 3.0
        with patch.object(
            adaptive_module,
            "compute_neural_guidance_adaptive_k",
            wraps=compute_neural_guidance_adaptive_k,
        ) as decision_spy:
            result = run_ood_adaptive_neural_guidance_seeastar(
                grid,
                (0, 0),
                (4, 5),
                phi,
                0.7,
                2.0,
                seed=51,
                capture_trace=True,
            )
        self.assertEqual(decision_spy.call_count, 1)
        self.assertEqual(result.search.configured_k, result.selected_k)
        self.assertTrue(
            all(size <= result.selected_k for size in result.search.candidate_sizes)
        )
        self.assertTrue(result.search.selection_trace)
        self.assertEqual(result.found_ground_truth_cost, len(result.path) - 1)

    def test_ood_changes_only_selected_k_and_passes_the_same_phi(self) -> None:
        grid = np.zeros((5, 6), dtype=np.uint8)
        phi = np.linspace(0.0, 1.7, grid.size).reshape(grid.shape)
        original_core = adaptive_module.neural_guidance_search.run_guidance_uniform_seeastar
        with patch.object(
            adaptive_module.neural_guidance_search,
            "run_guidance_uniform_seeastar",
            wraps=original_core,
        ) as core_spy:
            low = run_ood_adaptive_neural_guidance_seeastar(
                grid, (0, 0), (4, 5), phi, 0.0, 1.0, seed=11
            )
            high = run_ood_adaptive_neural_guidance_seeastar(
                grid, (0, 0), (4, 5), phi, 2.0, 1.0, seed=11
            )
        self.assertEqual(core_spy.call_count, 2)
        first_phi = core_spy.call_args_list[0].args[3]
        second_phi = core_spy.call_args_list[1].args[3]
        np.testing.assert_array_equal(first_phi, phi)
        np.testing.assert_array_equal(second_phi, phi)
        self.assertNotEqual(low.selected_k, high.selected_k)
        self.assertEqual(low.search.guidance_map.tolist(), high.search.guidance_map.tolist())
        low_config = core_spy.call_args_list[0].args[4]
        high_config = core_spy.call_args_list[1].args[4]
        self.assertEqual(low_config.seed, high_config.seed)
        self.assertEqual(low_config.max_expansions, high_config.max_expansions)
        self.assertEqual(low_config.k, low.selected_k)
        self.assertEqual(high_config.k, high.selected_k)
        self.assertEqual(core_spy.call_args_list[0].args[5], core_spy.call_args_list[1].args[5])

    def test_fixed_and_adaptive_same_selected_k_seed_have_identical_mechanics(self) -> None:
        grid = np.zeros((6, 7), dtype=np.uint8)
        grid[2, 3] = 1
        phi = np.linspace(0.05, 1.4, grid.size).reshape(grid.shape)
        fixed = run_neural_guidance_seeastar(
            grid, (0, 0), (5, 6), phi, k=7, seed=991, capture_trace=True
        )
        adaptive = run_ood_adaptive_neural_guidance_seeastar(
            grid,
            (0, 0),
            (5, 6),
            phi,
            3.0,
            5.0,
            k_min=7,
            k_max=7,
            seed=991,
            capture_trace=True,
        )
        self.assertEqual(adaptive.selected_k, 7)
        self.assertEqual(fixed.path, adaptive.search.path)
        self.assertEqual(fixed.expanded_order, adaptive.search.expanded_order)
        self.assertEqual(fixed.selection_trace, adaptive.search.selection_trace)
        self.assertEqual(fixed.parent_map, adaptive.search.parent_map)
        self.assertEqual(fixed.found_guidance_cost, adaptive.found_guidance_cost)
        self.assertEqual(fixed.found_ground_truth_cost, adaptive.found_ground_truth_cost)

    def test_adaptive_wrapper_reports_guidance_and_ground_truth_cost_separately(self) -> None:
        grid = np.zeros((2, 3), dtype=np.uint8)
        phi = np.asarray([[50.0, 0.3, 4.0], [2.0, 1.1, 0.8]])
        result = run_ood_adaptive_neural_guidance_seeastar(
            grid, (0, 0), (1, 2), phi, 0.0, 2.0, seed=77
        )
        self.assertTrue(result.search.found)
        self.assertEqual(result.found_ground_truth_cost, len(result.path) - 1)
        self.assertEqual(
            result.found_guidance_cost,
            sum(float(phi[node]) for node in result.path[1:]),
        )
        self.assertNotEqual(result.found_guidance_cost, result.found_ground_truth_cost)

    def test_legacy_conventional_adaptive_alias_keeps_existing_implementation(self) -> None:
        self.assertIs(run_legacy_unit_cost_adaptive_seeastar, run_ood_adaptive_seeastar)

    def test_guidance_experiment_reuses_phi_for_fixed_and_adaptive_arms(self) -> None:
        import experiments.adaptive_seeastar as experiment_module

        sample = _sample("D1", 4488)
        phi = np.linspace(0.1, 2.0, sample.grid_map.grid.size).reshape(
            sample.grid_map.grid.shape
        )
        with tempfile.TemporaryDirectory() as directory:
            fixed_original = experiment_module.run_neural_guidance_seeastar
            adaptive_original = experiment_module.run_ood_adaptive_neural_guidance_seeastar
            with (
                patch.object(
                    experiment_module,
                    "run_neural_guidance_seeastar",
                    wraps=fixed_original,
                ) as fixed_spy,
                patch.object(
                    experiment_module,
                    "run_ood_adaptive_neural_guidance_seeastar",
                    wraps=adaptive_original,
                ) as adaptive_spy,
            ):
                results = run_neural_guidance_seeastar_experiment(
                    [sample],
                    [phi],
                    [0.5],
                    1.0,
                    directory,
                    config=NeuralGuidanceSeeAStarExperimentConfig(
                        fixed_ks=(4, 5), k_min=4, k_max=5, seed=17
                    ),
                )
            self.assertEqual(fixed_spy.call_count, 2)
            adaptive_spy.assert_called_once()
            for call in fixed_spy.call_args_list:
                np.testing.assert_array_equal(call.args[3], phi)
            np.testing.assert_array_equal(adaptive_spy.call_args.args[3], phi)
            self.assertEqual(len(results["per_map"]), 3)
            for row in results["per_map"]:
                self.assertIn("found_ground_truth_cost", row)
                self.assertIn("found_guidance_accumulated_cost", row)
                self.assertEqual(row["phi_source"], "Neural A* guidance map")
            self.assertTrue(
                Path(directory, "neural_guidance_seeastar_per_map.csv").is_file()
            )
            self.assertTrue(
                Path(directory, "neural_guidance_seeastar_sampling_trace.csv").is_file()
            )


def _sample(distribution: str, seed: int):
    return make_dataset_sample(
        MapGenerationConfig(height=10, width=11, distribution=distribution, seed=seed),
        DEFAULT_COST_CONFIG,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class AdaptiveSearchIntegrationTests(unittest.TestCase):
    if torch is not None:
        class ZeroHeuristic(torch.nn.Module):
            def forward(self, inputs):
                return torch.zeros(
                    (inputs.shape[0], 1, inputs.shape[-2], inputs.shape[-1]),
                    dtype=inputs.dtype,
                    device=inputs.device,
                )
    else:  # pragma: no cover - test class is skipped without torch
        class ZeroHeuristic:
            pass

    def _detector_and_extractor(self):
        extractor = OODFeatureExtractor(
            OODFeatureConfig(
                base_channels=8,
                depth=2,
                embedding_dim=4,
                projection_dim=4,
            )
        ).eval()
        train = [_sample(distribution, 4100 + index) for index, distribution in enumerate(("D1", "D2", "D3"))]
        calibration = [
            _sample(distribution, 4200 + index)
            for index, distribution in enumerate(("D1", "D2", "D3"))
        ]
        detector = fit_mahalanobis_detector(train, extractor)
        calibrate_d95(detector, calibration, extractor)
        return extractor, detector, train, calibration

    def test_single_search_fixes_k_and_handles_small_or_empty_open(self) -> None:
        extractor, detector, _, _ = self._detector_and_extractor()
        grid = np.zeros((4, 5), dtype=np.uint8)
        policy = AdaptiveKPolicy(4, 9)
        feature_calls: list[None] = []
        hook = extractor.register_forward_hook(lambda _model, _inputs, _output: feature_calls.append(None))
        with patch.object(
            adaptive_module,
            "compute_adaptive_k",
            wraps=compute_adaptive_k,
        ) as adaptive_k_spy:
            result = run_ood_adaptive_seeastar(
                grid,
                (0, 0),
                (3, 4),
                extractor,
                detector,
                policy=policy,
                heuristic=lambda node, goal: 0.0,
                seed=77,
                heuristic_inference_seconds=0.25,
            )
        hook.remove()
        self.assertEqual(adaptive_k_spy.call_count, 1)
        self.assertEqual(len(feature_calls), 1)
        self.assertEqual(result.search.sampling_strategy, "uniform")
        self.assertTrue(policy.k_min <= result.decision.selected_k <= policy.k_max)
        self.assertTrue(result.search.found)
        self.assertTrue(
            all(size <= result.decision.selected_k for size in result.search.candidate_sizes)
        )
        self.assertTrue(
            all(
                sampled <= eligible
                for eligible, sampled in zip(
                    result.search.eligible_open_sizes, result.search.candidate_sizes
                )
            )
        )
        self.assertTrue(
            all(
                sampled == eligible
                for eligible, sampled in zip(
                    result.search.eligible_open_sizes, result.search.candidate_sizes
                )
                if eligible < policy.k_min
            )
        )
        self.assertGreaterEqual(
            result.total_end_to_end_seconds,
            result.search.search_time_seconds + result.heuristic_inference_seconds,
        )
        self.assertGreaterEqual(result.cnn_inference_seconds, 0.0)
        self.assertGreaterEqual(result.ood_inference_seconds, 0.0)

        blocked = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        unreachable = run_ood_adaptive_seeastar(
            blocked,
            (0, 0),
            (1, 1),
            extractor,
            detector,
            policy=policy,
            seed=77,
        )
        self.assertFalse(unreachable.search.found)
        self.assertEqual(unreachable.search.stop_reason, "open_exhausted")
        self.assertEqual(unreachable.search.final_open_size, 0)

    def test_adaptive_search_expands_the_lowest_f_sampled_candidate(self) -> None:
        extractor, detector, _, _ = self._detector_and_extractor()
        grid = np.zeros((6, 7), dtype=np.uint8)
        start, goal = (0, 0), (5, 6)
        sample_records: list[tuple[tuple[tuple[int, int], ...], dict[tuple[int, int], float]]] = []
        rank_records: list[tuple[tuple[tuple[int, int], ...], dict[tuple[int, int], float], tuple[int, int]]] = []
        original_select = UniformSampler.select
        original_rank = seeastar_module.lowest_f_candidate

        def record_sample(sampler, open_nodes, f_values, depths):
            candidates = tuple(original_select(sampler, open_nodes, f_values, depths))
            sample_records.append(
                (candidates, {node: float(f_values[node]) for node in candidates})
            )
            return list(candidates)

        def record_rank(candidates, f_values):
            candidate_tuple = tuple(candidates)
            candidate_fs = {node: float(f_values[node]) for node in candidate_tuple}
            selected = original_rank(candidate_tuple, f_values)
            rank_records.append((candidate_tuple, candidate_fs, selected))
            return selected

        with (
            patch.object(seeastar_module.UniformSampler, "select", new=record_sample),
            patch.object(seeastar_module, "lowest_f_candidate", new=record_rank),
        ):
            result = run_ood_adaptive_seeastar(
                grid,
                start,
                goal,
                extractor,
                detector,
                policy=AdaptiveKPolicy(2, 2),
                heuristic=lambda node, target: 0.0,
                seed=51,
            )

        self.assertTrue(result.search.found)
        self.assertGreaterEqual(len(sample_records), 2)
        self.assertEqual(len(sample_records), len(rank_records))
        for (sampled, sampled_fs), (ranked, ranked_fs, selected) in zip(
            sample_records, rank_records
        ):
            self.assertEqual(sampled, ranked)
            self.assertEqual(sampled_fs, ranked_fs)
            self.assertEqual(
                selected,
                min(sampled, key=lambda node: (sampled_fs[node], node)),
            )

        # The first selection expands the start; the second selected node is
        # non-goal and must be the next state recorded as actually expanded.
        second_selected = rank_records[1][2]
        self.assertNotEqual(second_selected, goal)
        self.assertEqual(result.search.expanded_order[0], start)
        self.assertEqual(result.search.expanded_order[1], second_selected)

    def test_fixed_seed_reproduces_score_k_and_search(self) -> None:
        extractor, detector, _, _ = self._detector_and_extractor()
        grid = np.zeros((5, 6), dtype=np.uint8)
        policy = AdaptiveKPolicy(2, 7)
        first = run_ood_adaptive_seeastar(
            grid, (0, 0), (4, 5), extractor, detector,
            policy=policy, seed=123,
        )
        second = run_ood_adaptive_seeastar(
            grid, (0, 0), (4, 5), extractor, detector,
            policy=policy, seed=123,
        )
        self.assertEqual(first.decision, second.decision)
        self.assertEqual(first.search.path, second.search.path)
        self.assertEqual(first.search.expanded_order, second.search.expanded_order)

    def test_experiment_writes_all_uniform_ablation_outputs_and_plots(self) -> None:
        extractor, detector, train, calibration = self._detector_and_extractor()
        evaluation = [
            _sample(distribution, 4300 + index)
            for index, distribution in enumerate(("D1", "D2", "D3", "D4", "D5", "D6"))
        ]
        self.assertTrue(
            set(map_fingerprint(item.grid_map.grid) for item in evaluation).isdisjoint(
                set(detector.training_fingerprints) | set(detector.calibration_fingerprints)
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            results = run_adaptive_seeastar_experiment(
                evaluation,
                extractor,
                detector,
                self.ZeroHeuristic(),
                "H1",
                directory,
                config=AdaptiveSeeAStarExperimentConfig(
                    k_bounds=((1, 3), (2, 2)),
                    seed=66,
                ),
            )
            methods = {row["method"] for row in results["per_map"]}
            self.assertEqual(
                methods,
                {
                    "fixed_k_min",
                    "fixed_k_mid",
                    "fixed_k_max",
                    "adaptive_high_ood_smaller_k",
                    "adaptive_reverse_high_ood_larger_k",
                    "adaptive_clipped_high_ood_smaller_k",
                },
            )
            self.assertTrue(
                all(row["sampling_strategy"] == "uniform" for row in results["per_map"])
            )
            self.assertEqual({row["distribution"] for row in results["summary"]},
                             {"D1", "D2", "D3", "D4", "D5", "D6"})
            self.assertTrue(results["paired_comparisons"])
            self.assertTrue(results["heuristic_errors"])
            for filename in (
                "adaptive_seeastar_per_map.csv",
                "adaptive_seeastar_sampling_trace.csv",
                "adaptive_seeastar_summary.csv",
                "adaptive_seeastar_paired_comparisons.csv",
                "adaptive_seeastar_paired_summary.csv",
                "adaptive_seeastar_heuristic_error.csv",
                "adaptive_seeastar_config.json",
            ):
                self.assertTrue((Path(directory) / filename).is_file(), filename)
            self.assertTrue(results["plots"])
            adaptive_rows = [
                row
                for row in results["per_map"]
                if row["method"] == "adaptive_high_ood_smaller_k"
            ]
            self.assertTrue(
                all(row["cnn_inference_seconds"] > 0.0 for row in adaptive_rows)
            )
            fixed_rows = [row for row in results["per_map"] if row["method"].startswith("fixed")]
            self.assertTrue(all(row["cnn_inference_seconds"] == 0.0 for row in fixed_rows))
            self.assertTrue(all(row["candidates_sampled_max"] <= row["selected_k"] for row in results["per_map"]))
            self.assertTrue(
                all(0.0 <= row["expanded_grid_coverage"] <= 1.0 for row in results["per_map"])
            )


if __name__ == "__main__":
    unittest.main()

