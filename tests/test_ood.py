"""Tests for independent feature extraction and continuous Mahalanobis OOD scores."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.dataset import make_dataset_sample, map_fingerprint
from data.map_generator import MapGenerationConfig
from experiments.ood_detection import run_ood_detection
from ood.contrastive import (
    ContrastiveTrainingConfig,
    load_feature_extractor,
    nt_xent_loss,
    train_contrastive_feature_extractor,
)
from ood.evaluation import aupr_trapezoid, auroc, evaluate_ood_detector
from ood.feature_extractor import OODFeatureConfig, OODFeatureExtractor, build_ood_input
from ood.mahalanobis import (
    CalibrationError,
    MahalanobisConfig,
    MahalanobisDetector,
    calibrate_d95,
    fit_mahalanobis_detector,
    ledoit_wolf_covariance,
    load_detector,
    normalized_ood_score,
    save_detector,
)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _sample(distribution: str, seed: int):
    return make_dataset_sample(
        MapGenerationConfig(height=10, width=11, distribution=distribution, seed=seed)
    )


def _small_feature_config() -> OODFeatureConfig:
    return OODFeatureConfig(
        base_channels=4,
        depth=2,
        embedding_dim=8,
        projection_dim=4,
        normalization="group",
    )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class FeatureExtractorTests(unittest.TestCase):
    def test_input_channels_and_embedding_dimensions(self) -> None:
        grid = np.array([[0, 1, 0], [0, 0, 0]], dtype=np.uint8)
        features = build_ood_input(grid, (0, 0), (1, 2))
        self.assertEqual(features.shape, (3, 2, 3))
        self.assertEqual(features[0, 0, 1], 1.0)
        self.assertEqual(features[1, 0, 0], 1.0)
        self.assertEqual(features[2, 1, 2], 1.0)
        model = OODFeatureExtractor(_small_feature_config()).eval()
        embedded = model(torch.as_tensor(np.stack((features, features)), dtype=torch.float32))
        projected = model.project(embedded)
        self.assertEqual(tuple(embedded.shape), (2, 8))
        self.assertEqual(tuple(projected.shape), (2, 4))
        self.assertTrue(torch.allclose(projected.norm(dim=1), torch.ones(2), atol=1e-6))

    def test_contrastive_loss_is_finite_and_repeatable(self) -> None:
        first = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        second = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        loss_a = nt_xent_loss(first, second)
        loss_b = nt_xent_loss(first, second)
        self.assertTrue(bool(torch.isfinite(loss_a).item()))
        torch.testing.assert_close(loss_a, loss_b)

    def test_one_epoch_contrastive_training_saves_checkpoint_and_config(self) -> None:
        samples = [_sample("D1", 12001), _sample("D2", 12002)]
        config = ContrastiveTrainingConfig(
            epochs=1, batch_size=2, seed=71, model=_small_feature_config()
        )
        with tempfile.TemporaryDirectory() as directory:
            model, history = train_contrastive_feature_extractor(samples, directory, config=config)
            self.assertEqual(len(history), 1)
            self.assertTrue((Path(directory) / "feature_extractor.pt").is_file())
            self.assertTrue((Path(directory) / "feature_training_config.json").is_file())
            self.assertTrue((Path(directory) / "feature_history.csv").is_file())
            restored, payload = load_feature_extractor(Path(directory) / "feature_extractor.pt")
            self.assertEqual(payload["epoch"], 1)
            self.assertEqual(restored.config.embedding_dim, 8)


class MahalanobisMathTests(unittest.TestCase):
    def test_ledoit_wolf_covariance_is_finite_symmetric_and_shrunk(self) -> None:
        embeddings = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
        mean, covariance, shrinkage = ledoit_wolf_covariance(embeddings)
        self.assertEqual(mean.shape, (3,))
        self.assertTrue(np.isfinite(covariance).all())
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
        self.assertGreaterEqual(shrinkage, 0.0)
        self.assertLessEqual(shrinkage, 1.0)

    def test_distance_uses_regularized_precision_and_is_reproducible(self) -> None:
        detector = MahalanobisDetector(
            mean=np.zeros(2),
            sample_covariance=np.eye(2),
            covariance=np.eye(2),
            precision=np.eye(2),
            shrinkage=0.5,
            ridge_added=1e-6,
            num_training_maps=2,
            training_fingerprints=(),
            config=MahalanobisConfig(),
        )
        points = np.array([[0.0, 0.0], [3.0, 4.0]])
        expected = np.array([0.0, 5.0])
        np.testing.assert_allclose(detector.distance(points), expected)
        np.testing.assert_array_equal(detector.distance(points), detector.distance(points))

    def test_continuous_score_boundaries_monotonicity_and_large_distance(self) -> None:
        d95 = 2.5
        scores = normalized_ood_score(np.array([0.0, 0.5 * d95, d95, 2.0 * d95]), d95)
        self.assertEqual(float(scores[0]), 0.0)
        self.assertAlmostEqual(float(scores[2]), 1.0 - np.exp(-1.0), places=14)
        self.assertTrue(np.all(np.diff(scores) > 0.0))
        self.assertGreater(float(scores[3]), float(scores[2]))
        large = float(normalized_ood_score(1e308, 1e-12))
        self.assertLess(large, 1.0)
        self.assertGreater(large, 0.99)
        with self.assertRaises(CalibrationError):
            normalized_ood_score(1.0, 0.0)

    def test_auroc_and_trapezoidal_aupr(self) -> None:
        self.assertEqual(auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])), 1.0)
        self.assertEqual(aupr_trapezoid(np.array([0.0, 1.0]), np.array([2.0, 3.0])), 1.0)
        self.assertAlmostEqual(auroc(np.array([1.0]), np.array([1.0])), 0.5)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DetectorSplitAndCalibrationTests(unittest.TestCase):
    def _splits(self):
        train = [_sample("D1", 13001), _sample("D2", 13002), _sample("D3", 13003)]
        validation = [_sample("D1", 14001), _sample("D2", 14002), _sample("D3", 14003)]
        ood = [_sample("D4", 15004), _sample("D5", 15005), _sample("D6", 15006)]
        return train, validation, ood

    def test_fit_uses_d1_to_d3_training_maps_and_adds_positive_ridge(self) -> None:
        train, _, _ = self._splits()
        model = OODFeatureExtractor(_small_feature_config())
        detector = fit_mahalanobis_detector(train, model)
        self.assertEqual(detector.num_training_maps, len(train))
        self.assertEqual(
            set(detector.training_fingerprints),
            {map_fingerprint(sample.grid_map.grid) for sample in train},
        )
        self.assertGreater(detector.ridge_added, 0.0)
        self.assertGreater(float(np.linalg.eigvalsh(detector.covariance).min()), 0.0)
        with self.assertRaisesRegex(ValueError, "D4"):
            fit_mahalanobis_detector([_sample("D4", 15999), train[0]], model)

    def test_d95_comes_from_held_out_id_and_rejects_near_zero(self) -> None:
        train, validation, _ = self._splits()
        model = OODFeatureExtractor(_small_feature_config()).eval()
        detector = fit_mahalanobis_detector(train, model)
        calibrated = calibrate_d95(detector, validation, model)
        self.assertGreater(calibrated.d95, calibrated.config.d95_minimum)
        self.assertEqual(calibrated.num_calibration_maps, len(validation))
        with self.assertRaisesRegex(ValueError, "overlap"):
            calibrate_d95(detector, [train[0], validation[1], validation[2]], model)

        class ConstantEncoder(torch.nn.Module):
            def forward(self, inputs):
                return torch.zeros((inputs.shape[0], 2), device=inputs.device)

        constant_detector = MahalanobisDetector(
            mean=np.zeros(2), sample_covariance=np.eye(2), covariance=np.eye(2),
            precision=np.eye(2), shrinkage=1.0, ridge_added=1e-6,
            num_training_maps=2, training_fingerprints=(), config=MahalanobisConfig(),
        )
        with self.assertRaises(CalibrationError):
            calibrate_d95(constant_detector, validation, ConstantEncoder())

    def test_detector_save_load_round_trip(self) -> None:
        train, validation, _ = self._splits()
        model = OODFeatureExtractor(_small_feature_config()).eval()
        detector = calibrate_d95(fit_mahalanobis_detector(train, model), validation, model)
        with tempfile.TemporaryDirectory() as directory:
            path = save_detector(detector, Path(directory) / "stats.npz")
            restored = load_detector(path)
        np.testing.assert_array_equal(restored.mean, detector.mean)
        np.testing.assert_array_equal(restored.precision, detector.precision)
        self.assertEqual(restored.d95, detector.d95)
        self.assertEqual(restored.shrinkage, detector.shrinkage)

    def test_end_to_end_run_writes_detector_metrics_and_plots(self) -> None:
        train, validation, ood = self._splits()
        with tempfile.TemporaryDirectory() as directory:
            result = run_ood_detection(
                train,
                validation,
                ood,
                directory,
                training_config=ContrastiveTrainingConfig(
                    epochs=1,
                    batch_size=2,
                    seed=72,
                    model=_small_feature_config(),
                ),
                bootstrap_repetitions=5,
                seed=72,
            )
            self.assertGreater(result["detector"].d95, 0.0)
            for name in (
                "feature_extractor.pt",
                "feature_training_config.json",
                "feature_history.csv",
                "mahalanobis_detector.npz",
                "mahalanobis_detector.json",
                "ood_evaluation.json",
                "ood_scores.csv",
                "ood_binary_metrics.csv",
            ):
                self.assertTrue((Path(directory) / name).is_file(), name)
            self.assertTrue(result["plots"])

    def test_evaluation_reports_continuous_scores_binary_metrics_and_h_error(self) -> None:
        train, validation, ood = self._splits()
        model = OODFeatureExtractor(_small_feature_config()).eval()
        detector = calibrate_d95(fit_mahalanobis_detector(train, model), validation, model)
        class ZeroHeuristic(torch.nn.Module):
            def forward(self, inputs):
                return torch.zeros(
                    (inputs.shape[0], 1, inputs.shape[-2], inputs.shape[-1]),
                    device=inputs.device,
                )

        results = evaluate_ood_detector(
            detector,
            model,
            validation,
            ood,
            heuristic_models={"H1": (ZeroHeuristic(), "H1")},
            bootstrap_repetitions=30,
            seed=44,
        )
        self.assertEqual(len(results["scores"]), len(validation) + len(ood))
        self.assertEqual({row["distribution"] for row in results["score_summaries_by_distribution"]},
                         {"D1", "D2", "D3", "D4", "D5", "D6"})
        self.assertEqual(len(results["binary_metrics"]), 4)
        self.assertEqual(results["binary_metrics"][0]["positive_class"], "OOD")
        self.assertEqual(len(results["heuristic_errors"]), len(validation) + len(ood))
        self.assertEqual(len(results["heuristic_error_correlations"]), 28)
        self.assertIn("distance_mean_ci95_low", results["score_summaries_by_distribution"][0])
        self.assertIn("mean_distance_gap_ood_minus_id", results["binary_metrics"][0])

        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            from visualization.ood import plot_heuristic_error_scatter, plot_ood_score_distributions
        except ImportError:
            self.skipTest("Matplotlib is not installed")
        with tempfile.TemporaryDirectory() as directory:
            score_plots = plot_ood_score_distributions(
                results["score_summaries_by_distribution"], results["scores"], directory
            )
            correlation_plots = plot_heuristic_error_scatter(
                results["heuristic_errors"], directory
            )
            self.assertEqual(len(score_plots), 2)
            self.assertEqual(len(correlation_plots), 1)
            self.assertTrue(all(path.is_file() for path in (*score_plots.values(), *correlation_plots.values())))


if __name__ == "__main__":
    unittest.main()
