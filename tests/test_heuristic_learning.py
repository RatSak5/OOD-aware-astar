"""Tests for separate conventional CNN heuristic training methods H1-H3."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from algorithms.movement import DEFAULT_COST_CONFIG
from data.dataset import DatasetSample, generate_standard_splits, make_dataset_sample
from data.ground_truth import compute_ground_truth
from data.map_generator import GridMap, MapGenerationConfig
from experiments.heuristic_comparison import run_heuristic_comparison
from heuristics.cea import CEALossConfig, cross_entropy_admissibility_loss
from heuristics.data import build_heuristic_input, collate_heuristic_samples
from heuristics.evaluation import evaluate_heuristic_model
from heuristics.inference import (
    fixed_goal_heuristic,
    load_heuristic_checkpoint,
    predict_heuristic_map,
)
from heuristics.losses import (
    ConstraintAwareLossConfig,
    compute_heuristic_losses,
)
from models.heuristic_cnn import (
    HeuristicCNN,
    HeuristicCNNConfig,
    decode_heuristic_output,
)
from training.heuristic_training import (
    HeuristicTrainingConfig,
    _validate_splits,
    train_heuristic_model,
)
from visualization.heuristic_learning import (
    plot_heuristic_histories,
    plot_ood_heuristic_metrics,
)

try:
    import torch
except ImportError:  # pragma: no cover - depends on local packages
    torch = None


def _disconnected_sample() -> DatasetSample:
    """Small D1-labeled map with a free but goal-unreachable right component."""

    grid = np.zeros((5, 5), dtype=np.uint8)
    grid[:, 2] = 1
    start, goal = (2, 0), (2, 1)
    generation = MapGenerationConfig(height=5, width=5, distribution="D1", seed=765)
    grid_map = GridMap(grid, start, goal, "D1", 765, "test fixture")
    ground_truth = compute_ground_truth(grid, start, goal, DEFAULT_COST_CONFIG)
    return DatasetSample(grid_map, ground_truth, generation, DEFAULT_COST_CONFIG)


class HeuristicDataTests(unittest.TestCase):
    def test_input_channels_are_obstacles_start_and_goal(self) -> None:
        grid = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        features = build_heuristic_input(grid, (0, 0), (1, 1))
        self.assertEqual(features.shape, (3, 2, 2))
        np.testing.assert_array_equal(features[0], [[0, 1], [0, 0]])
        np.testing.assert_array_equal(features[1], [[1, 0], [0, 0]])
        np.testing.assert_array_equal(features[2], [[0, 0], [0, 1]])

    def test_fixed_goal_adapter_exposes_raw_map_values_to_astar(self) -> None:
        prediction = np.array([[3.5, 2.0], [1.0, 0.0]])
        heuristic = fixed_goal_heuristic(prediction, (1, 1))
        self.assertEqual(heuristic((0, 0), (1, 1)), 3.5)
        with self.assertRaisesRegex(ValueError, "different goal"):
            heuristic((0, 0), (0, 1))

    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_batch_masks_obstacles_and_unreachable_cells_and_has_bidirectional_edges(self) -> None:
        batch = collate_heuristic_samples([_disconnected_sample()])
        self.assertFalse(bool(batch.valid_mask[0, :, 2].any().item()))
        self.assertFalse(bool(batch.valid_mask[0, :, 3:].any().item()))
        self.assertTrue(bool(batch.valid_mask[0, :, :2].all().item()))
        self.assertGreater(batch.edge_indices.shape[0], 0)
        self.assertTrue(bool((batch.edge_costs == 1.0).all().item()))
        edge_pairs = {
            (int(source), int(destination))
            for _, source, destination in batch.edge_indices.tolist()
        }
        self.assertTrue(all((v, u) in edge_pairs for u, v in edge_pairs))

    def test_train_validation_splits_are_map_independent_and_d1_to_d3(self) -> None:
        train = make_dataset_sample(
            MapGenerationConfig(height=8, width=8, distribution="D1", seed=8101)
        )
        validation = make_dataset_sample(
            MapGenerationConfig(height=8, width=8, distribution="D2", seed=8202)
        )
        self.assertEqual(_validate_splits([train], [validation]), (8, 8))
        with self.assertRaisesRegex(ValueError, "Duplicate map"):
            _validate_splits([train], [train])


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class HeuristicLossTests(unittest.TestCase):
    def test_cea_matches_positive_target_formula(self) -> None:
        probabilities = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float64)
        logits = probabilities.log().reshape(1, 3, 1, 1)
        target = torch.tensor([[[2]]])
        mask = torch.tensor([[[True]]])
        config = CEALossConfig(beta=1.0, eta=0.1)
        actual = cross_entropy_admissibility_loss(logits, target, mask, config)
        weighted_admissible_mass = (1.0 / 2.0) * 0.25 + 1.0 * 0.25
        expected = -np.log(weighted_admissible_mass) - 0.1 * np.log(0.25)
        self.assertAlmostEqual(float(actual.item()), float(expected), places=6)

    def test_cea_zero_cost_boundary_is_finite_and_prefers_class_zero(self) -> None:
        logits = torch.tensor([[[[1.0]], [[0.0]], [[-1.0]]]], requires_grad=True)
        target = torch.tensor([[[0]]])
        mask = torch.tensor([[[True]]])
        loss = cross_entropy_admissibility_loss(logits, target, mask)
        self.assertTrue(bool(torch.isfinite(loss).item()))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all().item()))

    def test_h1_is_masked_mean_squared_error(self) -> None:
        raw = torch.tensor([[[[1.0, 3.0], [99.0, 0.0]]]], requires_grad=True)
        target = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
        valid = torch.tensor([[[True, True], [False, True]]])
        edges = torch.empty((0, 3), dtype=torch.long)
        costs = torch.empty((0,), dtype=torch.float32)
        goals = torch.tensor([[1, 1]])
        total, terms, prediction = compute_heuristic_losses(
            "H1", raw, target, valid, edges, costs, goals
        )
        self.assertAlmostEqual(float(total.item()), 5.0 / 3.0, places=6)
        self.assertAlmostEqual(float(terms["loss_fit"].item()), 5.0 / 3.0, places=6)
        self.assertEqual(float(prediction[0, 1, 0].item()), 99.0)

    def test_h2_uses_unit_ground_truth_edge_costs_and_asymmetric_fit(self) -> None:
        raw = torch.tensor([[[[3.0, 1.0, 0.0]]]], requires_grad=True)
        target = torch.tensor([[[1.0, 0.0, 0.0]]])
        valid = torch.ones_like(target, dtype=torch.bool)
        # Both directions for each undirected edge. c(u,v)=1 comes from the
        # shared project ground-truth model, never from a Neural A* Phi map.
        edges = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 1, 2], [0, 2, 1]])
        costs = torch.ones(4)
        goals = torch.tensor([[0, 2]])
        total, terms, prediction = compute_heuristic_losses(
            "H2", raw, target, valid, edges, costs, goals
        )
        self.assertAlmostEqual(float(terms["loss_fit"].item()), 50.0 / 3.0, places=5)
        self.assertAlmostEqual(float(terms["loss_adm"].item()), 5.0 / 3.0, places=5)
        self.assertAlmostEqual(float(terms["loss_cons"].item()), 0.25, places=5)
        self.assertAlmostEqual(float(terms["loss_goal"].item()), 0.0, places=5)
        expected_total = 50.0 / 3.0 + 10.0 * 5.0 / 3.0 + 10.0 * 0.25
        self.assertAlmostEqual(float(total.item()), expected_total, places=5)
        total.backward()
        self.assertTrue(bool(torch.isfinite(raw.grad).all().item()))
        self.assertEqual(float(prediction[0, 0, 0].item()), 3.0)

    def test_cnn_regression_and_ordinal_heads_have_expected_shapes(self) -> None:
        inputs = torch.rand(2, 3, 9, 7)
        regression = HeuristicCNN(
            HeuristicCNNConfig(base_channels=8, depth=3, normalization="group")
        )(inputs)
        self.assertEqual(tuple(regression.shape), (2, 1, 9, 7))
        self.assertTrue(bool((regression >= 0.0).all().item()))
        ordinal_config = HeuristicCNNConfig(
            base_channels=8,
            depth=2,
            normalization="group",
            output_mode="ordinal",
        )
        ordinal_logits = HeuristicCNN(ordinal_config, num_classes=12)(inputs)
        self.assertEqual(tuple(ordinal_logits.shape), (2, 12, 9, 7))
        classes = decode_heuristic_output(ordinal_logits, "H3")
        self.assertEqual(tuple(classes.shape), (2, 9, 7))

    def test_evaluator_reports_unmodified_prediction_violations(self) -> None:
        class RampPredictor(torch.nn.Module):
            def forward(self, inputs):
                batch, _, height, width = inputs.shape
                columns = torch.arange(width, device=inputs.device, dtype=inputs.dtype)
                values = 10.0 + 2.0 * columns.view(1, 1, 1, width)
                return values.expand(batch, 1, height, width)

        metrics = evaluate_heuristic_model(
            RampPredictor(), [_disconnected_sample()], "H1"
        )["D1"]
        self.assertEqual(metrics["admissibility_violation_pct"], 100.0)
        self.assertGreater(metrics["consistency_violation_pct"], 0.0)
        self.assertEqual(metrics["maximum_overestimation"], 12.0)
        self.assertEqual(metrics["goal_prediction_error"], 12.0)

    def test_training_each_method_writes_history_and_restorable_checkpoint(self) -> None:
        train_sample = make_dataset_sample(
            MapGenerationConfig(height=6, width=6, distribution="D1", seed=9901)
        )
        validation_sample = make_dataset_sample(
            MapGenerationConfig(height=6, width=6, distribution="D2", seed=9902)
        )
        model_config = HeuristicCNNConfig(base_channels=8, depth=2, normalization="group")
        for method in ("H1", "H2", "H3"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as directory:
                config = HeuristicTrainingConfig(
                    method=method,
                    epochs=1,
                    batch_size=1,
                    early_stopping_patience=1,
                    model=model_config,
                    seed=91,
                )
                model, history = train_heuristic_model(
                    [train_sample], [validation_sample], directory, config=config
                )
                self.assertTrue(history)
                self.assertTrue((Path(directory) / "best_checkpoint.pt").is_file())
                self.assertTrue((Path(directory) / "last_checkpoint.pt").is_file())
                self.assertTrue((Path(directory) / "history.csv").is_file())
                restored, checkpoint = load_heuristic_checkpoint(
                    Path(directory) / "best_checkpoint.pt"
                )
                self.assertEqual(checkpoint["method"], method)
                predicted = predict_heuristic_map(
                    restored,
                    train_sample.grid_map.grid,
                    train_sample.grid_map.start,
                    train_sample.grid_map.goal,
                    method=method,
                )
                self.assertEqual(predicted.shape, train_sample.grid_map.grid.shape)
                self.assertTrue(np.isfinite(predicted).all())


class HeuristicPlotTests(unittest.TestCase):
    def test_training_and_ood_plots_cover_losses_errors_and_violations(self) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            self.skipTest("Matplotlib is not installed")
        rows = [
            {
                "epoch": 1,
                "split": "train",
                "distribution": "all",
                "loss_total": 2.0,
                "heuristic_mae": 0.5,
                "admissibility_violation_pct": 10.0,
                "consistency_violation_pct": 5.0,
            },
            {
                "epoch": 1,
                "split": "validation",
                "distribution": "all",
                "loss_total": 3.0,
                "heuristic_mae": 0.8,
                "admissibility_violation_pct": 15.0,
                "consistency_violation_pct": 7.0,
            },
        ]
        metric = {
            distribution: {
                "heuristic_mae": 1.0,
                "admissibility_violation_pct": 2.0,
                "consistency_violation_pct": 3.0,
            }
            for distribution in ("D4", "D5", "D6")
        }
        with tempfile.TemporaryDirectory() as directory:
            from matplotlib import pyplot as plt
            paths = plot_heuristic_histories({"H1": rows}, directory)
            paths.update(plot_ood_heuristic_metrics({"H1": metric}, directory))
            self.assertIn("loss_curves.png", paths)
            self.assertIn("consistency_violation_curves.png", paths)
            self.assertEqual(len(paths), 7)
            self.assertTrue(all(path.is_file() for path in paths.values()))
            plt.close("all")


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class HeuristicExperimentIntegrationTests(unittest.TestCase):
    def test_comparison_trains_separately_and_reports_d4_to_d6(self) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
        except ImportError:
            self.skipTest("Matplotlib is not installed")
        splits = generate_standard_splits(
            train_count=1,
            validation_count=1,
            ood_test_count=3,
            base_seed=10777,
            map_config=MapGenerationConfig(height=8, width=8),
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_heuristic_comparison(
                splits["train"],
                splits["validation"],
                splits["test"],
                directory,
                base_config=HeuristicTrainingConfig(
                    epochs=1,
                    batch_size=1,
                    early_stopping_patience=1,
                    seed=40,
                    model=HeuristicCNNConfig(
                        base_channels=8,
                        depth=2,
                        normalization="group",
                    ),
                ),
            )
            self.assertEqual(set(result["models"]), {"H1", "H2", "H3"})
            for method in ("H1", "H2", "H3"):
                self.assertEqual(
                    set(result["ood_metrics"][method]) - {"all"},
                    {"D4", "D5", "D6"},
                )
            self.assertTrue((Path(directory) / "ood_metrics.csv").is_file())
            self.assertTrue((Path(directory) / "validation_metrics.csv").is_file())
            self.assertTrue((Path(directory) / "plots" / "ood_prediction_error.png").is_file())


if __name__ == "__main__":
    unittest.main()
