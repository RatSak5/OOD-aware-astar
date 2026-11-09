"""Fast collapse-guard and checkpoint inference tests for Neural A*."""

from __future__ import annotations

import gc
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data.dataset import make_dataset_sample
from data.map_generator import MapGenerationConfig
from neural_astar.inference import load_neural_astar_checkpoint, predict_guidance
from neural_astar.model import NeuralAStarUNet
from neural_astar.training import (
    NeuralAStarTrainingConfig,
    _assert_phi_healthy,
    _forward_guidance_with_logits,
    _probe_phi_health,
    _require_finite_gradients,
    _require_finite_tensor,
    _summarize_phi_health,
    _tensor_inputs,
    train_neural_astar,
)
from neural_astar.differentiable import torch


@unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
class NeuralAStarHealthTests(unittest.TestCase):
    @staticmethod
    def _sample():
        return make_dataset_sample(
            MapGenerationConfig(height=10, width=10, distribution="D1", seed=774201)
        )

    def test_high_contrast_guidance_warns_but_does_not_stop_training(self) -> None:
        class HighContrastGuidance(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Conv2d(2, 1, kernel_size=1)
                torch.nn.init.zeros_(self.projection.weight)
                torch.nn.init.zeros_(self.projection.bias)
                with torch.no_grad():
                    self.projection.weight[0, 1, 0, 0] = 60.0
                    self.projection.bias.fill_(-12.0)

            def forward(self, features):
                return torch.sigmoid(self.projection(features))

        model = HighContrastGuidance()
        summary = _probe_phi_health(model, [self._sample()], "cpu")
        self.assertGreater(summary["logit_mean_abs"], 8.0)
        self.assertGreater(summary["traversable_phi_spatial_std_max"], 1e-3)
        self.assertGreater(summary["phi_saturated_fraction"], 0.95)
        with self.assertLogs("neural_astar.training", level="WARNING") as captured:
            _assert_phi_healthy(summary, context="high-contrast test")
        self.assertIn("mean absolute pre-sigmoid logit", captured.output[0])
        self.assertIn("exceeds threshold 8", captured.output[0])
        self.assertIn("continuing training", captured.output[0])

        validation = make_dataset_sample(
            MapGenerationConfig(height=10, width=10, distribution="D2", seed=774202)
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertLogs("neural_astar.training", level="WARNING"):
                _, history = train_neural_astar(
                    [self._sample()],
                    [validation],
                    directory,
                    config=NeuralAStarTrainingConfig(epochs=1, seed=774201, device="cpu"),
                    model=model,
                )
            self.assertEqual(len(history), 1)
            with (Path(directory) / "step_health.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            row = next(row for row in rows if row["health_probe"] == "1")
            self.assertGreater(float(row["logit_mean_abs"]), 8.0)
            self.assertGreater(float(row["traversable_phi_spatial_std_max"]), 1e-3)
        self.assertIn("traversable_phi_maps_below_1e_3_count", row)

    def test_constant_high_guidance_is_detected_independently_of_logit_warning(self) -> None:
        class ConstantGuidance(torch.nn.Module):
            def __init__(self, value: float):
                super().__init__()
                self.guidance_head = torch.nn.Conv2d(2, 1, kernel_size=1)
                torch.nn.init.zeros_(self.guidance_head.weight)
                torch.nn.init.constant_(self.guidance_head.bias, value)

            def forward(self, features):
                return torch.sigmoid(self.guidance_head(features))

        summary = _probe_phi_health(ConstantGuidance(20.0), [self._sample()], "cpu")
        self.assertGreater(summary["logit_mean_abs"], 8.0)
        with self.assertLogs("neural_astar.training", level="WARNING") as captured:
            _assert_phi_healthy(summary, context="constant-high test")
        self.assertTrue(any("mean absolute pre-sigmoid logit" in line for line in captured.output))
        self.assertTrue(any("low-spatial-variation diagnostic" in line for line in captured.output))
        self.assertLess(summary["traversable_phi_spatial_std_max"], 1e-3)
        self.assertEqual(summary["traversable_phi_maps_below_1e_3_count"], 1.0)
        self.assertEqual(summary["traversable_phi_maps_below_1e_3_proportion"], 1.0)

    def test_constant_low_guidance_is_detected_independently_of_logit_warning(self) -> None:
        class ConstantGuidance(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.guidance_head = torch.nn.Conv2d(2, 1, kernel_size=1)
                torch.nn.init.zeros_(self.guidance_head.weight)
                torch.nn.init.constant_(self.guidance_head.bias, -20.0)

            def forward(self, features):
                return torch.sigmoid(self.guidance_head(features))

        summary = _probe_phi_health(ConstantGuidance(), [self._sample()], "cpu")
        self.assertGreater(summary["logit_mean_abs"], 8.0)
        with self.assertLogs("neural_astar.training", level="WARNING") as captured:
            _assert_phi_healthy(summary, context="constant-low test")
        self.assertTrue(any("low-spatial-variation diagnostic" in line for line in captured.output))
        self.assertLess(summary["traversable_phi_spatial_std_max"], 1e-3)
        self.assertEqual(summary["traversable_phi_maps_below_1e_3_count"], 1.0)
        self.assertEqual(summary["traversable_phi_maps_below_1e_3_proportion"], 1.0)

    def test_constant_saturated_at_one_is_spatial_collapse(self) -> None:
        class SaturatedGuidance(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.guidance_head = torch.nn.Conv2d(2, 1, kernel_size=1)
                torch.nn.init.zeros_(self.guidance_head.weight)
                torch.nn.init.constant_(self.guidance_head.bias, 20.0)

            def forward(self, features):
                return torch.sigmoid(self.guidance_head(features))

        summary = _probe_phi_health(SaturatedGuidance(), [self._sample()], "cpu")
        self.assertEqual(summary["phi_fraction_gt_0_99"], 1.0)
        self.assertGreater(summary["logit_mean_abs"], 8.0)
        with self.assertLogs("neural_astar.training", level="WARNING") as captured:
            _assert_phi_healthy(summary, context="constant-saturated-at-one test")
        self.assertTrue(any("low-spatial-variation diagnostic" in line for line in captured.output))

    def test_freshly_initialized_low_variation_model_is_reported_but_non_fatal(self) -> None:
        class FreshlyInitializedGuidance(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.guidance_head = torch.nn.Conv2d(2, 1, kernel_size=1)
                torch.nn.init.zeros_(self.guidance_head.weight)
                torch.nn.init.constant_(self.guidance_head.bias, 0.096)
                with torch.no_grad():
                    self.guidance_head.weight[0, 1, 0, 0] = 0.012

            def forward(self, features):
                return torch.sigmoid(self.guidance_head(features))

        sample = self._sample()
        summary = _probe_phi_health(
            FreshlyInitializedGuidance(), [sample, sample, sample, sample], "cpu"
        )
        self.assertAlmostEqual(summary["traversable_phi_mean"], 0.524, delta=0.001)
        self.assertAlmostEqual(summary["traversable_phi_spatial_std_mean"], 0.0004, delta=0.0002)
        self.assertEqual(summary["traversable_phi_maps_below_1e_3_count"], 4.0)
        self.assertEqual(summary["traversable_phi_maps_below_1e_3_proportion"], 1.0)
        with self.assertLogs("neural_astar.training", level="WARNING") as captured:
            _assert_phi_healthy(summary, context="fresh-initialization test")
        self.assertTrue(any("4/4 probe maps" in line and "non-fatal" in line for line in captured.output))

    def test_healthy_spatially_varying_guidance_does_not_trip_guard(self) -> None:
        torch.manual_seed(712)
        model = NeuralAStarUNet().cpu()
        with torch.no_grad():
            model.guidance_head.weight.mul_(10.0)
            model.guidance_head.bias.zero_()
        summary = _probe_phi_health(model, [self._sample()], "cpu")
        self.assertGreaterEqual(summary["phi_spatial_std_max"], 1e-3)
        self.assertLessEqual(summary["phi_saturated_fraction"], 0.95)
        self.assertLessEqual(summary["logit_mean_abs"], 8.0)
        _assert_phi_healthy(summary, context="healthy varying-Phi test")

    def test_one_constant_map_does_not_establish_collapse_across_probes(self) -> None:
        constant_phi = torch.full((1, 2, 2), 1.0)
        varying_phi = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])
        constant_logits = torch.full((1, 1, 2, 2), 20.0)
        varying_logits = torch.tensor([[[[-20.0, -20.0], [20.0, 20.0]]]])
        masks = [torch.ones((2, 2), dtype=torch.bool), torch.ones((2, 2), dtype=torch.bool)]
        summary = _summarize_phi_health(
            [constant_phi, varying_phi], [constant_logits, varying_logits], masks
        )
        self.assertEqual(summary["traversable_phi_spatial_std_min"], 0.0)
        self.assertGreater(summary["traversable_phi_spatial_std_max"], 1e-3)
        self.assertEqual(summary["traversable_phi_maps_below_1e_3_count"], 1.0)
        self.assertEqual(summary["traversable_phi_maps_below_1e_3_proportion"], 0.5)
        with self.assertLogs("neural_astar.training", level="WARNING"):
            _assert_phi_healthy(summary, context="multiple-probe test")

    def test_obstacle_cells_are_excluded_from_traversable_statistics(self) -> None:
        free_phi = torch.tensor([0.1, 0.2, 0.8, 0.9])
        phi_maps = [
            torch.cat((free_phi, torch.tensor([0.0, 1.0, 0.0, 1.0]))).reshape(1, 1, 8),
            torch.cat((free_phi, torch.tensor([0.0]))).reshape(1, 1, 5),
        ]
        logits_maps = [
            torch.tensor([-2.0, -1.0, 1.0, 2.0, 6.0, -6.0, 7.0, -7.0]).reshape(1, 1, 8),
            torch.tensor([-2.0, -1.0, 1.0, 2.0, 20.0]).reshape(1, 1, 5),
        ]
        masks = [
            torch.tensor([[True] * 4 + [False] * 4]),
            torch.tensor([[True] * 4 + [False]]),
        ]
        summary = _summarize_phi_health(phi_maps, logits_maps, masks)
        expected_free_std = float(free_phi.std().item())
        self.assertAlmostEqual(summary["traversable_phi_spatial_std_mean"], expected_free_std)
        self.assertAlmostEqual(summary["traversable_phi_mean"], 0.5)
        self.assertAlmostEqual(summary["traversable_logit_mean_abs"], 1.5)
        self.assertAlmostEqual(summary["obstacle_logit_mean_abs"], 13.25)
        self.assertIn("all_cell_phi_mean", summary)
        self.assertIn("all_cell_logit_mean_abs", summary)
        self.assertNotAlmostEqual(summary["all_cell_phi_mean"], summary["traversable_phi_mean"])
        for field in (
            "traversable_logit_signed_mean", "traversable_logit_std_mean",
            "traversable_logit_min", "traversable_logit_max", "traversable_logit_median_mean",
            "traversable_logit_p05_mean", "traversable_logit_p95_mean",
            "traversable_logit_fraction_gt_4", "traversable_logit_fraction_lt_minus_4",
            "traversable_logit_fraction_abs_gt_4", "traversable_logit_fraction_abs_gt_8",
            "traversable_phi_std_mean", "traversable_phi_min", "traversable_phi_max",
            "traversable_phi_fraction_lt_0_01", "traversable_phi_fraction_gt_0_99",
            "obstacle_logit_signed_mean", "obstacle_phi_mean", "obstacle_phi_std_mean",
        ):
            self.assertIn(field, summary)

    def test_non_finite_logits_guidance_losses_and_gradients_fail_immediately(self) -> None:
        class BadLogits(torch.nn.Module):
            def __init__(self, value: float):
                super().__init__()
                self.projection = torch.nn.Conv2d(2, 1, kernel_size=1)
                torch.nn.init.zeros_(self.projection.weight)
                torch.nn.init.constant_(self.projection.bias, value)

            def forward(self, features):
                return torch.sigmoid(self.projection(features))

        for bad_value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(logit=bad_value):
                model = BadLogits(bad_value)
                inputs, _ = _tensor_inputs(self._sample(), "cpu")
                with self.assertRaises(FloatingPointError):
                    _forward_guidance_with_logits(model, inputs)

        logits = torch.zeros((1, 1, 2, 2))
        for bad_value in (float("nan"), float("inf")):
            with self.subTest(guidance=bad_value):
                with self.assertRaises(FloatingPointError):
                    _summarize_phi_health(
                        [torch.full((1, 2, 2), bad_value)], [logits]
                    )
                with self.assertRaises(FloatingPointError):
                    _require_finite_tensor(torch.tensor(bad_value), "training loss")

        class BadGuidanceOutput(torch.nn.Module):
            def __init__(self, value: float):
                super().__init__()
                self.projection = torch.nn.Conv2d(2, 1, kernel_size=1)
                torch.nn.init.zeros_(self.projection.weight)
                torch.nn.init.zeros_(self.projection.bias)
                self.value = value

            def forward(self, features):
                finite_logits = self.projection(features)
                return torch.full_like(finite_logits, self.value)

        inputs, _ = _tensor_inputs(self._sample(), "cpu")
        for bad_value in (float("nan"), float("inf")):
            with self.subTest(model_guidance=bad_value):
                with self.assertRaises(ValueError):
                    _forward_guidance_with_logits(BadGuidanceOutput(bad_value), inputs)

        parameter = torch.nn.Parameter(torch.ones(1))
        for bad_value in (float("nan"), float("inf")):
            with self.subTest(gradient=bad_value):
                parameter.grad = torch.tensor([bad_value])
                with self.assertRaises(FloatingPointError):
                    _require_finite_gradients([parameter])

    def test_checkpoint_roundtrip_without_model_argument_runs_predict_guidance(self) -> None:
        torch.manual_seed(104)
        model = NeuralAStarUNet().cpu()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4)
        features = torch.rand(1, 2, 8, 8)
        target = torch.linspace(0.1, 0.9, 64).reshape(1, 1, 8, 8)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(features), target)
            loss.backward()
            optimizer.step()
        config = NeuralAStarTrainingConfig(epochs=1, seed=104, device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "trained_checkpoint.pt"
            torch.save(
                {
                    "epoch": 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": {**config.__dict__} if hasattr(config, "__dict__") else {
                        field: getattr(config, field) for field in config.__dataclass_fields__
                    },
                },
                checkpoint_path,
            )
            del model
            gc.collect()
            restored, payload = load_neural_astar_checkpoint(checkpoint_path, device="cpu")
            self.assertEqual(payload["epoch"], 1)
            guidance, elapsed = predict_guidance(
                restored, np.zeros((8, 8), dtype=np.uint8), (0, 0), (7, 7), device="cpu"
            )
            self.assertEqual(guidance.shape, (8, 8))
            self.assertTrue(np.isfinite(guidance).all())
            self.assertGreaterEqual(float(guidance.min()), 0.0)
            self.assertLessEqual(float(guidance.max()), 1.0)
            self.assertGreaterEqual(elapsed, 0.0)

    def test_resume_rejects_checkpoint_with_mismatched_training_config(self) -> None:
        class TinyGuidance(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Conv2d(2, 1, kernel_size=1)

            def forward(self, features):
                return torch.sigmoid(self.projection(features))

        model = TinyGuidance()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)
        training = self._sample()
        validation = make_dataset_sample(
            MapGenerationConfig(height=10, width=10, distribution="D2", seed=774202)
        )
        with tempfile.TemporaryDirectory() as directory:
            torch.save(
                {
                    "epoch": 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": {"normalization": "none"},
                },
                Path(directory) / "last_checkpoint.pt",
            )
            with self.assertRaisesRegex(ValueError, "different configuration"):
                train_neural_astar(
                    [training],
                    [validation],
                    directory,
                    config=NeuralAStarTrainingConfig(epochs=1, seed=774201, device="cpu"),
                    model=model,
                    resume=True,
                )

    def test_gradient_accumulation_consumes_the_effective_batch(self) -> None:
        class TinyGuidance(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Conv2d(2, 1, kernel_size=1)

            def forward(self, features):
                return torch.sigmoid(self.projection(features))

        training = [self._sample()]
        training.append(make_dataset_sample(
            MapGenerationConfig(height=10, width=10, distribution="D2", seed=774202)
        ))
        validation = make_dataset_sample(
            MapGenerationConfig(height=10, width=10, distribution="D3", seed=774203)
        )
        with tempfile.TemporaryDirectory() as directory:
            _, history = train_neural_astar(
                training,
                [validation],
                directory,
                config=NeuralAStarTrainingConfig(
                    epochs=1,
                    batch_size=1,
                    gradient_accumulation_steps=2,
                    seed=774201,
                    device="cpu",
                ),
                model=TinyGuidance(),
            )
            with (Path(directory) / "step_health.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(history), 1)
            self.assertEqual({int(row["optimizer_step"]) for row in rows}, {1})
            health_row = next(row for row in rows if row["health_probe"] == "1")
            self.assertIn("traversable_logit_mean_abs", health_row)
            self.assertIn("obstacle_phi_std_mean", health_row)
            self.assertTrue(np.isfinite(float(health_row["traversable_logit_mean_abs"])))

    def test_resume_extends_legacy_health_csv_without_losing_old_columns(self) -> None:
        class TinyGuidance(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Conv2d(2, 1, kernel_size=1)

            def forward(self, features):
                return torch.sigmoid(self.projection(features))

        training = self._sample()
        validation = make_dataset_sample(
            MapGenerationConfig(height=10, width=10, distribution="D2", seed=774202)
        )
        config = NeuralAStarTrainingConfig(epochs=1, seed=774201, device="cpu")
        legacy_fields = (
            "epoch", "optimizer_step", "loss", "phi_spatial_std_mean", "phi_spatial_std_min",
            "phi_spatial_std_max", "phi_across_map_mean_std", "phi_fraction_lt_0_01",
            "phi_fraction_gt_0_99", "phi_saturated_fraction", "logit_mean_abs", "logit_max_abs",
            "gradient_norm", "health_probe", "epoch_end",
        )
        with tempfile.TemporaryDirectory() as directory:
            model = TinyGuidance()
            train_neural_astar([training], [validation], directory, config=config, model=model)
            health_path = Path(directory) / "step_health.csv"
            with health_path.open(newline="", encoding="utf-8") as stream:
                old_rows = list(csv.DictReader(stream))
            with health_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=legacy_fields)
                writer.writeheader()
                writer.writerows({key: row[key] for key in legacy_fields} for row in old_rows)

            _, history = train_neural_astar(
                [training], [validation], directory, config=config, model=TinyGuidance(), resume=True
            )
            with health_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                migrated_rows = list(reader)
                fields = set(reader.fieldnames or ())
            self.assertIn("traversable_logit_mean_abs", fields)
            self.assertEqual(len(migrated_rows), len(old_rows))
            self.assertEqual(migrated_rows[-1]["logit_mean_abs"], old_rows[-1]["logit_mean_abs"])
            self.assertEqual(migrated_rows[-1]["traversable_logit_mean_abs"], "")
            self.assertEqual(len(history), 1)

    def test_health_summary_uses_pre_sigmoid_logits_and_both_saturation_tails(self) -> None:
        phi = torch.tensor([[[0.005, 0.5], [0.995, 0.25]]])
        logits = torch.tensor([[[[-5.0, 0.0], [5.0, -1.0]]]])
        summary = _summarize_phi_health([phi], [logits])
        self.assertAlmostEqual(summary["phi_fraction_lt_0_01"], 0.25)
        self.assertAlmostEqual(summary["phi_fraction_gt_0_99"], 0.25)
        self.assertAlmostEqual(summary["phi_saturated_fraction"], 0.5)
        self.assertAlmostEqual(summary["logit_mean_abs"], 2.75)
        self.assertIn("all_cell_logit_mean_abs", summary)
        self.assertNotIn("traversable_logit_mean_abs", summary)


if __name__ == "__main__":
    unittest.main()
