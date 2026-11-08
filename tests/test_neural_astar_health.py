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
    NeuralAStarCollapseError,
    NeuralAStarTrainingConfig,
    _assert_phi_healthy,
    _probe_phi_health,
    _summarize_phi_health,
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

    def test_constant_output_network_trips_collapse_guard(self) -> None:
        class ConstantGuidance(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.guidance_head = torch.nn.Conv2d(2, 1, kernel_size=1)
                torch.nn.init.zeros_(self.guidance_head.weight)
                torch.nn.init.zeros_(self.guidance_head.bias)

            def forward(self, features):
                return torch.sigmoid(self.guidance_head(features))

        summary = _probe_phi_health(ConstantGuidance(), [self._sample()], "cpu")
        with self.assertRaises(NeuralAStarCollapseError):
            _assert_phi_healthy(summary, context="constant-output test")

    def test_saturated_at_one_network_trips_collapse_guard(self) -> None:
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
        with self.assertRaises(NeuralAStarCollapseError):
            _assert_phi_healthy(summary, context="saturated-at-one test")

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

    def test_health_summary_uses_pre_sigmoid_logits_and_both_saturation_tails(self) -> None:
        phi = torch.tensor([[[0.005, 0.5], [0.995, 0.25]]])
        logits = torch.tensor([[[[-5.0, 0.0], [5.0, -1.0]]]])
        summary = _summarize_phi_health([phi], [logits])
        self.assertAlmostEqual(summary["phi_fraction_lt_0_01"], 0.25)
        self.assertAlmostEqual(summary["phi_fraction_gt_0_99"], 0.25)
        self.assertAlmostEqual(summary["phi_saturated_fraction"], 0.5)
        self.assertAlmostEqual(summary["logit_mean_abs"], 2.75)


if __name__ == "__main__":
    unittest.main()
