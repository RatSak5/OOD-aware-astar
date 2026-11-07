"""Neural A* representation, hard search, and optional PyTorch tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from algorithms.movement import get_neighbors, get_transition_cost
from data.dataset import make_dataset_sample
from data.map_generator import MapGenerationConfig
from neural_astar.differentiable import (
    differentiable_neural_search,
    identity_backward_argmax,
    torch,
)
from neural_astar.guidance import (
    accumulate_guidance_cost,
    build_encoder_input,
    path_to_mask,
    validate_guidance_map,
)
from neural_astar.inference import load_neural_astar_checkpoint, plan_with_neural_astar
from neural_astar.model import NeuralAStarUNet, make_guidance_tensor
from neural_astar.search import normalized_open_weights, run_neural_astar
from neural_astar.training import (
    NeuralAStarTrainingConfig,
    _validate_splits,
    train_neural_astar,
)


class GuidanceRepresentationTests(unittest.TestCase):
    def test_encoder_input_inverts_project_map_and_encodes_both_endpoints(self) -> None:
        grid = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        features = build_encoder_input(grid, (0, 0), (1, 1))
        self.assertEqual(features.shape, (2, 2, 2))
        np.testing.assert_array_equal(features[0], [[1, 0], [1, 1]])
        np.testing.assert_array_equal(features[1], [[1, 0], [0, 1]])

    def test_same_start_and_goal_sums_the_two_one_hot_maps(self) -> None:
        features = build_encoder_input(np.zeros((1, 1), dtype=np.uint8), (0, 0), (0, 0))
        self.assertEqual(features[1, 0, 0], 2.0)

    def test_guidance_map_requires_matching_shape_finite_values_and_sigmoid_range(self) -> None:
        valid = validate_guidance_map(np.array([[0.0, 0.5, 1.0]]), (1, 3))
        self.assertEqual(valid.dtype, np.float64)
        for invalid in (
            np.array([[0.0, 1.1]]),
            np.array([[0.0, np.nan]]),
            np.array([0.0, 1.0]),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_guidance_map(invalid)
        with self.assertRaises(ValueError):
            validate_guidance_map(np.zeros((2, 2)), (3, 3))

    def test_path_to_mask_marks_exact_ground_truth_cells(self) -> None:
        mask = path_to_mask(((0, 0), (1, 1), (1, 2)), (2, 3))
        np.testing.assert_array_equal(mask, [[1, 0, 0], [0, 1, 1]])
        np.testing.assert_array_equal(path_to_mask(None, (2, 2)), np.zeros((2, 2)))

    def test_guidance_accumulation_is_destination_entry_cost(self) -> None:
        self.assertEqual(accumulate_guidance_cost(0.25, 0.5), 0.75)
        with self.assertRaises(ValueError):
            accumulate_guidance_cost(0.0, 1.1)


class NeuralAStarForwardTests(unittest.TestCase):
    def test_selector_weights_are_normalized_only_over_open_and_stable(self) -> None:
        guidance_cost = np.array([[1.0e6, 1.0e6 + 1.0, 1.0e6 + 5.0]])
        heuristic = np.zeros_like(guidance_cost)
        open_mask = np.array([[True, True, False]])
        weights = normalized_open_weights(guidance_cost, heuristic, open_mask, 1.0e-9)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertEqual(weights[0, 2], 0.0)
        self.assertEqual(weights[0, 0], 1.0)
        self.assertEqual(weights[0, 1], 0.0)

    def test_unit_line_accumulates_destination_cell_guidance_cost(self) -> None:
        grid = np.zeros((1, 3), dtype=np.uint8)
        phi = np.array([[0.9, 0.3, 0.4]])
        result = run_neural_astar(grid, (0, 0), (0, 2), phi)
        self.assertTrue(result.found)
        self.assertEqual(result.path, ((0, 0), (0, 1), (0, 2)))
        self.assertAlmostEqual(result.accumulated_guidance_cost[0, 1], 0.3)
        self.assertAlmostEqual(result.goal_guidance_cost, 0.7)
        self.assertAlmostEqual(result.accumulated_guidance_cost[0, 0], 0.0)
        self.assertEqual(result.temperature, np.sqrt(3))
        # Phi is unrelated to the project's unit ground-truth transition cost.
        self.assertEqual(get_transition_cost(grid, (0, 0), (0, 1)), 1.0)

    def test_diagonal_corner_cutting_is_blocked(self) -> None:
        grid = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        result = run_neural_astar(grid, (0, 0), (1, 1), np.zeros((2, 2)))
        self.assertFalse(result.found)
        self.assertIsNone(result.path)
        self.assertEqual(result.selected_order, ((0, 0),))
        self.assertEqual(get_neighbors(grid, (0, 0)), [])

    def test_search_path_uses_only_shared_legal_successors(self) -> None:
        grid = np.array(
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 1, 1, 0]],
            dtype=np.uint8,
        )
        phi = np.linspace(0.0, 1.0, grid.size).reshape(grid.shape)
        result = run_neural_astar(grid, (0, 0), (3, 3), phi)
        self.assertTrue(result.found)
        self.assertEqual(result.path[0], (0, 0))
        self.assertEqual(result.path[-1], (3, 3))
        for source, destination in zip(result.path, result.path[1:]):
            self.assertIn(destination, get_neighbors(grid, source))

    def test_start_equals_goal_terminates_without_charging_start(self) -> None:
        result = run_neural_astar(np.zeros((2, 2)), (1, 1), (1, 1), np.ones((2, 2)))
        self.assertTrue(result.found)
        self.assertEqual(result.path, ((1, 1),))
        self.assertEqual(result.goal_guidance_cost, 0.0)
        self.assertEqual(result.selected_order, ((1, 1),))

    def test_search_is_reproducible_for_identical_inputs(self) -> None:
        grid = np.zeros((4, 5), dtype=np.uint8)
        grid[1, 2] = 1
        phi = np.random.default_rng(87).random(grid.shape)
        first = run_neural_astar(grid, (0, 0), (3, 4), phi)
        second = run_neural_astar(grid, (0, 0), (3, 4), phi)
        self.assertEqual(first.path, second.path)
        self.assertEqual(first.selected_order, second.selected_order)
        np.testing.assert_array_equal(
            first.accumulated_guidance_cost, second.accumulated_guidance_cost
        )

    def test_guidance_dimensions_and_values_are_checked_before_search(self) -> None:
        grid = np.zeros((2, 2), dtype=np.uint8)
        with self.assertRaises(ValueError):
            run_neural_astar(grid, (0, 0), (1, 1), np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            run_neural_astar(grid, (0, 0), (1, 1), np.full((2, 2), 1.01))


class OptionalPyTorchTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_identity_backward_argmax_has_hard_forward_and_identity_gradient(self) -> None:
        values = torch.tensor([[0.1, 0.7, 0.2]], requires_grad=True)
        output = identity_backward_argmax(values)
        torch.testing.assert_close(output, torch.tensor([[0.0, 1.0, 0.0]]))
        (output * torch.tensor([[2.0, 3.0, 4.0]])).sum().backward()
        torch.testing.assert_close(values.grad, torch.tensor([[2.0, 3.0, 4.0]]))

    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_differentiable_search_finds_path_and_backpropagates(self) -> None:
        grid = np.zeros((2, 3), dtype=np.uint8)
        phi = torch.full((2, 3), 0.5, requires_grad=True)
        result = differentiable_neural_search(grid, (0, 0), (1, 2), phi)
        self.assertTrue(result.found)
        self.assertEqual(result.path[0], (0, 0))
        self.assertEqual(result.path[-1], (1, 2))
        target = torch.as_tensor(path_to_mask(result.path, grid.shape))
        loss = torch.abs(result.closed_mask - target).sum() / target.numel()
        loss.backward()
        self.assertIsNotNone(phi.grad)
        self.assertTrue(bool(torch.isfinite(phi.grad).all().item()))

    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_vgg_unet_has_sigmoid_output_for_odd_dimensions(self) -> None:
        model = NeuralAStarUNet()
        output = model(torch.rand(1, 2, 17, 19))
        checked = make_guidance_tensor(output)
        self.assertEqual(tuple(checked.shape), (1, 1, 17, 19))

    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_inference_wrapper_times_guidance_and_search_separately(self) -> None:
        class TinyGuidanceNet(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projection = torch.nn.Conv2d(2, 1, kernel_size=1)

            def forward(self, inputs):
                return torch.sigmoid(self.projection(inputs))

        grid = np.zeros((4, 5), dtype=np.uint8)
        output = plan_with_neural_astar(TinyGuidanceNet(), grid, (0, 0), (3, 4))
        self.assertEqual(output.guidance_map.shape, grid.shape)
        self.assertTrue(output.search.found)
        self.assertGreaterEqual(output.model_inference_seconds, 0.0)
        self.assertGreaterEqual(output.total_seconds, output.model_inference_seconds)

    @unittest.skipIf(torch is None, "PyTorch is not installed in this environment")
    def test_one_epoch_training_saves_loss_log_and_checkpoints(self) -> None:
        class TinyGuidanceNet(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projection = torch.nn.Conv2d(2, 1, kernel_size=1)

            def forward(self, inputs):
                return torch.sigmoid(self.projection(inputs))

        train_sample = NeuralAStarSplitPolicyTests._sample("D1", 9001)
        validation_sample = NeuralAStarSplitPolicyTests._sample("D2", 9002)
        with tempfile.TemporaryDirectory() as directory:
            _, history = train_neural_astar(
                [train_sample],
                [validation_sample],
                directory,
                config=NeuralAStarTrainingConfig(epochs=1, batch_size=1, seed=44),
                model=TinyGuidanceNet(),
            )
            self.assertEqual(len(history), 1)
            self.assertTrue((Path(directory) / "best_checkpoint.pt").is_file())
            self.assertTrue((Path(directory) / "last_checkpoint.pt").is_file())
            self.assertTrue((Path(directory) / "training_config.json").is_file())
            self.assertTrue((Path(directory) / "losses.csv").is_file())
            restored, checkpoint = load_neural_astar_checkpoint(
                Path(directory) / "best_checkpoint.pt", model=TinyGuidanceNet()
            )
            self.assertIsInstance(restored, TinyGuidanceNet)
            self.assertEqual(checkpoint["epoch"], 1)


class NeuralAStarVisualizationTests(unittest.TestCase):
    def test_visualization_writes_all_requested_panels(self) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            from visualization.neural_astar import visualize_neural_astar
        except ImportError:
            self.skipTest("Matplotlib is not installed in this environment")
        grid = np.zeros((2, 3), dtype=np.uint8)
        result = run_neural_astar(grid, (0, 0), (1, 2), np.full((2, 3), 0.4))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "neural_astar.png"
            figure, axes = visualize_neural_astar(
                grid, (0, 0), (1, 2), result, output_path=destination
            )
            self.assertTrue(destination.is_file())
            self.assertEqual(axes.size, 6)
            import matplotlib.pyplot as plt
            plt.close(figure)


class NeuralAStarSplitPolicyTests(unittest.TestCase):
    @staticmethod
    def _sample(distribution: str, seed: int):
        return make_dataset_sample(
            MapGenerationConfig(
                height=10,
                width=10,
                distribution=distribution,
                seed=seed,
            )
        )

    def test_train_and_validation_accept_only_independent_d1_to_d3_samples(self) -> None:
        train = self._sample("D1", 1001)
        validation = self._sample("D2", 2002)
        _validate_splits([train], [validation])

    def test_training_policy_rejects_ood_d4_to_d6(self) -> None:
        training = self._sample("D1", 1003)
        ood = self._sample("D4", 2004)
        with self.assertRaisesRegex(ValueError, "uses D4"):
            _validate_splits([ood], [training])

    def test_training_policy_rejects_layout_leakage(self) -> None:
        sample = self._sample("D1", 3005)
        with self.assertRaisesRegex(ValueError, "Duplicate map layout"):
            _validate_splits([sample], [sample])


if __name__ == "__main__":
    unittest.main()
