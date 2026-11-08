import ast
import json
from pathlib import Path
import tempfile
import unittest

from experiments.prepare_kaggle_bundle import _bootstrap_cell, _preflight_cell
from experiments.run_paths import create_run_directory, validate_run_directory


class PhaseBReadinessTests(unittest.TestCase):
    def test_cuda_preflight_requires_only_primary_models(self):
        source = _preflight_cell()
        ast.parse(source)

        self.assertIn('"NeuralAStar"', source)
        self.assertIn('"IndependentOODFeatureExtractor"', source)
        self.assertNotIn("HeuristicCNN", source)
        self.assertNotIn('"H1"', source)
        self.assertNotIn('"H2"', source)
        self.assertNotIn('"H3', source)
        self.assertNotIn("OutOfMemoryError", source)
        self.assertIn('DEVICE_REQUEST != "cuda"', source)

        pipeline = Path("experiments/rectangle_pipeline.py").read_text(encoding="utf-8")
        tree = ast.parse(pipeline)
        train_function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "train_models"
        )
        train_source = ast.get_source_segment(pipeline, train_function)
        self.assertIsNotNone(train_source)
        self.assertNotIn("HeuristicCNN", train_source)
        self.assertNotIn("train_heuristic", train_source)

    def test_neural_astar_pipeline_uses_measured_stable_candidate(self):
        pipeline = Path("experiments/rectangle_pipeline.py").read_text(encoding="utf-8")
        tree = ast.parse(pipeline)
        train_function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "train_models"
        )
        config_call = next(
            node.value
            for node in train_function.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "neural_config"
                for target in node.targets
            )
        )
        self.assertIsInstance(config_call, ast.Call)
        settings = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in config_call.keywords
            if keyword.arg in {"epochs", "batch_size", "learning_rate"}
        }
        self.assertEqual(settings, {"epochs": 10, "batch_size": 1, "learning_rate": 0.0003})

    def test_notebook_bootstrap_allocates_a_unique_run_root(self):
        source = _bootstrap_cell()
        ast.parse(source)

        self.assertIn("from experiments.run_paths import create_run_directory", source)
        self.assertIn("RESULT_ROOT = create_run_directory(RESULTS_ROOT)", source)

    def test_each_run_gets_an_isolated_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            results_root = Path(temporary) / "results"
            first = create_run_directory(results_root)
            second = create_run_directory(results_root)

            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, results_root / "runs")
            self.assertEqual(second.parent, results_root / "runs")
            self.assertTrue(first.is_dir() and second.is_dir())
            first_metadata = json.loads(
                (first / "run_metadata.json").read_text(encoding="utf-8")
            )
            second_metadata = json.loads(
                (second / "run_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(first_metadata["run_variant"], first.name)
            self.assertEqual(second_metadata["run_variant"], second.name)
            self.assertEqual(validate_run_directory(first), first.resolve())


if __name__ == "__main__":
    unittest.main()
