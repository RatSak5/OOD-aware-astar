"""Run every Kaggle notebook code cell in a patched, CPU-only local sandbox.

Usage: python -B tests/kaggle_notebook_harness.py

Only training, validation/calibration, and ID-pilot NPZ archives are copied.
Dataset generation and OOD model training/calibration are stubbed. The notebook's
real Neural A* lower-level trainer runs for three small-subset updates. With OOD
pilot approval false, the section must print the exact skip message, load no OOD
data, and continue through report, manifest, and archive creation.
"""
from __future__ import annotations
import csv, json, os, shutil, subprocess, sys, tempfile
import io
import zipfile
from contextlib import redirect_stdout
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_experiments_64x64_rectangles.ipynb"
AUTHORIZED = ("training", "validation_calibration", "id_pilot")
LIMITS = {"training": 24, "validation_calibration": 8, "id_pilot": 6}

def _ignore(_directory, names):
    skip = {".git", "results", "tmp", "tests", "__pycache__", ".pytest_cache"}
    return {name for name in names if name in skip or name.endswith((".npz", ".pt", ".pth", ".ckpt", ".pyc", ".zip"))}

def run():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    if not any("GITHUB_REPO_URL" in "".join(c.get("source", [])) for c in nb["cells"]):
        raise RuntimeError("Expected clone-and-run notebook was not found.")
    data_root = ROOT / "data" / "64x64_rectangles_v1"
    for name in AUTHORIZED:
        if not (data_root / f"{name}.npz").is_file():
            raise FileNotFoundError(data_root / f"{name}.npz")
    (ROOT / "tmp").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="notebook_local_", dir=ROOT / "tmp") as raw_tmp:
        tmp = Path(raw_tmp); source_copy = tmp / "source"; shutil.copytree(ROOT, source_copy, ignore=_ignore)
        cwd = tmp / "cwd"; cwd.mkdir()
        old_cwd, old_path, old_run = Path.cwd(), list(sys.path), subprocess.run
        ns = {"__name__": "__main__", "__builtins__": __builtins__, "display": lambda *_a, **_k: None}
        loaded, executed, gate_output = [], [], None

        def fake_git(cmd, capture_output=True, text=True, **kwargs):
            if cmd[0] == "git" and len(cmd) > 1 and cmd[1] == "clone":
                shutil.copytree(source_copy, Path(cmd[3]))
                return subprocess.CompletedProcess(cmd, 0, "", "")
            if cmd[0] == "git" and "rev-parse" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "LOCAL_NOTEBOOK_TEST_COMMIT", "")
            return old_run(cmd, capture_output=capture_output, text=text, **kwargs)

        def install_stubs():
            import data.rectangular_dataset as rectangular_dataset
            import experiments.rectangle_pipeline as pipeline
            import ood.contrastive as contrastive
            import ood.mahalanobis as mahalanobis
            original_load = pipeline.load_samples
            cache = {}
            sources = {name: data_root / f"{name}.npz" for name in AUTHORIZED}

            def tiny_generate(destination):
                destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
                for name, source in sources.items(): shutil.copyfile(source, destination / f"{name}.npz")
                return {"map_count": 5200, "unique_layout_hashes": 5200}

            def small_load(folder, split):
                if split not in AUTHORIZED: raise AssertionError(f"blocked unauthorized split: {split}")
                loaded.append(split)
                if split not in cache: cache[split] = original_load(folder, split)
                by_group = {
                    group: [sample for sample in cache[split] if sample.grid_map.distribution == group]
                    for group in ("D1", "D2", "D3")
                }
                # Keep each tiny subset representative; the notebook's map
                # gallery expects at least one map from all three ID groups.
                selected = [
                    by_group[group][index]
                    for index in range(max(len(values) for values in by_group.values()))
                    for group, values in by_group.items()
                    if index < len(values)
                ][:LIMITS[split]]
                if len(selected) != LIMITS[split]:
                    raise AssertionError(f"could not select {LIMITS[split]} balanced samples from {split}")
                return selected

            rectangular_dataset.generate_dataset = tiny_generate
            pipeline.load_samples = small_load

            def tiny_train(folder, result_root, *, device="cpu", resume=True):
                from neural_astar.training import NeuralAStarTrainingConfig, train_neural_astar
                folder, result_root = Path(folder), Path(result_root)
                train = small_load(folder, "training")[:3]
                validation = small_load(folder, "validation_calibration")[:2]
                config = NeuralAStarTrainingConfig(epochs=1, batch_size=1, learning_rate=0.001,
                                                   seed=20261306, device="cpu")
                _, history = train_neural_astar(train, validation, result_root / "models" / "neural_astar",
                                                config=config, resume=False)
                model_dir = result_root / "models" / "neural_astar"
                complete = json.loads((model_dir / "training_complete.json").read_text())
                ood_dir = result_root / "models" / "ood_feature_extractor"; ood_dir.mkdir(parents=True)
                (ood_dir / "best_checkpoint.pt").write_bytes(b"local test stub")
                (ood_dir / "feature_history.csv").write_text("epoch,train_loss,validation_loss,epoch_seconds\\n1,0.5,0.5,0\\n", encoding="utf-8")
                complete["history_rows"] = len(history)
                complete["configuration"] = {key: getattr(config, key) for key in config.__dataclass_fields__}
                return {"device": "cpu", "training_maps": len(small_load(folder, "training")),
                        "validation_maps": len(small_load(folder, "validation_calibration")),
                        "models": {"NeuralAStar": complete, "OODFeatureExtractor": {"embedding_dim": 128, "history_rows": 1}}}

            def tiny_calibrate(_folder, result_root, *, device="cpu"):
                result_root = Path(result_root); ready = result_root / "models" / "model_ready.json"
                ready.parent.mkdir(parents=True, exist_ok=True)
                ready.write_text(json.dumps({"training_complete": True, "d95_calibrated": True,
                                             "sealed_ood_test_loaded": False}), encoding="utf-8")
                return {"d95": 1.0, "embedding_dim": 128, "training_sample_count": 3,
                        "calibration_sample_count": 2, "ledoit_wolf_shrinkage": 0.2,
                        "ridge_added": 1e-6, "score_checks": {"finite": True}}

            class Detector:
                d95 = 1.0
                def distance(self, _z):
                    import numpy as np
                    return np.asarray([0.5])

            pipeline.train_models = tiny_train
            pipeline.calibrate_detector = tiny_calibrate
            pipeline._extract_embedding = lambda *_a: __import__("numpy").zeros(128)
            contrastive.load_feature_extractor = lambda *_a, **_k: (object(), {})
            mahalanobis.load_detector = lambda *_a, **_k: Detector()

        try:
            os.chdir(cwd); sys.path[:] = [x for x in sys.path if Path(x or ".").resolve() != ROOT]
            subprocess.run = fake_git
            for index, cell in enumerate(nb["cells"]):
                if cell.get("cell_type") != "code": continue
                source = "".join(cell.get("source", []))
                # The checked-in notebook has explicit pilot approval enabled;
                # exercise only its safe skip branch in this local harness.
                source = source.replace(
                    "OOD_PILOT_RELEASE_APPROVED = True",
                    "OOD_PILOT_RELEASE_APPROVED = False",
                )
                if index == 2:
                    source = source.replace('GITHUB_REPO_URL = "https://github.com/RatSak5/OOD-aware-astar.git"',
                                            f"GITHUB_REPO_URL = {str(source_copy)!r}")
                    source = source.replace('GIT_REF = "fix/neural-astar-collapse"  # approved training-fix branch', "GIT_REF = None")
                # Test-only patches: no CUDA and tiny subsets. Keep REQUIRE_LEARNED_PHI=True
                # to verify a failed health check is recorded without aborting later cells.
                if index == 4:
                    source = source.replace(
                        'if not torch.cuda.is_available():\n'
                        '    raise RuntimeError("CUDA is mandatory for training. Enable a GPU accelerator and restart the session.")',
                        '# CUDA is intentionally skipped by the CPU-only harness.',
                    )
                    source = source.replace("if missing:", "if missing and False:")
                if index == 10:
                    source = source.replace(
                        "if not torch.cuda.is_available() or torch.cuda.device_count() < 1:",
                        "if False:  # CPU-only harness patch",
                    )
                    source = source.replace("torch.cuda.set_device(0)", "# skipped by CPU-only harness")
                    source = source.replace(
                        'CUDA_DEVICE = torch.device("cuda:0")',
                        'CUDA_DEVICE = torch.device("cpu")  # harness-only patch',
                    )
                    source = source.replace(
                        'print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| CUDA", torch.version.cuda)',
                        'print("CPU harness:", torch.__version__)',
                    )
                    source = source.replace("FIXED_K = (4, 5, 10, 16)", "FIXED_K = (10,)")
                    source = source.replace("SEEDS = (11, 23, 47, 89, 131)", "SEEDS = (11,)")
                    source = source.replace("BOOTSTRAP_RESAMPLES = 5000", "BOOTSTRAP_RESAMPLES = 20")
                source = source.replace('device="cuda"', 'device="cpu"')
                source = source.replace('training_report.get("device") == "cuda"',
                                        'training_report.get("device") == "cpu"')
                if source.startswith("train_samples = load_samples"):
                    source = source.replace("== (3000, 600, 200)", "== (24, 8, 6)")
                if source.startswith("# Evaluation-only Phi health check"):
                    source = source.replace("health_samples = val_samples[:200]", "health_samples = val_samples[:4]")
                    source = source.replace("if len(health_samples) != 200:", "if len(health_samples) != 4:")
                    source = source.replace('_tensor_inputs(sample, "cuda")', '_tensor_inputs(sample, "cpu")')
                if index == 41:
                    if ns["INTEGRATION_CHECKS"]["maps_checked"] == 200:
                        raise AssertionError("The tiny harness unexpectedly satisfied the full ID-map prerequisite.")
                if index == 41:
                    captured = io.StringIO()
                    with redirect_stdout(captured):
                        exec(compile(source, f"notebook-cell-{index}", "exec"), ns)
                    gate_output = captured.getvalue().strip()
                    expected_skip = "OOD pilot NOT released: skipping this section..."
                    if gate_output != expected_skip:
                        raise AssertionError(f"OOD gate output differed from expected skip: {gate_output!r}")
                else:
                    exec(compile(source, f"notebook-cell-{index}", "exec"), ns)
                executed.append(index)
                if index == 2: install_stubs()
                if index == 10:
                    import torch
                    torch.cuda.synchronize = lambda *_a, **_k: None
            expected_skip = "OOD pilot NOT released: skipping this section..."
            if gate_output != expected_skip:
                raise AssertionError(f"OOD-pilot cell did not print the expected skip message: {gate_output!r}")
            if "ood_pilot" in ns["INFERENCE"] or (Path(ns["DATA_ROOT"]) / "ood_pilot.npz").exists():
                raise AssertionError("OOD pilot data was accessed while its release flag was False.")
            if any(split not in AUTHORIZED for split in loaded): raise AssertionError(f"unauthorized data access: {loaded}")
            result_root = Path(ns["RESULT_ROOT"])
            needed = ("models/neural_astar/step_health.csv", "search_results.csv", "per_map_comparison.csv",
                      "algorithm_comparison.csv", "paired_comparison.csv", "phi_health.json", "report.md",
                      "results_manifest.json", "phase_b_results.zip")
            missing = [name for name in needed if not (result_root / name).is_file()]
            if missing: raise AssertionError(f"missing notebook outputs: {missing}")
            report_text = (result_root / "report.md").read_text(encoding="utf-8")
            if "HEALTH CHECK FAILED" not in report_text:
                raise AssertionError("report.md omitted the failing Phi health banner.")
            health_doc = json.loads((result_root / "phi_health.json").read_text(encoding="utf-8"))
            if not health_doc.get("failed") or "mean" not in health_doc.get("model_loss", {}):
                raise AssertionError("phi_health.json omitted the failed state or learned-model loss.")
            expected_constant_keys = {"0.0", "0.25", "1.0"}
            if set(health_doc.get("constant_phi_control_losses", {})) != expected_constant_keys:
                raise AssertionError("phi_health.json omitted one or more constant-Phi control losses.")
            manifest = json.loads((result_root / "results_manifest.json").read_text(encoding="utf-8"))
            if not {"phi_health.json", "report.md"}.issubset(manifest):
                raise AssertionError("results_manifest.json omitted the health artifact or report.")
            with zipfile.ZipFile(result_root / "phase_b_results.zip") as archive:
                archive_names = set(archive.namelist())
            if not {"phi_health.json", "report.md", "results_manifest.json"}.issubset(archive_names):
                raise AssertionError("phase_b_results.zip omitted the report, manifest, or health artifact.")
            with (result_root / "models" / "neural_astar" / "step_health.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                health_reader = csv.DictReader(stream)
                health_rows = list(health_reader)
                health_columns = set(health_reader.fieldnames or ())
            required_health_columns = {"optimizer_step", "loss", "gradient_norm", "phi_spatial_std_mean", "logit_mean_abs"}
            if not required_health_columns.issubset(health_columns):
                raise AssertionError(f"step-health columns missing: {required_health_columns - health_columns}")
            if {int(float(row["optimizer_step"])) for row in health_rows} != {1, 2, 3}:
                raise AssertionError("step_health.csv must contain every optimizer-step loss")
            with (result_root / "models" / "neural_astar" / "losses.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                loss_columns = next(csv.reader(stream))
            if loss_columns != ["epoch", "train_loss", "validation_loss", "epoch_seconds"]:
                raise AssertionError(f"losses.csv interface changed: {loss_columns}")
            table = ns["RESULTS"]
            algorithms = set(table["algorithm"].astype(str))
            controls = {"seeastar_const_phi1_k10", "seeastar_const_phi025_k10"}
            if not controls.issubset(algorithms): raise AssertionError(f"missing control arms: {controls - algorithms}")
            if set(table["phi_source"].astype(str)) != {"learned", "constant"}: raise AssertionError("phi_source mismatch")
            if sorted(p.name for p in Path(ns["DATA_ROOT"]).iterdir()) != sorted(f"{x}.npz" for x in AUTHORIZED):
                raise AssertionError("data root contains a restricted or missing archive")
            return {"code_cells_executed": len(executed), "ood_gate_output": gate_output,
                    "loaded_splits": sorted(set(loaded)), "verified_outputs": list(needed),
                    "constant_controls": sorted(controls), "phi_source": sorted(set(table["phi_source"].astype(str)))}
        finally:
            subprocess.run = old_run; os.chdir(old_cwd); sys.path[:] = old_path

if __name__ == "__main__": print(json.dumps(run(), indent=2))
