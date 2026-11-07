"""Build initial training and separately gated OOD-pilot bundles.

NPZ files are copied and hashed as opaque bytes. Their archive format and
array contents are not inspected or parsed.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import shutil
import zipfile

RUN_ID = "64x64_rectangles_v1"
INITIAL_BUNDLE = f"{RUN_ID}_kaggle_upload.zip"
OOD_BUNDLE = f"{RUN_ID}_ood_pilot.zip"
SOURCE_DIRS = (
    "adaptive", "algorithms", "benchmarks", "data", "experiments",
    "heuristics", "models", "neural_astar", "ood", "seeastar", "training",
)
INITIAL_SPLITS = ("training", "validation_calibration", "id_pilot")
INITIAL_COUNTS = {"training": 900, "validation_calibration": 180, "id_pilot": 60}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(directory: Path) -> dict[str, dict[str, int | str]]:
    return {
        p.relative_to(directory).as_posix(): {
            "bytes": p.stat().st_size,
            "sha256": _sha256(p),
        }
        for p in sorted(directory.rglob("*")) if p.is_file()
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _filter_manifest(path: Path, splits: set[str], scope: str) -> tuple[dict, Counter[str]]:
    # This reads manifest JSON only; it does not inspect an NPZ.
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = [item for item in manifest["maps"] if item.get("split") in splits]
    counts = Counter(str(item["split"]) for item in records)
    if set(counts) != splits:
        raise ValueError(f"Unexpected manifest splits: {dict(counts)}")
    groups = {str(item["distribution"]) for item in records}
    manifest["maps"] = records
    manifest["splits"] = {
        key: value for key, value in manifest["splits"].items() if key in splits
    }
    manifest["density_intervals_percent"] = {
        key: value for key, value in manifest["density_intervals_percent"].items()
        if key in groups
    }
    manifest["total_maps"] = len(records)
    manifest["bundle_scope"] = scope
    return manifest, counts


def _zip_directory(directory: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(directory).as_posix())


def _zip_listing(path: Path) -> list[dict[str, int | str]]:
    # Only the outer bundle directory is read; NPZ member bytes remain opaque.
    with zipfile.ZipFile(path) as archive:
        return [
            {"path": item.filename, "bytes": item.file_size}
            for item in sorted(archive.infolist(), key=lambda x: x.filename)
            if not item.is_dir()
        ]


def _source_inventory(root: Path) -> dict[str, dict[str, int | str]]:
    inventory: dict[str, dict[str, int | str]] = {}
    for dirname in SOURCE_DIRS:
        for path in sorted((root / dirname).rglob("*.py")):
            if "__pycache__" not in path.parts:
                inventory[path.relative_to(root).as_posix()] = {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
    return inventory


def _bootstrap_cell() -> str:
    return '''from pathlib import Path
import hashlib, json, os, sys, zipfile

ON_KAGGLE = Path("/kaggle/working").is_dir()
INPUT_ROOT = Path("/kaggle/input") if ON_KAGGLE else Path.cwd()
WORK_ROOT = Path("/kaggle/working") if ON_KAGGLE else Path.cwd()
PREPARED_ROOT = WORK_ROOT / "phase_b_inputs"
RUN_ID = "64x64_rectangles_v1"
INITIAL_BUNDLE_NAME = "64x64_rectangles_v1_kaggle_upload.zip"
REQUIRED_INITIAL_ARCHIVES = ("training.npz", "validation_calibration.npz", "id_pilot.npz")

def _safe_extract(archive_path, target):
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(archive_path) as bundle:
        for member in bundle.infolist():
            resolved = (target / member.filename).resolve()
            if resolved != root and root not in resolved.parents:
                raise ValueError(f"Unsafe path in input bundle: {member.filename}")
        bundle.extractall(target)

def _has_initial_data(path):
    return (path / "manifest.json").is_file() and all(
        (path / name).is_file() for name in REQUIRED_INITIAL_ARCHIVES
    )

PROJECT_ROOT = None
INITIAL_EXTRACT_ROOT = None
if ON_KAGGLE:
    candidates = sorted(INPUT_ROOT.rglob(INITIAL_BUNDLE_NAME))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Attach exactly one initial Phase B bundle named {INITIAL_BUNDLE_NAME}; "
            f"found {len(candidates)}."
        )
    bundle_path = candidates[0]
    digest = hashlib.sha256()
    with bundle_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    INITIAL_EXTRACT_ROOT = PREPARED_ROOT / f"initial_{digest.hexdigest()[:12]}"
    _safe_extract(bundle_path, INITIAL_EXTRACT_ROOT)
    code_archives = sorted(INITIAL_EXTRACT_ROOT.rglob("project_code.zip"))
    if len(code_archives) != 1:
        raise FileNotFoundError("Initial bundle must contain one project_code.zip.")
    PROJECT_ROOT = INITIAL_EXTRACT_ROOT / "project_code"
    _safe_extract(code_archives[0], PROJECT_ROOT)
else:
    configured_project = os.environ.get("RECTANGLE_PROJECT_ROOT")
    if configured_project:
        candidate = Path(configured_project).resolve()
        if (candidate / "data" / "rectangular_dataset.py").is_file():
            PROJECT_ROOT = candidate
    if PROJECT_ROOT is None:
        for candidate in (Path.cwd(), Path.cwd().parent):
            if (candidate / "data" / "rectangular_dataset.py").is_file():
                PROJECT_ROOT = candidate.resolve()
                break
if PROJECT_ROOT is None:
    raise RuntimeError("Could not locate Phase B project source.")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from experiments.run_paths import create_run_directory

configured_data = os.environ.get("RECTANGLE_DATA_INPUT")
if configured_data:
    DATA_ROOT = Path(configured_data).resolve()
    if not _has_initial_data(DATA_ROOT):
        raise FileNotFoundError(
            "RECTANGLE_DATA_INPUT must contain manifest.json and exactly the required "
            "training, validation_calibration, and id_pilot archives."
        )
elif ON_KAGGLE:
    DATA_ROOT = INITIAL_EXTRACT_ROOT / "dataset"
    if not _has_initial_data(DATA_ROOT):
        missing = [name for name in ("manifest.json",) + REQUIRED_INITIAL_ARCHIVES
                   if not (DATA_ROOT / name).is_file()]
        raise FileNotFoundError("Initial data bundle is incomplete; missing: " + ", ".join(missing))
else:
    candidates = (WORK_ROOT / "data" / RUN_ID, PROJECT_ROOT / "data" / RUN_ID)
    DATA_ROOT = next((p.resolve() for p in candidates if _has_initial_data(p)), None)
    if DATA_ROOT is None:
        raise FileNotFoundError(
            "Missing initial Phase B data: manifest.json, training.npz, "
            "validation_calibration.npz, and id_pilot.npz. Generation fallback is disabled."
        )

RESULTS_ROOT = WORK_ROOT / "results" / RUN_ID
RESULT_ROOT = create_run_directory(RESULTS_ROOT)
RUN_VARIANT = RESULT_ROOT.name
OOD_PILOT_RELEASE_APPROVED = False
DEVICE_REQUEST = "cuda"
print(json.dumps({
    "project_root": str(PROJECT_ROOT),
    "data_root": str(DATA_ROOT),
    "result_root": str(RESULT_ROOT),
    "run_variant": RUN_VARIANT,
    "required_initial_archives": REQUIRED_INITIAL_ARCHIVES,
    "ood_pilot_loaded": False,
    "ood_pilot_release_approved": OOD_PILOT_RELEASE_APPROVED,
    "device_request": DEVICE_REQUEST,
}, indent=2))'''


def _runtime_cell() -> str:
    return '''from experiments.rectangle_pipeline import save_environment
import json
import torch

environment = save_environment(RESULT_ROOT)
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is mandatory for Phase B neural-network training. Enable a Kaggle GPU "
        "accelerator and restart the session; CPU fallback is disabled."
    )
if torch.cuda.device_count() < 1:
    raise RuntimeError("CUDA reported available but no CUDA device is visible.")
CUDA_DEVICE_INDEX = 0
torch.cuda.set_device(CUDA_DEVICE_INDEX)
CUDA_DEVICE = torch.device(f"cuda:{CUDA_DEVICE_INDEX}")
DEVICE_REQUEST = "cuda"
if not all(environment.get(key) for key in ("gpu_model", "cuda_version", "pytorch_version")):
    raise RuntimeError("GPU/CUDA/PyTorch environment metadata is incomplete.")
if not environment.get("cuda_available"):
    raise RuntimeError("Saved environment metadata does not confirm CUDA availability.")
for required in ("manifest.json", "training.npz", "validation_calibration.npz", "id_pilot.npz"):
    if not (DATA_ROOT / required).is_file():
        raise FileNotFoundError(f"Required initial input is missing: {DATA_ROOT / required}")
print(json.dumps({
    "python_version": environment["python_version"],
    "pytorch_version": environment["pytorch_version"],
    "cuda_available": environment["cuda_available"],
    "cuda_version": environment["cuda_version"],
    "gpu_model": environment["gpu_model"],
    "gpu_memory_bytes": environment["gpu_memory_bytes"],
    "cpu_model": environment["cpu_model"],
    "logical_cpu_count": environment["logical_cpu_count"],
    "system_memory": environment["system_memory"],
    "selected_device": str(CUDA_DEVICE),
}, indent=2))'''


def _preflight_cell() -> str:
    return '''import gc
import json
import torch
from neural_astar.model import NeuralAStarUNet
from ood.feature_extractor import OODFeatureExtractor
from ood.contrastive import nt_xent_loss

if not torch.cuda.is_available():
    raise RuntimeError("CUDA became unavailable before model preflight; training is stopped.")
if DEVICE_REQUEST != "cuda" or CUDA_DEVICE.type != "cuda":
    raise RuntimeError("Preflight requires the explicitly selected CUDA device.")
CUDA_PREFLIGHT_RESULTS = []

def _cuda_step(name, model, inputs, optimizer_factory, expected_shapes, loss_fn):
    model = model.to(CUDA_DEVICE)
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters or any(p.device != CUDA_DEVICE for p in parameters):
        raise RuntimeError(f"{name}: model parameters are not all on CUDA.")
    if any(x.device != CUDA_DEVICE for x in inputs):
        raise RuntimeError(f"{name}: input batches are not on CUDA.")
    optimizer = optimizer_factory(model.parameters())
    optimizer.zero_grad(set_to_none=True)
    first = parameters[0]
    before = first.detach().clone()
    outputs, loss = loss_fn(model, inputs)
    values = outputs if isinstance(outputs, (tuple, list)) else (outputs,)
    shapes = []
    for value in values:
        if not torch.is_tensor(value) or value.device != CUDA_DEVICE:
            raise RuntimeError(f"{name}: forward output is not on CUDA.")
        shapes.append(tuple(int(d) for d in value.shape))
    if tuple(shapes) != tuple(expected_shapes):
        raise RuntimeError(f"{name}: unexpected output shapes {shapes}.")
    if not torch.is_tensor(loss) or loss.device != CUDA_DEVICE or loss.ndim != 0:
        raise RuntimeError(f"{name}: loss is not a scalar CUDA tensor.")
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError(f"{name}: loss is non-finite.")
    loss.backward()
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients or any(g.device != CUDA_DEVICE for g in gradients):
        raise RuntimeError(f"{name}: gradients are missing or not on CUDA.")
    if any(not bool(torch.isfinite(g).all().item()) for g in gradients):
        raise RuntimeError(f"{name}: non-finite CUDA gradient.")
    optimizer.step()
    torch.cuda.synchronize(CUDA_DEVICE)
    if torch.equal(before, first.detach()):
        raise RuntimeError(f"{name}: CUDA optimizer step did not change a parameter.")
    states = [v for state in optimizer.state.values() for v in state.values()
              if torch.is_tensor(v)]
    if any(value.device != CUDA_DEVICE for value in states):
        raise RuntimeError(f"{name}: optimizer state is not on CUDA.")
    record = {
        "model": name,
        "device": str(CUDA_DEVICE),
        "input_shapes": [tuple(int(d) for d in x.shape) for x in inputs],
        "output_shapes": shapes,
        "parameters_on_cuda": True,
        "forward_backward_optimizer_on_cuda": True,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(CUDA_DEVICE)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(CUDA_DEVICE)),
    }
    del optimizer, model, inputs, outputs, values, loss, gradients, states, before, first
    gc.collect()
    torch.cuda.empty_cache()
    return record

def _mean_output(model, inputs):
    value = model(inputs[0])
    return value, value.float().mean()

def _contrastive_output(model, inputs):
    first = model.project(model(inputs[0]))
    second = model.project(model(inputs[1]))
    return (first, second), nt_xent_loss(first, second, temperature=0.2)

torch.cuda.reset_peak_memory_stats(CUDA_DEVICE)
x = torch.rand(1, 2, 64, 64, device=CUDA_DEVICE)
CUDA_PREFLIGHT_RESULTS.append(_cuda_step(
    "NeuralAStar", NeuralAStarUNet(), [x],
    lambda p: torch.optim.RMSprop(p, lr=1e-3), ((1, 1, 64, 64),), _mean_output))

torch.cuda.reset_peak_memory_stats(CUDA_DEVICE)
x1 = torch.rand(32, 3, 64, 64, device=CUDA_DEVICE)
x2 = torch.rand(32, 3, 64, 64, device=CUDA_DEVICE)
CUDA_PREFLIGHT_RESULTS.append(_cuda_step(
    "IndependentOODFeatureExtractor", OODFeatureExtractor(), [x1, x2],
    lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=1e-5),
    ((32, 64), (32, 64)), _contrastive_output))

CUDA_PREFLIGHT_COMPLETE = True
print(json.dumps(CUDA_PREFLIGHT_RESULTS, indent=2))'''


def _ood_release_cell() -> str:
    return '''import json, shutil, zipfile
from experiments.rectangle_pipeline import cache_pilot_inference

ready_path = RESULT_ROOT / "models" / "model_ready.json"
integration_path = RESULT_ROOT / "integration_tests" / "inference_to_search.json"
if not ready_path.is_file() or not integration_path.is_file():
    raise PermissionError("OOD release requires completed training, D95 calibration, and ID-only integration.")
ready = json.loads(ready_path.read_text(encoding="utf-8"))
integration = json.loads(integration_path.read_text(encoding="utf-8"))
if not (ready.get("training_complete") and ready.get("d95_calibrated")):
    raise PermissionError("OOD pilot stays locked until training and D95 calibration are complete.")
if integration.get("status") != "passed" or integration.get("ood_pilot_loaded"):
    raise PermissionError("OOD pilot stays locked until ID-only integration passes.")
if not OOD_PILOT_RELEASE_APPROVED:
    raise PermissionError("Explicit approval for the separate OOD-pilot upload is required.")

ood_name = "64x64_rectangles_v1_ood_pilot.zip"
ood_bundles = sorted(INPUT_ROOT.rglob(ood_name))
if len(ood_bundles) != 1:
    raise FileNotFoundError(f"Attach exactly one separately approved {ood_name}; found {len(ood_bundles)}.")
release_root = WORK_ROOT / "phase_b_inputs" / "ood_pilot_release"
release_root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ood_bundles[0]) as bundle:
    members = set(bundle.namelist())
    required = {"dataset/ood_pilot.npz", "dataset/manifest.json",
                "README_ood_pilot.md", "upload_manifest.json"}
    if members != required:
        raise ValueError("Separate OOD-pilot bundle has an unexpected file list.")
    root = release_root.resolve()
    for member in members:
        target = (release_root / member).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Unsafe OOD bundle path: {member}")
    bundle.extractall(release_root)
manifest = json.loads((release_root / "dataset" / "manifest.json").read_text(encoding="utf-8"))
records = manifest.get("maps", [])
if len(records) != 60 or any(row.get("split") != "ood_pilot" for row in records):
    raise ValueError("OOD manifest must contain exactly 60 OOD-pilot records.")
shutil.copyfile(release_root / "dataset" / "ood_pilot.npz", DATA_ROOT / "ood_pilot.npz")
inference_caches = cache_pilot_inference(DATA_ROOT, RESULT_ROOT, device=DEVICE_REQUEST)
print(json.dumps({"ood_pilot_release_gate_passed": True,
                  "ood_pilot_records": len(records),
                  "inference_caches": inference_caches}, indent=2))'''


def _make_initial(root: Path, data_root: Path, notebook_path: Path) -> tuple[Path, dict, Counter[str]]:
    output = root / "results" / RUN_ID / "kaggle_upload"
    data_out = output / "dataset"
    for name in ("dataset", "configs", "notebooks"):
        (output / name).mkdir(parents=True, exist_ok=True)
    obsolete_copy = data_out / "ood_pilot.npz"
    if obsolete_copy.exists():
        obsolete_copy.unlink()
    for filename in ("training.npz", "validation_calibration.npz", "id_pilot.npz"):
        shutil.copyfile(data_root / filename, data_out / filename)
    manifest, counts = _filter_manifest(
        data_root / "manifest.json", set(INITIAL_SPLITS),
        "training, validation_calibration, and ID pilot only",
    )
    if counts != Counter(INITIAL_COUNTS) or len(manifest["maps"]) != 1140:
        raise ValueError(f"Initial manifest counts mismatch: {dict(counts)}")
    _write_json(data_out / "manifest.json", manifest)
    with zipfile.ZipFile(output / "project_code.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for dirname in SOURCE_DIRS:
            for path in sorted((root / dirname).rglob("*.py")):
                if "__pycache__" not in path.parts:
                    archive.write(path, path.relative_to(root).as_posix())
    shutil.copyfile(notebook_path, output / "notebooks" / notebook_path.name)
    config = root / "results" / RUN_ID / "configs" / "adaptive_sensitivity_ablations.json"
    shutil.copyfile(config, output / "configs" / config.name)
    (output / "README_upload.md").write_text(
        f"""# Initial Kaggle training bundle — {RUN_ID}

This bundle contains 1,140 manifest records: training (900), validation/calibration (180), and ID pilot (60). The filtered manifest includes only these splits and D1-D3.

The notebook requires CUDA and checks forward/backward/optimizer execution only for Neural A* and the independent OOD encoder. H1/H2/H3 remain optional secondary baselines. Each notebook startup allocates a unique output directory beneath `results/{RUN_ID}/runs/`; outputs from distinct runs are isolated.

The OOD-pilot archive and per-map records are in a separate local bundle and are excluded here. ID holdout and sealed OOD test archives and per-map records are excluded. NPZs are copied and hashed as opaque bytes; this builder does not parse them.

Proposed Kaggle destination: title 64x64 Rectangles v1 Phase B; slug 64x64-rectangles-v1-phase-b. Not uploaded. Verify owner and private visibility before future upload.

No credentials, checkpoints, Prompt 8 artifacts, or unrelated personal files are packaged.
""", encoding="utf-8")
    inventory = _inventory(output)
    inventory.pop("upload_manifest.json", None)
    _write_json(output / "upload_manifest.json", {
        "run_id": RUN_ID,
        "bundle_type": "initial_training",
        "files": inventory,
        "excluded": ["ood_pilot.npz and its per-map records",
                     "id_holdout.npz and its per-map records",
                     "ood_test_sealed.npz and its per-map records"],
        "npz_handling": "opaque byte copy and SHA-256 only",
    })
    bundle = root / "results" / RUN_ID / INITIAL_BUNDLE
    _zip_directory(output, bundle)
    return bundle, _inventory(output), counts


def _verify_initial_bundle(
    bundle: Path,
    *,
    root: Path,
    expected_files: dict[str, dict[str, int | str]],
    expected_counts: Counter[str],
) -> tuple[list[dict[str, int | str]], dict[str, dict[str, int | str]]]:
    """Verify the rebuilt bundle without parsing any NPZ payloads."""

    source_inventory = _source_inventory(root)
    with zipfile.ZipFile(bundle, "r") as outer:
        names = {item.filename for item in outer.infolist() if not item.is_dir()}
        if names != set(expected_files):
            raise ValueError("Initial bundle members differ from the staged package inventory.")

        npz_members = {
            name for name in names if name.startswith("dataset/") and name.endswith(".npz")
        }
        authorized_npz = {
            "dataset/training.npz",
            "dataset/validation_calibration.npz",
            "dataset/id_pilot.npz",
        }
        if npz_members != authorized_npz:
            raise ValueError(f"Unexpected dataset archives in initial bundle: {sorted(npz_members)}")

        upload_manifest = json.loads(outer.read("upload_manifest.json").decode("utf-8"))
        if upload_manifest.get("bundle_type") != "initial_training":
            raise ValueError("The upload manifest does not describe the initial training bundle.")
        if upload_manifest.get("excluded") != [
            "ood_pilot.npz and its per-map records",
            "id_holdout.npz and its per-map records",
            "ood_test_sealed.npz and its per-map records",
        ]:
            raise ValueError("The upload manifest does not preserve all dataset exclusions.")

        # NPZ members are read only as opaque bytes for copy/hash verification.
        for name in sorted(authorized_npz):
            payload = outer.read(name)
            expected = expected_files[name]
            if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise ValueError(f"Opaque byte/hash verification failed for {name}.")

        packaged_manifest = json.loads(outer.read("dataset/manifest.json").decode("utf-8"))
        records = packaged_manifest.get("maps", [])
        packaged_counts = Counter(str(row.get("split")) for row in records)
        if packaged_counts != expected_counts or len(records) != sum(expected_counts.values()):
            raise ValueError(f"Filtered dataset manifest count mismatch: {dict(packaged_counts)}")
        if set(packaged_counts) != set(INITIAL_SPLITS):
            raise ValueError("Filtered dataset manifest contains an unauthorized split.")

        notebook_path = "notebooks/kaggle_experiments_64x64_rectangles.ipynb"
        notebook_payload = outer.read(notebook_path)
        notebook_expected = expected_files[notebook_path]
        if (
            len(notebook_payload) != notebook_expected["bytes"]
            or hashlib.sha256(notebook_payload).hexdigest() != notebook_expected["sha256"]
        ):
            raise ValueError("Packaged notebook does not match the current local notebook.")

        code_payload = outer.read("project_code.zip")
        with zipfile.ZipFile(BytesIO(code_payload), "r") as source_archive:
            source_names = {
                item.filename for item in source_archive.infolist() if not item.is_dir()
            }
            if source_names != set(source_inventory):
                raise ValueError("Packaged source file list does not match the current project source.")
            for name, expected in source_inventory.items():
                payload = source_archive.read(name)
                if (
                    len(payload) != expected["bytes"]
                    or hashlib.sha256(payload).hexdigest() != expected["sha256"]
                ):
                    raise ValueError(f"Packaged source differs from the live source: {name}")

        upload_files = upload_manifest.get("files", {})
        for name, expected in upload_files.items():
            if name == "upload_manifest.json":
                continue
            payload = outer.read(name)
            if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise ValueError(f"Bundle member hash does not match upload_manifest.json: {name}")

        listing = [
            {"path": item.filename, "bytes": item.file_size}
            for item in sorted(outer.infolist(), key=lambda item: item.filename)
            if not item.is_dir()
        ]
    return listing, source_inventory


def _write_initial_bundle_manifest(
    root: Path,
    bundle: Path,
    files: dict[str, dict[str, int | str]],
    counts: Counter[str],
) -> dict[str, object]:
    listing, sources = _verify_initial_bundle(
        bundle,
        root=root,
        expected_files=files,
        expected_counts=counts,
    )
    record: dict[str, object] = {
        "run_id": RUN_ID,
        "bundle_type": "initial_training",
        "path": str(bundle.resolve()),
        "bytes": bundle.stat().st_size,
        "sha256": _sha256(bundle),
        "manifest_counts": dict(sorted(counts.items())),
        "files": files,
        "source_files": sources,
        "zip_contents": listing,
        "excluded": [
            "ood_pilot.npz and its per-map records",
            "id_holdout.npz and its per-map records",
            "ood_test_sealed.npz and its per-map records",
        ],
        "npz_handling": "opaque byte copy and SHA-256 verification only; no NPZ parsing or array loading",
        "uploaded": False,
        "verification": {
            "only_authorized_npz_members_present": True,
            "filtered_manifest_counts_verified": True,
            "packaged_notebook_matches_live_notebook": True,
            "packaged_python_sources_match_live_sources": True,
            "all_bundle_member_hashes_verified": True,
        },
    }
    reports = root / "results" / RUN_ID / "reports"
    _write_json(reports / "initial_training_bundle_manifest.json", record)
    lines = [
        f"# Initial training bundle manifest — {RUN_ID}",
        "",
        f"- Path: `{bundle}`",
        f"- Bytes: {record['bytes']}",
        f"- SHA-256: `{record['sha256']}`",
        f"- Manifest counts: {dict(sorted(counts.items()))}",
        "- Upload status: not uploaded.",
        "- OOD pilot, ID holdout, and sealed OOD test: excluded.",
        "- NPZ handling: opaque byte copy and hash verification; no arrays loaded.",
        "",
        "## Verification",
        "",
        "- Packaged notebook matches the current local notebook.",
        "- Every packaged Python source file matches the current local source inventory.",
        "- The only packaged NPZ files are training, validation/calibration, and ID pilot.",
        "- Filtered manifest records and all bundle member hashes were checked.",
        "",
        "## Bundle contents",
        "",
        "| Member | Uncompressed bytes |",
        "|---|---:|",
        *[f"| `{item['path']}` | {item['bytes']} |" for item in listing],
        "",
        "## Packaged Python source inventory",
        "",
        "| Source file | Bytes | SHA-256 |",
        "|---|---:|---|",
        *[
            f"| `{name}` | {item['bytes']} | `{item['sha256']}` |"
            for name, item in sorted(sources.items())
        ],
        "",
    ]
    (reports / "initial_training_bundle_manifest.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    historical_json = reports / "complete_bundle_manifest.json"
    historical_markdown = reports / "complete_bundle_manifest.md"
    if historical_json.is_file():
        complete_record = json.loads(historical_json.read_text(encoding="utf-8"))
        complete_record["current_initial_training_bundle"] = {
            "manifest": "initial_training_bundle_manifest.json",
            "path": record["path"],
            "bytes": record["bytes"],
            "sha256": record["sha256"],
            "manifest_counts": record["manifest_counts"],
            "uploaded": False,
        }
        complete_record["current_rebuild_note"] = (
            "The initial bundle and source inventory above are current in "
            "initial_training_bundle_manifest.json. Separate OOD-pilot metadata "
            "was carried forward from the prior report; its archive was not accessed or rebuilt."
        )
        _write_json(historical_json, complete_record)
    if historical_markdown.is_file():
        note = (
            "**Current initial bundle:** use `initial_training_bundle_manifest.md` and `.json` "
            "for the rebuilt initial bundle, current source inventory, and fresh hashes. "
            "The separate OOD-pilot entry below is historical; its archive was not accessed or rebuilt.\n\n"
        )
        prior_text = historical_markdown.read_text(encoding="utf-8")
        if not prior_text.startswith("**Current initial bundle:**"):
            historical_markdown.write_text(note + prior_text, encoding="utf-8")
    (reports / "kaggle_upload_review.md").write_text(
        f"""# Phase B local bundle review — {RUN_ID}

No Kaggle upload or execution occurred.

The rebuilt initial training bundle is `{bundle.name}`, {record['bytes']} bytes, SHA-256 `{record['sha256']}`. It contains 1,140 records: training (900), validation/calibration (180), and ID pilot (60). It excludes OOD pilot, ID holdout, and sealed OOD test data.

The packaged notebook and Python source inventory match the current local files. Opaque NPZ bytes were copied and hashed without parsing or loading arrays. See `initial_training_bundle_manifest.md` and `.json` for the full inventory and verification results.

""",
        encoding="utf-8",
    )
    return record


def _make_ood(root: Path, data_root: Path) -> tuple[Path, dict, Counter[str]]:
    output = root / "results" / RUN_ID / "ood_pilot_bundle"
    data_out = output / "dataset"
    data_out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(data_root / "ood_pilot.npz", data_out / "ood_pilot.npz")
    manifest, counts = _filter_manifest(
        data_root / "manifest.json", {"ood_pilot"}, "separate gated OOD pilot only"
    )
    if counts != Counter({"ood_pilot": 60}) or len(manifest["maps"]) != 60:
        raise ValueError(f"OOD-pilot manifest count mismatch: {dict(counts)}")
    _write_json(data_out / "manifest.json", manifest)
    (output / "README_ood_pilot.md").write_text(
        f"""# Separate OOD-pilot bundle — {RUN_ID}

Contains the OOD-pilot archive and its 60 OOD-pilot-only manifest records. Its raw NPZ bytes were copied and SHA-256 hashed; the NPZ container and arrays were not inspected or parsed in this correction.

Its headers and array shapes were inspected prematurely in an earlier turn. No map evaluation, model training, detector fitting, or D95 calibration used the OOD pilot.

Not uploaded or attached. Release only after training, detector fitting, D95 calibration, and ID-only integration pass, and after explicit approval for the separate upload. ID holdout and sealed OOD test are excluded.
""", encoding="utf-8")
    inventory = _inventory(output)
    inventory.pop("upload_manifest.json", None)
    _write_json(output / "upload_manifest.json", {
        "run_id": RUN_ID,
        "bundle_type": "gated_ood_pilot",
        "files": inventory,
        "excluded": ["training.npz", "validation_calibration.npz", "id_pilot.npz",
                     "id_holdout.npz", "ood_test_sealed.npz"],
        "release_status": "prepared locally; not uploaded or attached",
        "premature_header_inspection": "yes; no evaluation, training, detector fitting, or D95 calibration used it",
        "npz_handling": "opaque byte copy and SHA-256 only",
    })
    bundle = root / "results" / RUN_ID / OOD_BUNDLE
    _zip_directory(output, bundle)
    return bundle, _inventory(output), counts


def _write_manifests(root: Path, initial: Path, initial_files: dict, initial_counts: Counter[str],
                     ood: Path, ood_files: dict, ood_counts: Counter[str]) -> None:
    reports = root / "results" / RUN_ID / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    project_sources = _source_inventory(root)
    initial_record = {
        "path": str(initial), "bytes": initial.stat().st_size, "sha256": _sha256(initial),
        "manifest_counts": dict(sorted(initial_counts.items())), "files": initial_files,
        "zip_contents": _zip_listing(initial),
        "excluded": ["ood_pilot.npz and per-map records", "id_holdout.npz and per-map records",
                     "ood_test_sealed.npz and per-map records"],
        "uploaded": False, "npz_handling": "opaque byte copy and hashing; no NPZ parsing",
    }
    ood_record = {
        "path": str(ood), "bytes": ood.stat().st_size, "sha256": _sha256(ood),
        "manifest_counts": dict(sorted(ood_counts.items())), "files": ood_files,
        "zip_contents": _zip_listing(ood), "uploaded_or_attached": False,
        "premature_header_inspection": (
            "Headers and array shapes inspected prematurely earlier; no evaluation, training, "
            "detector fitting, or D95 calibration used it."
        ),
        "npz_handling": "opaque byte copy and hashing; no NPZ parsing",
    }
    payload = {
        "run_id": RUN_ID, "initial_training_bundle": initial_record,
        "separate_ood_pilot_bundle": ood_record,
        "project_code_sources": project_sources,
        "sealed_ood_test_opened_during_correction": False,
        "kaggle_upload_or_execution_performed": False,
    }
    _write_json(reports / "complete_bundle_manifest.json", payload)
    _write_json(reports / "ood_pilot_bundle_manifest.json", ood_record)
    lines = [
        f"# Phase B bundle manifests — {RUN_ID}", "",
        "NPZ files were copied and hashed as opaque bytes. No NPZ container was inspected or parsed, and no arrays were read in this correction.", "",
        "## Initial training bundle", "",
        f"- Path: {initial}", f"- Bytes: {initial_record['bytes']}",
        f"- SHA-256: {initial_record['sha256']}",
        f"- Manifest counts: {initial_record['manifest_counts']}",
        "- OOD pilot and its per-map records: absent.",
        "- ID holdout and sealed OOD test: absent.", "- Uploaded: no.", "",
        "| File | Bytes | SHA-256 |", "|---|---:|---|",
    ]
    lines += [f"| {name} | {row['bytes']} | {row['sha256']} |"
              for name, row in sorted(initial_files.items())]
    lines += ["", "### Initial ZIP listing", "", "| Member | Uncompressed bytes |", "|---|---:|"]
    lines += [f"| {item['path']} | {item['bytes']} |" for item in initial_record["zip_contents"]]
    lines += ["", "## Separate OOD-pilot bundle", "", f"- Path: {ood}",
              f"- Bytes: {ood_record['bytes']}", f"- SHA-256: {ood_record['sha256']}",
              f"- Manifest counts: {ood_record['manifest_counts']}",
              "- Uploaded or attached: no.",
              "- Premature header inspection: yes; no map evaluation, training, fitting, or calibration used it.",
              "", "| File | Bytes | SHA-256 |", "|---|---:|---|"]
    lines += [f"| {name} | {row['bytes']} | {row['sha256']} |"
              for name, row in sorted(ood_files.items())]
    lines += ["", "### Separate OOD ZIP listing", "", "| Member | Uncompressed bytes |", "|---|---:|---|"]
    lines += [f"| {item['path']} | {item['bytes']} |" for item in ood_record["zip_contents"]]
    lines += ["", "## Nested project source files", "", "| Source file | Bytes | SHA-256 |", "|---|---:|---|"]
    lines += [f"| {name} | {info['bytes']} | {info['sha256']} |"
              for name, info in sorted(project_sources.items())]
    lines += ["", "## Status", "",
              "- Sealed OOD-test archive not opened during this correction.",
              "- No browser action, upload, Kaggle job, training, calibration, integration test, or pilot occurred."]
    (reports / "complete_bundle_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (reports / "ood_pilot_bundle_manifest.md").write_text(
        "\n".join([
            f"# Separate OOD-pilot bundle — {RUN_ID}", "",
            f"- Path: {ood}", f"- Bytes: {ood_record['bytes']}",
            f"- SHA-256: {ood_record['sha256']}",
            f"- Manifest counts: {ood_record['manifest_counts']}",
            "- Not uploaded or attached.",
            "- Prior header inspection was premature; no evaluation, training, detector fitting, or D95 calibration used it.",
            "- NPZ treated as opaque bytes during this correction.", "",
            "| File | Bytes | SHA-256 |", "|---|---:|---|",
            *[f"| {name} | {row['bytes']} | {row['sha256']} |"
              for name, row in sorted(ood_files.items())],
        ]) + "\n", encoding="utf-8",
    )
    (reports / "kaggle_upload_review.md").write_text(
        f"""# Phase B local bundle review — {RUN_ID}

No Kaggle upload or execution occurred.

Initial bundle: {initial.name}, {initial_record['bytes']} bytes, SHA-256 {initial_record['sha256']}; 1,140 records across training (900), validation/calibration (180), and ID pilot (60). It excludes OOD pilot and its per-map records, ID holdout, and sealed OOD test.

Separate OOD pilot bundle: {ood.name}, {ood_record['bytes']} bytes, SHA-256 {ood_record['sha256']}. It remains local and unattached. Headers and array shapes were inspected prematurely earlier; no evaluation, training, detector fitting, or D95 calibration used it.

See complete_bundle_manifest.md for file listings and hashes.
""", encoding="utf-8")


def _update_phase_report(root: Path) -> None:
    path = root / "results" / RUN_ID / "reports" / "phase_b_report.md"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "The uploaded manifest is filtered to 1,200 records from only the four permitted training/calibration/pilot splits. The complete outer and nested file inventory, SHA-256 hashes, and credential/path scan results are in [complete_bundle_manifest.md](complete_bundle_manifest.md) and [complete_bundle_manifest.json](complete_bundle_manifest.json). Kaggle authentication is the only current access blocker. Upload and execution remain gated on your review and approval.",
        "The initial bundle manifest contains 1,140 records from training, validation/calibration, and ID pilot only. OOD pilot is in a separate local bundle and is not uploaded or attached. The complete file inventory and SHA-256 hashes are in [complete_bundle_manifest.md](complete_bundle_manifest.md) and [complete_bundle_manifest.json](complete_bundle_manifest.json). The previous Windows browser attempt was blocked by URL policy enforcement; account and private visibility remain unverified. Upload and execution remain gated on review and approval.",
    )
    text = text.replace(
        "**No pilot has run, so there are no ID/OOD performance metrics to report.**",
        "**No pilot has run, so there are no ID/OOD performance metrics to report. OOD-pilot NPZ headers and array shapes were inspected prematurely earlier; no map evaluation, model training, detector fitting, or D95 calibration used it. This correction treated all NPZ files as opaque bytes.**",
    )
    path.write_text(text, encoding="utf-8")


def _prepare_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    def find(marker: str) -> dict:
        matches = [cell for cell in cells if marker in "".join(cell.get("source", []))]
        if len(matches) != 1:
            raise ValueError(f"Expected one cell for {marker}; found {len(matches)}")
        return matches[0]
    def set_cell(cell: dict, source: str, code: bool = False) -> None:
        cell["source"] = [line + "\n" for line in source.splitlines()]
        if code:
            cell["execution_count"] = None
            cell["outputs"] = []

    set_cell(cells[0], """# 64x64 Rectangular-Obstacle Phase B

Attach the initial private training bundle with training, validation/calibration, and ID-pilot data only. Startup requires these three archives and their filtered manifest. It does not generate data or extract/load OOD-pilot data.

Each notebook startup allocates a unique directory under `results/64x64_rectangles_v1/runs/`. Checkpoints, histories, detector/calibration files, caches, pilot outputs, reports, and ZIPs remain inside that run directory.

After model training, detector fitting, D95 calibration, and ID-only integration pass, obtain separate approval before uploading OOD pilot. Attach it and set OOD_PILOT_RELEASE_APPROVED to True before the gated release cell. ID holdout and sealed OOD test are never included.""")
    set_cell(find("ON_KAGGLE"), _bootstrap_cell(), True)
    set_cell(find("save_environment"), _runtime_cell(), True)
    runtime_heading = next(
        cell for cell in cells
        if "## Runtime and data" in "".join(cell.get("source", []))
        or "## Runtime and initial data" in "".join(cell.get("source", []))
    )
    set_cell(runtime_heading, """## Runtime and initial data

Record runtime details, require CUDA, select cuda:0 explicitly, and check only the three initial archives. Dataset-generation fallback is disabled. OOD pilot is not extracted or loaded here.""")
    train_heading = next(
        cell for cell in cells
        if "## Train 64x64 models" in "".join(cell.get("source", []))
        or "## CUDA preflight and model training" in "".join(cell.get("source", []))
    )
    set_cell(train_heading, """## CUDA preflight and model training

Before training, synthetic inputs verify CUDA model parameters, batches, forward/backward passes, and optimizer updates for the two primary models: Neural A* and the independent OOD encoder. H1, H2, and H3 are optional secondary baselines and are not part of the mandatory preflight.""")
    preflight = next((cell for cell in cells if cell.get("id") == "phaseb-cuda-preflight-v1"), None)
    if preflight is None:
        preflight = {"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": [], "source": [], "id": "phaseb-cuda-preflight-v1"}
        cells.insert(cells.index(train_heading), preflight)
    set_cell(preflight, _preflight_cell(), True)
    set_cell(find("train_models"), """from experiments.rectangle_pipeline import train_models

if not CUDA_PREFLIGHT_COMPLETE or DEVICE_REQUEST != "cuda":
    raise RuntimeError("CUDA preflight must pass before model training.")
training_report = train_models(DATA_ROOT, RESULT_ROOT, device="cuda", resume=True)
if training_report.get("device") != "cuda":
    raise RuntimeError("Training did not report CUDA; stop Phase B.")
print(json.dumps(training_report, indent=2))""", True)
    set_cell(find("calibrate_detector"), """from experiments.rectangle_pipeline import calibrate_detector
calibration_report = calibrate_detector(DATA_ROOT, RESULT_ROOT, device="cuda")
print(json.dumps(calibration_report, indent=2))""", True)
    set_cell(find("run_integration_tests"), """from experiments.rectangle_pipeline import run_integration_tests
integration_report = run_integration_tests(DATA_ROOT, RESULT_ROOT, device="cuda")
print(json.dumps(integration_report, indent=2))""", True)
    set_cell(find("cache_pilot_inference"), _ood_release_cell(), True)
    set_cell(find("run_pilot"), """from benchmarks.rectangle_pilot import run_pilot
pilot_report = run_pilot(DATA_ROOT, RESULT_ROOT, device="cuda")
print(json.dumps(pilot_report, indent=2))""", True)
    set_cell(find("## Stop after the pilot"), """## Stop after the pilot

Review the Phase B report and raw pilot outputs. There is no full-benchmark stage. ID holdout and sealed OOD test require separate explicit approval before final evaluation.""")
    path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_initial(project_root: Path | None = None) -> tuple[Path, dict[str, object]]:
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    data_root = root / "data" / RUN_ID
    notebook = root / "notebooks" / "kaggle_experiments_64x64_rectangles.ipynb"
    _prepare_notebook(notebook)
    initial, initial_files, initial_counts = _make_initial(root, data_root, notebook)
    record = _write_initial_bundle_manifest(root, initial, initial_files, initial_counts)
    return initial, record


def prepare(project_root: Path | None = None, *, include_ood: bool = False) -> tuple[Path, Path | None]:
    root = (project_root or Path(__file__).resolve().parents[1]).resolve()
    initial, initial_record = prepare_initial(root)
    if not include_ood:
        return initial, None
    data_root = root / "data" / RUN_ID
    ood, ood_files, ood_counts = _make_ood(root, data_root)
    # This legacy combined-manifest path is intentionally opt-in because it
    # opens the separate OOD-pilot NPZ for opaque copying and hashing.
    initial_files = initial_record["files"]
    initial_counts = Counter(initial_record["manifest_counts"])
    _write_manifests(root, initial, initial_files, initial_counts, ood, ood_files, ood_counts)
    _update_phase_report(root)
    return initial, ood


if __name__ == "__main__":
    initial_path, manifest = prepare_initial()
    print(initial_path)
    print(manifest["sha256"])
