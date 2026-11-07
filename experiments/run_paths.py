"""Helpers for allocating isolated Phase B run directories."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4


def create_run_directory(results_root: str | Path) -> Path:
    """Create and return a fresh run directory beneath ``results_root/runs``.

    A full UUID makes collisions vanishingly unlikely, while ``mkdir`` without
    ``exist_ok`` provides the final atomic uniqueness check. Existing run
    outputs are never reused or overwritten.
    """

    parent = Path(results_root) / "runs"
    parent.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc)
    timestamp = created_at.strftime("%Y%m%dT%H%M%S.%fZ")
    for _ in range(5):
        run_variant = f"{timestamp}_{uuid4().hex}"
        run_directory = parent / run_variant
        try:
            run_directory.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        metadata = {
            "run_variant": run_variant,
            "created_at_utc": created_at.isoformat(),
            "result_root": str(run_directory.resolve()),
        }
        (run_directory / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return run_directory

    raise RuntimeError("Could not allocate a unique Phase B run directory.")


def validate_run_directory(run_directory: str | Path) -> Path:
    """Require an existing directory allocated by :func:`create_run_directory`."""

    path = Path(run_directory).resolve()
    if path.parent.name != "runs":
        raise ValueError("Run outputs must live beneath a dedicated 'runs' directory.")
    metadata_path = path / "run_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"Run metadata is missing from {path}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("run_variant") != path.name:
        raise ValueError(f"Run metadata does not match directory name: {path}.")
    return path
