"""Pooled Gaussian Mahalanobis detector with train-only fitting and ID calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from data.dataset import DatasetSample, map_fingerprint
from ood.feature_extractor import OODFeatureExtractor, build_ood_input

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    def tqdm(iterable: Any, **kwargs: Any) -> Any:
        description = kwargs.get("desc", "Embedding maps")
        total = len(iterable)

        def _iterate() -> Any:
            for index, item in enumerate(iterable, start=1):
                yield item
                print(f"\r{description}: {index}/{total}", end="", flush=True)
            if total:
                print()

        return _iterate()


class CalibrationError(ValueError):
    """Raised if held-out ID distances cannot provide a stable D95 scale."""


@dataclass(frozen=True, slots=True)
class MahalanobisConfig:
    """Numerical settings for Ledoit-Wolf shrinkage and D95 calibration."""

    ridge_relative: float = 1e-6
    minimum_ridge: float = 1e-12
    d95_minimum: float = 1e-12
    d95_quantile: float = 0.95

    def __post_init__(self) -> None:
        if not np.isfinite(self.ridge_relative) or self.ridge_relative < 0:
            raise ValueError("ridge_relative must be finite and non-negative.")
        if not np.isfinite(self.minimum_ridge) or self.minimum_ridge <= 0:
            raise ValueError("minimum_ridge must be finite and positive.")
        if not np.isfinite(self.d95_minimum) or self.d95_minimum <= 0:
            raise ValueError("d95_minimum must be finite and positive.")
        if not 0.0 < self.d95_quantile < 1.0:
            raise ValueError("d95_quantile must be between 0 and 1.")


@dataclass(slots=True)
class MahalanobisDetector:
    """Fitted feature distribution and optional validation-derived D95."""

    mean: np.ndarray
    sample_covariance: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    shrinkage: float
    ridge_added: float
    num_training_maps: int
    training_fingerprints: tuple[str, ...]
    config: MahalanobisConfig
    d95: float | None = None
    num_calibration_maps: int = 0
    calibration_fingerprints: tuple[str, ...] = ()

    @property
    def embedding_dim(self) -> int:
        return int(self.mean.shape[0])

    def distance(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute raw, unnormalized Mahalanobis distance for one or many z."""

        values = np.asarray(embeddings, dtype=np.float64)
        was_vector = values.ndim == 1
        if was_vector:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.embedding_dim:
            raise ValueError(f"Expected embeddings shaped (n, {self.embedding_dim}).")
        if not np.isfinite(values).all():
            raise ValueError("Embeddings must contain only finite values.")
        centered = values - self.mean
        quadratic = np.einsum("ni,ij,nj->n", centered, self.precision, centered)
        # Small negative values can arise from floating-point roundoff.
        distances = np.sqrt(np.maximum(quadratic, 0.0))
        return distances[0:1] if was_vector else distances

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Return the continuous score 1-exp(-D_M/D95), never a probability."""

        if self.d95 is None:
            raise CalibrationError("Detector must be calibrated before score() is used.")
        return normalized_ood_score(self.distance(embeddings), self.d95)


def _validate_embeddings(embeddings: np.ndarray, minimum_rows: int = 2) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < minimum_rows or values.shape[1] < 1:
        raise ValueError("Embeddings must be a 2-D array with at least two rows.")
    if not np.isfinite(values).all():
        raise ValueError("Embeddings must contain only finite values.")
    return values


def ledoit_wolf_covariance(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate mean and isotropically shrunk covariance using Ledoit-Wolf.

    Implements the feasible identity-target estimator from Ledoit & Wolf
    (2004), Eq. (14), using covariance normalization 1/n. The returned
    covariance has not yet received the detector's additional numerical ridge.
    """

    values = _validate_embeddings(embeddings)
    mean = values.mean(axis=0, dtype=np.float64)
    centered = values - mean
    count, dimension = centered.shape
    sample = centered.T @ centered / count
    target_scale = float(np.trace(sample) / dimension)
    delta = sample - target_scale * np.eye(dimension, dtype=np.float64)
    delta_squared = float(np.sum(delta * delta))
    if delta_squared <= np.finfo(np.float64).eps * max(target_scale**2 * dimension, 1.0):
        shrinkage = 1.0
    else:
        # Algebraically equals n^-2 * sum_i ||x_i x_i^T - S||_F^2,
        # without allocating an n-by-p-by-p tensor.
        norm_squared = np.einsum("ni,ni->n", centered, centered)
        cross = np.einsum("ni,ij,nj->", centered, sample, centered)
        beta = (
            np.sum(norm_squared * norm_squared)
            - 2.0 * cross
            + count * float(np.sum(sample * sample))
        ) / (count * count)
        beta = max(float(beta), 0.0)
        shrinkage = min(beta, delta_squared) / delta_squared
        shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
    covariance = (1.0 - shrinkage) * sample + shrinkage * target_scale * np.eye(
        dimension, dtype=np.float64
    )
    return mean, covariance, shrinkage


def _embed_samples(
    samples: Sequence[DatasetSample], extractor: Any, *, device: str = "cpu"
) -> np.ndarray:
    if torch is None:
        raise ImportError("Embedding extraction requires PyTorch.")
    if not samples:
        raise ValueError("At least one map is required to extract embeddings.")
    extractor = extractor.to(device)
    extractor.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for sample in tqdm(samples, desc="Extracting map embeddings", leave=False):
            channels = build_ood_input(
                sample.grid_map.grid, sample.grid_map.start, sample.grid_map.goal
            )
            inputs = torch.as_tensor(channels, dtype=torch.float32, device=device).unsqueeze(0)
            embedding = extractor(inputs)
            if embedding.ndim != 2 or embedding.shape[0] != 1:
                raise ValueError("Feature extractor must return one fixed-dimensional vector per map.")
            outputs.append(embedding[0].detach().cpu().numpy().astype(np.float64))
    return _validate_embeddings(np.stack(outputs), minimum_rows=1)


def _require_split(samples: Sequence[DatasetSample], allowed: set[str], label: str) -> tuple[str, ...]:
    if not samples:
        raise ValueError(f"{label} maps cannot be empty.")
    fingerprints: list[str] = []
    seen: set[str] = set()
    for index, sample in enumerate(samples):
        distribution = sample.grid_map.distribution
        if distribution not in allowed:
            raise ValueError(
                f"{label}[{index}] is {distribution}; allowed distributions: {sorted(allowed)}."
            )
        fingerprint = map_fingerprint(sample.grid_map.grid)
        if fingerprint in seen:
            raise ValueError(f"Duplicate map layout within {label} at index {index}.")
        seen.add(fingerprint)
        fingerprints.append(fingerprint)
    present = {sample.grid_map.distribution for sample in samples}
    if present != allowed:
        raise ValueError(
            f"{label} must include every required distribution {sorted(allowed)}; "
            f"received {sorted(present)}."
        )
    return tuple(fingerprints)


def fit_mahalanobis_detector(
    training_samples: Sequence[DatasetSample],
    extractor: OODFeatureExtractor,
    *,
    config: MahalanobisConfig = MahalanobisConfig(),
    device: str = "cpu",
) -> MahalanobisDetector:
    """Fit mean/covariance using only unique D1-D3 training map embeddings."""

    fingerprints = _require_split(training_samples, {"D1", "D2", "D3"}, "training")
    embeddings = _embed_samples(training_samples, extractor, device=device)
    mean, shrunk, shrinkage = ledoit_wolf_covariance(embeddings)
    scale = max(float(np.trace(shrunk) / shrunk.shape[0]), config.minimum_ridge)
    ridge = max(config.minimum_ridge, config.ridge_relative * scale)
    identity = np.eye(shrunk.shape[0], dtype=np.float64)
    added = ridge
    for _ in range(10):
        regularized = shrunk + added * identity
        try:
            np.linalg.cholesky(regularized)
            break
        except np.linalg.LinAlgError:
            added *= 10.0
    else:
        raise np.linalg.LinAlgError("Could not regularize covariance to positive definite form.")
    precision = np.linalg.solve(regularized, identity)
    return MahalanobisDetector(
        mean=mean,
        sample_covariance=(embeddings - mean).T @ (embeddings - mean) / len(embeddings),
        covariance=regularized,
        precision=precision,
        shrinkage=shrinkage,
        ridge_added=added,
        num_training_maps=len(training_samples),
        training_fingerprints=fingerprints,
        config=config,
    )


def calibrate_d95(
    detector: MahalanobisDetector,
    validation_samples: Sequence[DatasetSample],
    extractor: OODFeatureExtractor,
    *,
    device: str = "cpu",
) -> MahalanobisDetector:
    """Set D95 from unique, held-out D1-D3 validation maps only."""

    fingerprints = _require_split(validation_samples, {"D1", "D2", "D3"}, "validation")
    if set(fingerprints) & set(detector.training_fingerprints):
        raise ValueError("Calibration maps overlap training layouts.")
    embeddings = _embed_samples(validation_samples, extractor, device=device)
    distances = detector.distance(embeddings)
    d95 = float(np.quantile(distances, detector.config.d95_quantile, method="linear"))
    if not np.isfinite(d95) or d95 <= detector.config.d95_minimum:
        raise CalibrationError(
            f"Validation D95={d95!r} is at or below minimum "
            f"{detector.config.d95_minimum:g}; feature distances cannot be normalized safely."
        )
    detector.d95 = d95
    detector.num_calibration_maps = len(validation_samples)
    detector.calibration_fingerprints = fingerprints
    return detector


def normalized_ood_score(distances: np.ndarray | float, d95: float) -> np.ndarray:
    """Monotonically normalize distance; output is not a probability."""

    if not np.isfinite(d95) or d95 <= 0:
        raise CalibrationError("D95 must be finite and positive to calculate a score.")
    values = np.asarray(distances, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Mahalanobis distances must be finite and non-negative.")
    with np.errstate(over="ignore", under="ignore", invalid="raise", divide="raise"):
        ratio = values / float(d95)
        scores = -np.expm1(-ratio)
    # Keep all finite-distance outputs strictly below one despite exp underflow.
    return np.minimum(scores, np.nextafter(np.float64(1.0), np.float64(0.0)))


def save_detector(detector: MahalanobisDetector, path: str | Path) -> Path:
    """Save arrays to pickle-free NPZ and estimator/calibration metadata to JSON."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = destination.with_suffix(".json")
    metadata = {
        "schema_version": 1,
        "shrinkage": detector.shrinkage,
        "ridge_added": detector.ridge_added,
        "num_training_maps": detector.num_training_maps,
        "training_fingerprints": list(detector.training_fingerprints),
        "d95": detector.d95,
        "num_calibration_maps": detector.num_calibration_maps,
        "calibration_fingerprints": list(detector.calibration_fingerprints),
        "config": asdict(detector.config),
        "embedding_dim": detector.embedding_dim,
        "estimator": "Ledoit-Wolf identity-target shrinkage plus recorded ridge",
    }
    with destination.open("wb") as stream:
        np.savez_compressed(
            stream,
            mean=detector.mean,
            sample_covariance=detector.sample_covariance,
            covariance=detector.covariance,
            precision=detector.precision,
        )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def load_detector(path: str | Path) -> MahalanobisDetector:
    """Restore a previously saved fitted detector without pickle loading."""

    source = Path(path)
    metadata = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported detector schema version.")
    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return MahalanobisDetector(
        **arrays,
        shrinkage=float(metadata["shrinkage"]),
        ridge_added=float(metadata["ridge_added"]),
        num_training_maps=int(metadata["num_training_maps"]),
        training_fingerprints=tuple(metadata["training_fingerprints"]),
        config=MahalanobisConfig(**metadata["config"]),
        d95=None if metadata["d95"] is None else float(metadata["d95"]),
        num_calibration_maps=int(metadata["num_calibration_maps"]),
        calibration_fingerprints=tuple(metadata["calibration_fingerprints"]),
    )
