"""OOD-adaptive SeeA* extension with uniform sampling and fixed-per-map K."""

from __future__ import annotations

from dataclasses import dataclass
from math import expm1, floor, isfinite, nextafter
from time import perf_counter
from typing import Any, Literal

import numpy as np

from algorithms.astar import HeuristicFunction, chebyshev_heuristic
from algorithms.movement import DEFAULT_COST_CONFIG, Coord, GroundTruthCostConfig
from ood.feature_extractor import build_ood_input
from ood.mahalanobis import (
    CalibrationError,
    MahalanobisDetector,
    normalized_ood_score,
)
from seeastar.original import SeeAStarResult
from seeastar.uniform import UniformSeeAStarConfig, uniform_seeastar
from seeastar import neural_guidance as neural_guidance_search
from seeastar.neural_guidance import GuidanceSeeAStarResult

try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]


KDirection = Literal["higher_ood_smaller_k", "higher_ood_larger_k"]
KNormalization = Literal["smooth", "clipped"]


@dataclass(frozen=True, slots=True)
class AdaptiveKPolicy:
    """Fixed bounds and direction for one map-level K decision."""

    k_min: int = 4
    k_max: int = 32
    direction: KDirection = "higher_ood_smaller_k"
    normalization: KNormalization = "smooth"

    def __post_init__(self) -> None:
        if (
            isinstance(self.k_min, bool)
            or isinstance(self.k_max, bool)
            or not isinstance(self.k_min, int)
            or not isinstance(self.k_max, int)
        ):
            raise TypeError("K bounds must be integers.")
        if self.k_min < 1 or self.k_max < self.k_min:
            raise ValueError("Require 1 <= k_min <= k_max.")
        if self.direction not in {"higher_ood_smaller_k", "higher_ood_larger_k"}:
            raise ValueError("Unsupported adaptive K direction.")
        if self.normalization not in {"smooth", "clipped"}:
            raise ValueError("normalization must be 'smooth' or 'clipped'.")


@dataclass(frozen=True, slots=True)
class AdaptiveKDecision:
    """Raw detector output and the one K selected before search starts."""

    mahalanobis_distance: float
    smooth_ood_score: float
    policy_score: float
    selected_k: int
    k_min: int
    k_max: int
    direction: KDirection
    normalization: KNormalization


@dataclass(frozen=True, slots=True)
class OODAdaptiveSeeAStarResult:
    """Adaptive decision, uniform SeeA* result, and end-to-end timings."""

    decision: AdaptiveKDecision
    search: SeeAStarResult
    cnn_inference_seconds: float
    ood_inference_seconds: float
    heuristic_inference_seconds: float
    total_end_to_end_seconds: float


@dataclass(frozen=True, slots=True)
class NeuralGuidanceAdaptiveKDecision:
    """One map-level score and K decision for Neural-guidance SeeA*."""

    mahalanobis_distance: float
    d95: float
    ood_score: float
    selected_k: int
    k_min: int
    k_max: int
    direction: str = "higher_ood_smaller_k"
    normalization: str = "smooth_exponential"


@dataclass(frozen=True, slots=True)
class NeuralGuidanceAdaptiveSeeAStarResult:
    """Corrected adaptive search output with both path-cost measures."""

    decision: NeuralGuidanceAdaptiveKDecision
    search: GuidanceSeeAStarResult

    @property
    def selected_k(self) -> int:
        return self.decision.selected_k

    @property
    def ood_score(self) -> float:
        return self.decision.ood_score

    @property
    def found_ground_truth_cost(self) -> float:
        return self.search.found_ground_truth_cost

    @property
    def found_guidance_cost(self) -> float:
        return self.search.found_guidance_cost

    @property
    def path(self) -> tuple[Coord, ...] | None:
        return self.search.path


def clipped_ood_score(distance: float, d95: float, *, minimum_d95: float = 1e-12) -> float:
    """Return the clipped-normalization ablation ``min(D_M/D95, 1)``.

    This is an experiment ablation only; the proposed policy uses the smooth
    exponential score. A non-positive or too-small calibration scale is an
    explicit calibration error.
    """

    if not isfinite(d95) or d95 <= minimum_d95:
        raise CalibrationError(
            f"D95={d95!r} is at or below the safe minimum {minimum_d95:g}."
        )
    if not isfinite(distance) or distance < 0.0:
        raise ValueError("Mahalanobis distance must be finite and non-negative.")
    with np.errstate(over="ignore", divide="raise", invalid="raise"):
        ratio = float(distance) / float(d95)
    return float(min(ratio, 1.0))


def compute_adaptive_k(
    distance: float,
    d95: float,
    policy: AdaptiveKPolicy,
    *,
    minimum_d95: float = 1e-12,
) -> AdaptiveKDecision:
    """Normalize a raw distance and apply the configured integer K rule.

    Proposed direction:
    ``K = K_max - floor(s * (K_max - K_min))``.
    Reverse-direction and clipped-normalization settings are explicitly
    experimental comparisons; they still use uniform candidate sampling.

    With the smooth normalization, every finite Mahalanobis distance yields
    ``s < 1``. Therefore, when ``K_max > K_min``, the floor rule can bring K
    close to but not all the way to ``K_min``. The exact mathematical boundary
    ``s == 1`` maps to ``K_min``; it is not produced by finite-distance smooth
    normalization.
    """

    if not isfinite(distance) or distance < 0.0:
        raise ValueError("Mahalanobis distance must be finite and non-negative.")
    if not isfinite(d95) or d95 <= minimum_d95:
        raise CalibrationError(
            f"D95={d95!r} is at or below the safe minimum {minimum_d95:g}."
        )
    smooth = float(normalized_ood_score(float(distance), float(d95)))
    score = (
        smooth
        if policy.normalization == "smooth"
        else clipped_ood_score(distance, d95, minimum_d95=minimum_d95)
    )
    span = policy.k_max - policy.k_min
    if policy.direction == "higher_ood_smaller_k":
        selected = policy.k_max - floor(score * span)
    else:
        selected = policy.k_min + floor(score * span)
    selected = min(policy.k_max, max(policy.k_min, int(selected)))
    return AdaptiveKDecision(
        mahalanobis_distance=float(distance),
        smooth_ood_score=smooth,
        policy_score=float(score),
        selected_k=selected,
        k_min=policy.k_min,
        k_max=policy.k_max,
        direction=policy.direction,
        normalization=policy.normalization,
    )


def extract_map_embedding(
    feature_extractor: Any,
    grid: np.ndarray,
    start: Coord,
    goal: Coord,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Run one map through the Prompt 5 CNN and return its fixed-size embedding."""

    if torch is None:
        raise ImportError("OOD-adaptive SeeA* requires PyTorch for feature inference.")
    features = build_ood_input(grid, start, goal)
    inputs = torch.as_tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
    model = feature_extractor.to(device)
    model.eval()
    with torch.no_grad():
        embedding = model(inputs)
    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError("Feature extractor must return one embedding per map.")
    values = embedding[0].detach().cpu().numpy().astype(np.float64, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Feature extractor returned a non-finite embedding.")
    return values


def run_ood_adaptive_seeastar(
    grid: np.ndarray,
    start: Coord,
    goal: Coord,
    feature_extractor: Any,
    detector: MahalanobisDetector,
    *,
    policy: AdaptiveKPolicy = AdaptiveKPolicy(),
    heuristic: HeuristicFunction = chebyshev_heuristic,
    seed: int = 0,
    max_expansions: int | None = None,
    heuristic_inference_seconds: float = 0.0,
    cost_config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
    device: str = "cpu",
) -> OODAdaptiveSeeAStarResult:
    """Select K once from the map embedding, then run uniform SeeA* to exit.

    This public entry point never updates K during a search and cannot select a
    cluster or UCT sampler. If fewer than ``K_min`` states are eligible in
    OPEN, the shared uniform sampler uses every available state. An empty OPEN
    set follows SeeA*'s normal ``open_exhausted`` termination. If ``heuristic``
    uses a precomputed CNN map, pass its measured inference duration so the
    reported end-to-end time includes that work. Heuristic calls executed
    inside the search are already included in search time.
    """

    if detector.d95 is None:
        raise CalibrationError("Calibrate the ID-only D95 scale before adaptive search.")
    if (
        detector.num_training_maps < 2
        or not detector.training_fingerprints
        or detector.num_calibration_maps < 1
        or not detector.calibration_fingerprints
    ):
        raise CalibrationError(
            "Adaptive search requires saved train-only statistics and held-out ID D95 provenance."
        )
    if detector.d95 <= detector.config.d95_minimum:
        raise CalibrationError(
            f"D95={detector.d95:g} is at or below the configured safe minimum."
        )
    if not isfinite(heuristic_inference_seconds) or heuristic_inference_seconds < 0.0:
        raise ValueError("heuristic_inference_seconds must be finite and non-negative.")
    total_started = perf_counter()
    cnn_started = perf_counter()
    embedding = extract_map_embedding(
        feature_extractor, grid, start, goal, device=device
    )
    cnn_seconds = perf_counter() - cnn_started
    ood_started = perf_counter()
    distance = float(detector.distance(embedding)[0])
    decision = compute_adaptive_k(
        distance,
        float(detector.d95),
        policy,
        minimum_d95=detector.config.d95_minimum,
    )
    ood_seconds = perf_counter() - ood_started
    search = uniform_seeastar(
        grid,
        start,
        goal,
        heuristic=heuristic,
        config=UniformSeeAStarConfig(
            k=decision.selected_k,
            max_expansions=max_expansions,
            seed=seed,
        ),
        cost_config=cost_config,
    )
    return OODAdaptiveSeeAStarResult(
        decision=decision,
        search=search,
        cnn_inference_seconds=float(cnn_seconds),
        ood_inference_seconds=float(ood_seconds),
        heuristic_inference_seconds=float(heuristic_inference_seconds),
        total_end_to_end_seconds=float(
            heuristic_inference_seconds + perf_counter() - total_started
        ),
    )


def compute_neural_guidance_adaptive_k(
    mahalanobis_distance: float,
    d95: float,
    *,
    k_min: int = 4,
    k_max: int = 16,
) -> NeuralGuidanceAdaptiveKDecision:
    """Apply the approved smooth distance normalization and adaptive-K rule.

    The score is a monotonic normalized distance, not a probability. For very
    large finite ratios, floating-point rounding can produce exactly ``1``;
    it is represented by the nearest smaller float so the specified finite
    distance property ``s < 1`` and its corresponding K behavior are retained.
    """

    if isinstance(mahalanobis_distance, (bool, np.bool_)) or not isfinite(float(mahalanobis_distance)) or float(mahalanobis_distance) < 0.0:
        raise ValueError("Mahalanobis distance must be finite and non-negative.")
    if isinstance(d95, (bool, np.bool_)) or not isfinite(float(d95)) or float(d95) <= 0.0:
        raise CalibrationError("D95 must be finite and strictly positive.")
    if (
        isinstance(k_min, bool)
        or isinstance(k_max, bool)
        or not isinstance(k_min, int)
        or not isinstance(k_max, int)
        or k_min < 1
        or k_max < k_min
    ):
        raise ValueError("K bounds must be integers satisfying 1 <= Kmin <= Kmax.")

    distance = float(mahalanobis_distance)
    scale = float(d95)
    ratio = distance / scale
    score = float(-expm1(-ratio))
    if score >= 1.0:
        score = nextafter(1.0, 0.0)
    if not isfinite(score) or not 0.0 <= score < 1.0:
        raise ValueError("The normalized OOD score must be finite and in [0, 1).")
    selected_k = k_max - floor(score * (k_max - k_min))
    return NeuralGuidanceAdaptiveKDecision(
        mahalanobis_distance=distance,
        d95=scale,
        ood_score=score,
        selected_k=int(selected_k),
        k_min=k_min,
        k_max=k_max,
    )


def run_ood_adaptive_neural_guidance_seeastar(
    grid: np.ndarray,
    start: Coord,
    goal: Coord,
    phi: np.ndarray,
    mahalanobis_distance: float,
    d95: float,
    *,
    k_min: int = 4,
    k_max: int = 16,
    seed: int = 0,
    max_expansions: int | None = None,
    cost_config: GroundTruthCostConfig = DEFAULT_COST_CONFIG,
    capture_trace: bool = False,
) -> NeuralGuidanceAdaptiveSeeAStarResult:
    """Run corrected adaptive-K Neural-guidance SeeA* on one search instance.

    ``phi`` must be the Neural A* Phi map. The supplied, already-calibrated
    Mahalanobis distance and D95 are used to select K exactly once. The search
    then delegates to the same fixed-K API and shared guidance search core.
    OOD information changes only K. Ground-truth and accumulated guidance
    path costs are returned separately by this result.
    """

    decision = compute_neural_guidance_adaptive_k(
        mahalanobis_distance, d95, k_min=k_min, k_max=k_max
    )
    search = neural_guidance_search.run_neural_guidance_seeastar(
        grid,
        start,
        goal,
        phi,
        k=decision.selected_k,
        seed=seed,
        max_expansions=max_expansions,
        cost_config=cost_config,
        capture_trace=capture_trace,
    )
    return NeuralGuidanceAdaptiveSeeAStarResult(decision=decision, search=search)


# Explicit name for the pre-existing, conventional unit-cost adaptive method.
# The historic run_ood_adaptive_seeastar name remains backward compatible.
run_legacy_unit_cost_adaptive_seeastar = run_ood_adaptive_seeastar

