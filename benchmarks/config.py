"""Frozen, serializable configuration for the full evaluation pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

from data.map_generator import MapGenerationConfig


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Benchmark, training, and statistical settings approved for Prompt 8.

    The 90-map validation split is reserved for model selection and D95
    calibration. A separate 90-map D1-D3 holdout is used for confirmatory ID
    evaluation so the reported ID performance is not measured on maps used to
    select checkpoints or calibrate the OOD score. The primary fixed-K values
    and primary bounds also configure the distinct Neural-guidance SeeA* arms;
    H1/H2/H3 remain conventional heuristic predictions for their own arms.
    """

    height: int = 32
    width: int = 32
    train_count: int = 300
    validation_count: int = 90
    id_holdout_count: int = 90
    ood_test_count: int = 90
    base_seed: int = 20261106
    sampling_seeds: tuple[int, ...] = (11, 23, 47, 89, 131)
    k_bounds: tuple[tuple[int, int], ...] = ((4, 8), (4, 300), (8, 32))
    primary_k_bounds: tuple[int, int] = (4, 300)
    primary_fixed_ks: tuple[int, ...] = (4, 5, 10, 300)
    original_seeastar_k: int = 16
    primary_heuristic: str = "H1"
    secondary_heuristics: tuple[str, ...] = ("H2", "H3")
    device: str = "cpu"
    cpu_threads: int = 4
    heuristic_epochs: int = 50
    heuristic_batch_size: int = 16
    neural_astar_epochs: int = 10
    neural_astar_batch_size: int = 1
    ood_epochs: int = 40
    ood_batch_size: int = 32
    bootstrap_repetitions: int = 10_000
    permutation_repetitions: int = 100_000
    pilot_maps_per_distribution: int = 3
    pilot_sampling_seeds: tuple[int, ...] = (11, 23)
    timeout_seconds: float | None = None
    output_directory: str = "results/prompt8"

    def __post_init__(self) -> None:
        if min(self.height, self.width) < 2:
            raise ValueError("Map dimensions must both be at least two.")
        if min(
            self.train_count,
            self.validation_count,
            self.id_holdout_count,
            self.ood_test_count,
        ) < 1:
            raise ValueError("Every dataset split must be non-empty.")
        if self.train_count % 3 or self.validation_count % 3:
            raise ValueError("Train and validation counts must divide evenly across D1-D3.")
        if self.id_holdout_count % 3 or self.ood_test_count % 3:
            raise ValueError("Holdout counts must divide evenly across three distributions.")
        if not 0 <= self.base_seed < 2**64:
            raise ValueError("base_seed must be an unsigned 64-bit integer.")
        if not self.sampling_seeds or len(set(self.sampling_seeds)) != len(self.sampling_seeds):
            raise ValueError("Sampling seeds must be a non-empty unique sequence.")
        if any(seed < 0 for seed in self.sampling_seeds):
            raise ValueError("Sampling seeds must be non-negative.")
        if self.primary_heuristic not in {"H1", "H2", "H3"}:
            raise ValueError("primary_heuristic must be H1, H2, or H3.")
        if set(self.secondary_heuristics) - {"H1", "H2", "H3"}:
            raise ValueError("secondary_heuristics may only contain H1, H2, or H3.")
        if self.primary_k_bounds not in self.k_bounds:
            raise ValueError("primary_k_bounds must be included in k_bounds.")
        for bounds in self.k_bounds:
            if (
                len(bounds) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in bounds)
                or bounds[0] < 1
                or bounds[1] < bounds[0]
            ):
                raise ValueError("Every K bound must be an integer pair with 1 <= K_min <= K_max.")
        if len(set(self.k_bounds)) != len(self.k_bounds):
            raise ValueError("k_bounds must not contain duplicate pairs.")
        k_min, k_max = self.primary_k_bounds
        if not 1 <= k_min < k_max:
            raise ValueError("Primary K bounds must satisfy 1 <= K_min < K_max.")
        if self.primary_fixed_ks != tuple(sorted(set(self.primary_fixed_ks))):
            raise ValueError("primary_fixed_ks must be unique and sorted.")
        if any(not k_min <= value <= k_max for value in self.primary_fixed_ks):
            raise ValueError("Every primary fixed K must be within primary bounds.")
        if k_min not in self.primary_fixed_ks or k_min + 1 not in self.primary_fixed_ks:
            raise ValueError("Primary fixed Ks must include K_min and K_min+1.")
        if k_max not in self.primary_fixed_ks:
            raise ValueError("Primary fixed Ks must include K_max.")
        if self.original_seeastar_k < 1:
            raise ValueError("original_seeastar_k must be positive.")
        if self.device != "cpu":
            raise ValueError("The approved environment is CPU-only; device must be 'cpu'.")
        if self.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive.")
        if min(
            self.heuristic_epochs,
            self.heuristic_batch_size,
            self.neural_astar_epochs,
            self.neural_astar_batch_size,
            self.ood_epochs,
            self.ood_batch_size,
            self.bootstrap_repetitions,
            self.permutation_repetitions,
            self.pilot_maps_per_distribution,
        ) < 1:
            raise ValueError("Epochs, batches, repetitions, and pilot map count must be positive.")
        if not self.pilot_sampling_seeds:
            raise ValueError("At least one pilot sampling seed is required.")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when frozen after the pilot.")

    @property
    def map_generation_config(self) -> MapGenerationConfig:
        """Return the shared map template; split generators replace distribution/seed."""

        return MapGenerationConfig(height=self.height, width=self.width)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe settings, including the resolved map template."""

        payload: dict[str, object] = asdict(self)
        payload["map_generation"] = asdict(self.map_generation_config)
        payload["split_counts"] = {
            "train": self.train_count,
            "validation_calibration": self.validation_count,
            "id_holdout": self.id_holdout_count,
            "ood_test": self.ood_test_count,
        }
        payload["smooth_adaptive_k_note"] = (
            "Finite smooth scores stay below 1; for (4,300), adaptive K is 5..300. "
            "K=4 is retained as the nominal lower-bound reference."
        )
        payload["neural_guidance_search"] = {
            "fixed_method_identifier": "neural_guidance_fixed",
            "adaptive_method_identifier": "neural_guidance_adaptive",
            "fixed_k_values": list(self.primary_fixed_ks),
            "adaptive_k_bounds": list(self.primary_k_bounds),
            "phi_source": "Neural A* guidance map",
            "heuristic": "Chebyshev + 0.001 * Euclidean",
        }
        return payload

    def save(self, path: str | Path) -> Path:
        """Write the exact configuration as sorted, readable JSON."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkConfig":
        """Load a saved config, ignoring derived metadata keys."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        field_names = {item.name for item in fields(cls)}
        values = {key: value for key, value in payload.items() if key in field_names}
        for name in (
            "sampling_seeds",
            "k_bounds",
            "primary_k_bounds",
            "primary_fixed_ks",
            "secondary_heuristics",
            "pilot_sampling_seeds",
        ):
            if name in values:
                if name == "k_bounds":
                    values[name] = tuple(tuple(int(value) for value in pair) for pair in values[name])
                else:
                    values[name] = tuple(values[name])
        return cls(**values)
