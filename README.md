# OOD-Aware Adaptive Heuristic Search

Research code for studying learned guidance, heuristic search, OOD detection,
and selective sampling on reproducible grid-world maps. This repository
currently implements the grid-world foundation, conventional A*, Neural A*
guidance maps, CNN-based conventional heuristic learning, feature-space OOD
detection, faithful SeeA* search variants, and a proposed OOD-adaptive SeeA*
extension that changes K while retaining uniform sampling.

## Ground-truth movement and cost convention

Maps are two-dimensional NumPy arrays with `0` for traversable cells and `1`
for obstacles. Movement is eight-connected. A diagonal is legal only when the
destination and both adjacent orthogonal cells are traversable, so paths cannot
cut blocked corners.

The ground-truth edge cost is **1.0 for every legal move**, orthogonal or
diagonal. This matches the supplied Neural A* paper's grid-world benchmark,
which assigns a unit cost to each entered passable node and infinite cost to an
obstacle. Therefore, Dijkstra distance and optimal path cost are move counts.
The project-specific corner rule changes which transitions exist, not their
cost. This ground-truth cost is separate from Neural A*'s learned per-cell
guidance map and accumulated guidance cost. No `sqrt(2)` diagonal weight is
used here. If physical geometric length is reported later, it must be labeled
separately and computed from horizontal/vertical and diagonal move counts.

`GroundTruthCostConfig` deliberately supports only the documented
`unit_move_count` convention in this stage. All algorithms should obtain
neighbors through `algorithms.movement.get_neighbors` and transition costs
through `get_transition_cost`.

## Conventional A* baseline

`algorithms.astar.astar` accepts a heuristic callable `(node, goal) -> float`.
`zero_heuristic` reproduces Dijkstra's priority ordering. The default
`chebyshev_heuristic` is admissible and consistent for unit-cost eight-way
movement: one legal move changes the L-infinity distance by at most one and
costs one, while obstacles only remove possible routes. Manhattan distance
would overestimate under unit-cost diagonals, and Euclidean distance can also
overestimate (for example, offset `(2, 1)` has Euclidean distance greater than
its two-move unobstructed path), so neither is enabled. Custom learned
predictions can be passed through the callable interface; they do not inherit
the optimality guarantee without a separate proof.

Search counters count unique discovered cells, non-goal expansions, actual
re-expansions, and unique logical OPEN/CLOSED states; stale heap entries are
excluded. Search time includes map and endpoint validation, heuristic
evaluation, queue work, and path reconstruction.

## Neural A* guidance-map method

`neural_astar.search.run_neural_astar` accepts a predicted per-cell `Phi` in
`[0, 1]`. It selects OPEN cells using the paper's normalized exponential of
`-(G_phi + H) / tau`, with `tau=sqrt(width)` by default, and accumulates
`G_phi(next) = G_phi(current) + Phi(next)`. The start begins at zero, the goal's
guidance value is charged on entry, and this guidance sum is not the
ground-truth move count. `H` uses Chebyshev distance plus the paper's
`0.001 * EuclideanDistance` tie-break. The stable selector subtracts the minimum
OPEN priority before exponentiation.

`neural_astar.model.NeuralAStarUNet` predicts `Phi` from two channels:
passability (`1=free`) and the sum of the one-hot start/goal maps. Its
VGG16-block U-Net has the paper's encoder widths and sigmoid output, with
project-local initialization rather than downloaded pretrained weights. The
source equations, paper page references, and all adaptations are recorded in
[the implementation note](reports/neural_astar_implementation_note.md).

`neural_astar.differentiable.differentiable_neural_search` implements the hard
forward search with the paper's identity-backward argmax and Eq. (8) CLOSED
mask. `neural_astar.training.train_neural_astar` trains with the mean-L1 loss
between that mask and the Dijkstra path map, using RMSProp. It accepts D1-D3
training and validation samples and rejects D4-D6. Per-instance searches are
run serially within each mini-batch; their losses are averaged for one update.
Best/last checkpoints, training configuration, and epoch losses are written to
the selected output directory. Load a saved network with
`neural_astar.inference.load_neural_astar_checkpoint` before calling the
inference wrapper.

CNN training and inference require PyTorch. Plot export requires Matplotlib;
training progress uses tqdm when installed. For example, after installing
PyTorch, Matplotlib, and tqdm:

```python
from data.dataset import generate_standard_splits
from data.map_generator import MapGenerationConfig
from neural_astar.training import train_neural_astar, NeuralAStarTrainingConfig

splits = generate_standard_splits(
    train_count=120,
    validation_count=30,
    ood_test_count=30,
    base_seed=20261106,
    map_config=MapGenerationConfig(height=32, width=32),
)
model, history = train_neural_astar(
    splits["train"],
    splits["validation"],
    "checkpoints/neural_astar",
    config=NeuralAStarTrainingConfig(epochs=10, batch_size=2, seed=20261106),
)
```

The OOD test split is deliberately not passed to the training or validation
function. `plan_with_neural_astar` reports CNN inference time separately from
search time and includes both in its total runtime.

## SeeA* selective sampling

`seeastar.original.seeastar` implements the original SeeA* loop: obtain
`D = F(OPEN, K)`, then expand the candidate with minimum `f = g + h`. The
paper defines the loop generically and provides three sampler variants, so the
benchmark reports all three rather than assigning an unsupported single
default:

- `SeeA* (Uniform)` samples K distinct OPEN states uniformly, then ranks them
  by f.
- `SeeA* (Cluster)` uses the paper's online competitive clustering, per-cluster
  sampling, and minimum-f expansion. Because this project has no SeeA* learned
  node embedding, normalized grid coordinates are used as the documented map
  adaptation.
- `SeeA* (UCT)` selects K candidates by the paper's depth-adjusted E score,
  then expands the minimum-f candidate.

`seeastar.uniform.uniform_seeastar` is the separate uniform-only API intended
as the basis for the adaptive extension. It implements Algorithm 2: if OPEN
contains more than K states, it samples K distinct states uniformly without
replacement, then expands the sampled state with lowest f. If OPEN has at most
K states, the entire OPEN set is eligible. Its `UniformSeeAStarConfig` has no
clustering or UCT options.

The paper details, assumptions, equations, sampling rules, and adaptations are
in the [SeeA* implementation note](reports/seeastar_implementation_note.md).
Original and conventional-cost SeeA* methods use the shared eight-connected
successors, prohibit corner-cutting, and charge one ground-truth unit for each
legal transition. Chebyshev distance is the default heuristic for this graph;
a caller-supplied heuristic can be passed to both conventional A* and SeeA*.
The separate Neural-guidance SeeA* methods below use `Phi` as accumulated
search cost, and keep that cost distinct from ground-truth path cost.
Search outputs include sampling time and total runtime, OPEN/CLOSED counters,
the found path and cost, and optimality ratio versus target-directed
zero-heuristic Dijkstra.

```python
from data.dataset import generate_standard_splits
from data.map_generator import MapGenerationConfig
from seeastar import (
    SeeAStarConfig,
    UniformSeeAStarConfig,
    run_search_benchmarks,
    uniform_seeastar,
)

splits = generate_standard_splits(
    train_count=30,
    validation_count=30,
    ood_test_count=30,
    base_seed=20261106,
    map_config=MapGenerationConfig(height=32, width=32),
)
benchmark_maps = [*splits["validation"], *splits["test"]]
results = run_search_benchmarks(
    benchmark_maps,
    "experiments/seeastar",
    seeastar_config=SeeAStarConfig(k=50, seed=20261106),
)

# Uniform-only search API; K remains fixed for this run.
sample = benchmark_maps[0]
uniform_result = uniform_seeastar(
    sample.grid_map.grid,
    sample.grid_map.start,
    sample.grid_map.goal,
    config=UniformSeeAStarConfig(k=50, seed=20261106),
)
```

The runner writes per-map and per-distribution CSVs for Dijkstra, conventional
A*, and the paper's Uniform, Cluster, and UCT SeeA* strategies. SeeA* can return
a non-optimal path with arbitrary heuristics or selective sampling, so the
reported cost ratio is measured against Dijkstra rather than assumed to be 1.

## Neural-guidance fixed-K and adaptive-K SeeA*

`seeastar.run_neural_guidance_seeastar` is the fixed-K method, and
`adaptive.run_ood_adaptive_neural_guidance_seeastar` is its adaptive-K wrapper.
Both use one search core. They receive the Neural A* `Phi` map; H1/H2/H3
remaining-cost predictions stay separate conventional heuristic baselines.

The core initializes `g(start)=0`, does not charge `Phi(start)`, and adds the
destination value on each legal transition:

```text
g_candidate(v) = g(u) + Phi(v)
h(v) = ChebyshevDistance(v, goal) + 0.001 * EuclideanDistance(v, goal)
f(v) = g(v) + h(v)
```

Every expansion uniformly samples `min(K, eligible OPEN size)` candidates
without replacement, independently of their `f` values, then expands the
sampled candidate with minimum current `f`. OPEN parents and guidance costs
change only on strict improvement; equal-cost alternatives preserve the
existing parent, and CLOSED cells are never changed or reopened. Movement uses
the project’s eight-connected, no-corner-cutting rules.

The adaptive wrapper receives an independent Mahalanobis distance and an
already-calibrated ID `D95`. It computes `s = 1 - exp(-D_M / D95)` and
`K = Kmax - floor(s * (Kmax - Kmin))` once for each search, using primary bounds
`Kmin=4`, `Kmax=16`. For finite distances the score is in `[0,1)` and primary
K is 5–16. OOD changes K only; it does not alter `Phi`, cost accumulation,
heuristic, legal moves, CLOSED handling, candidate eligibility, or sampling.
The score is a normalized distance, not a probability.

Successful results report both ground-truth path cost (one per legal
transition, including diagonals) and guidance accumulated cost (sum of
`Phi(v)` for path nodes except the start). Guidance cost is not path length,
and this method is not claimed to be Dijkstra-optimal. Failed searches report
both costs as `None` in benchmark rows.

The earlier `adaptive.run_ood_adaptive_seeastar` and
`experiments.run_adaptive_seeastar_experiment` retain their conventional
unit-transition-cost meaning. The explicit
`run_legacy_unit_cost_adaptive_seeastar` alias makes that baseline clear. The
new `run_neural_guidance_seeastar_experiment` accepts cached `Phi` maps,
Mahalanobis distances, and calibrated `D95`; it performs no inference, fitting,
or calibration. See the [implementation note](reports/ood_adaptive_seeastar_implementation_note.md)
and [benchmark workflow](benchmarks/README.md) for dispatch and reporting.

## CNN conventional heuristic learning

The modules under `models/`, `heuristics/`, and `training/` implement
conventional remaining-cost prediction separately from Neural A*'s `Phi`:

- **H1** trains a Softplus CNN with masked MSE to Dijkstra distance-to-go.
- **H2** is the proposed asymmetric fit plus admissibility, directed-edge
  consistency, and goal penalties. Its starting coefficients are configurable
  and do not guarantee constraints.
- **H3** implements Futuhi and Sturtevant's ordinal CEA loss with integer
  distance classes. The map representation is an adaptation from their PDB
  state classifier; its zero-cost goal class uses the documented finite
  boundary convention in the [heuristic-learning note](<C:/Users/ratul/OneDrive/Documents/AI project/reports/heuristic_learning_implementation_note.md>).

All methods use obstacle, start, and goal channels. Dijkstra targets use the
same unit transition costs as conventional A*. Obstacles and cells unreachable
from the goal are excluded from regression and constraint losses. Raw
predictions are used for admissibility and consistency metrics without clipping
or fallback correction.

The comparison runner trains each method independently on D1-D3, selects its
best checkpoint on the independent D1-D3 validation maps, then reports D4-D6
metrics by distribution. It writes per-epoch loss/violation history CSVs,
checkpoint/config files, validation and OOD metric CSVs, and plots for loss,
prediction error, and both constraint violations. A minimal example is:

```python
from data.dataset import generate_standard_splits
from data.map_generator import MapGenerationConfig
from experiments.heuristic_comparison import run_heuristic_comparison
from training.heuristic_training import HeuristicTrainingConfig
from models.heuristic_cnn import HeuristicCNNConfig

splits = generate_standard_splits(
    train_count=300,
    validation_count=60,
    ood_test_count=90,
    base_seed=20261106,
    map_config=MapGenerationConfig(height=32, width=32),
)
results = run_heuristic_comparison(
    splits["train"],
    splits["validation"],
    splits["test"],
    "experiments/heuristic_comparison",
    base_config=HeuristicTrainingConfig(
        epochs=50,
        batch_size=16,
        seed=20261106,
        model=HeuristicCNNConfig(base_channels=32, depth=4),
    ),
)
```

Use `heuristics.inference.load_heuristic_checkpoint` and
`predict_heuristic_map` to obtain `h_pred`; `fixed_goal_heuristic` adapts the
raw map to the conventional A* callable interface. Learned predictions do not
inherit A* optimality guarantees unless their required properties are
established separately.

## Continuous feature-space OOD detection

`ood.feature_extractor.OODFeatureExtractor` is a CNN independent of the
heuristic prediction head. The optional SimCLR-style training path in
`ood.contrastive` uses horizontal/vertical map flips as graph-preserving views
of D1-D3 training maps. The detector uses the encoder embedding before its
contrastive projection head. Lee et al. use class-conditional Gaussians for
classifier features; this project adapts their feature-distance idea to one
pooled Gaussian over map embeddings. The adaptation and Ledoit-Wolf formula
are detailed in [the OOD implementation note](reports/ood_mahalanobis_implementation_note.md).

The covariance and mean use only D1-D3 training maps, with Ledoit-Wolf
identity-target shrinkage and a separately recorded ridge. `D95` is calibrated
from independent D1-D3 validation maps. The continuous output is raw
`D_M` plus `s = 1 - exp(-D_M / D95)`; `s` is not a probability. OOD test maps
D4-D6 are not used for fitting or calibration. AUROC, trapezoidal AUPR, FPR/FNR
at the explicitly reported `s >= 1-exp(-1)` operating point, score summaries,
and map-level bootstrap confidence intervals are supplementary to the
continuous scores. If H1/H2/H3 models are supplied separately, the evaluator
also measures raw heuristic MAE and its Pearson/Spearman association with
distance and score.

```python
from data.dataset import generate_standard_splits
from data.map_generator import MapGenerationConfig
from experiments.ood_detection import run_ood_detection
from ood.contrastive import ContrastiveTrainingConfig
from ood.feature_extractor import OODFeatureConfig

splits = generate_standard_splits(
    train_count=300,
    validation_count=90,
    ood_test_count=90,
    base_seed=20261106,
    map_config=MapGenerationConfig(height=32, width=32),
)
results = run_ood_detection(
    splits["train"], splits["validation"], splits["test"],
    "experiments/ood_detection",
    training_config=ContrastiveTrainingConfig(
        epochs=40,
        seed=20261106,
        model=OODFeatureConfig(base_channels=32, depth=4),
    ),
)
```

The run writes `feature_extractor.pt`, feature training configuration/history,
`mahalanobis_detector.npz` and its JSON metadata, per-map scores, per-type
summaries, binary metrics, confidence intervals, and plots. Pass a mapping such
as `{"H1": (heuristic_model, "H1")}` with `heuristic_models=` to compute
heuristic-error correlations. Those correlations are empirical and must be
reported from actual evaluation outputs; no OOD-to-error relationship is
assumed.

## Map distributions

| Label | Generation | Default obstacle density |
| --- | --- | ---: |
| D1 | Uniform random obstacles, low density | 0.10 |
| D2 | Uniform random obstacles, medium density | 0.25 |
| D3 | Uniform random obstacles, high density | 0.40 |
| D4 | Compact obstacle clusters | 0.28 |
| D5 | Intersecting corridor network | 0.35 |
| D6 | Mixture of random, clustered, corridor, and room layouts | 0.30 |

All dimensions, seeds, obstacle densities, cluster sizes/counts, corridor
widths/spacing, retry limits, and unreachable-map behavior are configurable in
`MapGenerationConfig`. D4 uses the requested density as an exact obstacle-cell
target; the corridor and room structures are shaped by density but may differ
from it due to geometry and connectivity. Corridor spacing is nominal, with
per-map jitter to produce independent layouts. D1-D3 use an exact obstacle
count rounded from density.

The standard split builder uses D1-D3 for training and independent validation
maps, and D4-D6 for OOD test maps. It fingerprints obstacle layouts without
endpoints and rejects duplicate layouts across all splits. With
`allow_unreachable=False`, generation retries when the sampled start and goal
are disconnected. Set it to `True` only for an experiment that explicitly
includes unreachable endpoint pairs.

## Example

```python
from data.dataset import generate_standard_splits, save_splits_npz
from data.map_generator import MapGenerationConfig

splits = generate_standard_splits(
    train_count=300,
    validation_count=60,
    ood_test_count=90,
    base_seed=20261106,
    map_config=MapGenerationConfig(height=32, width=32),
)
save_splits_npz("datasets", splits)
```

Each compressed NPZ split stores maps, endpoints, distribution labels, actual
seeds, generation configurations, float64 distance maps, optimal path costs,
reachability masks, and the cost-model configuration. Variable-length paths
are stored as flat `optimal_path_coordinates` plus `path_offsets`. Archives use
numeric and Unicode arrays and can be loaded with `allow_pickle=False` via
`data.dataset.load_dataset_npz`.

## Tests

Run the Prompt 1 unit tests with the Python environment used by the project:

```powershell
python -m unittest discover -s tests -v
```

The tests cover neighbor boundaries, obstacle and corner handling, unit edge
costs, Dijkstra distances, conventional A* correctness, Neural A* guidance
validation and cost accumulation, path validity, deterministic map generation,
split leakage, NPZ round trips, Neural A* and CNN heuristic training/inference,
H1/H2/H3 losses, raw constraint evaluation, continuous OOD scores and
correlations, SeeA* sampling and matched baselines, and visualization. CNN
tests require PyTorch.
