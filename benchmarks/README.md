# Prompt 8 benchmark workflow

The runner dispatches explicit method identifiers. The approved default uses
32 × 32 maps, 300 training maps, 90 validation/calibration maps, 90 ID holdout
maps, 90 OOD test maps, seed 20261106, and five paired sampling seeds. This
document describes planned dispatch; it does not report a pilot or benchmark
run.

## Search methods and dispatch

The benchmark keeps the following methods separate:

- `dijkstra`: ground-truth unit transition costs.
- `astar_zero`, `astar_chebyshev`, `astar_H1`, `astar_H2`, `astar_H3`:
  conventional A* with ground-truth costs and the named heuristic.
- `neural_astar`: Original Neural A* with its own `Phi` and original selection
  procedure.
- `original_seeastar_{uniform,cluster,uct}`: Original SeeA* implementations.
- `uniform_H1/H2/H3_*`: conventional unit-transition-cost uniform SeeA*
  baselines using the named heuristic predictions.
- `neural_guidance_fixed`: corrected fixed-K Neural-guidance SeeA*, configured
  at K = 4, 5, 10, or 16.
- `neural_guidance_adaptive`: corrected adaptive-K Neural-guidance SeeA* with
  primary bounds 4 and 16.

The two `neural_guidance_*` worker arms receive cached Neural A* `Phi` directly.
Adaptive dispatch also receives the independent raw Mahalanobis distance,
calibrated ID `D95`, both K bounds, and the search sampling seed. H1/H2/H3 maps
are never passed as `Phi`. Both corrected arms call the shared guidance-based
core; they do not route through conventional `uniform_seeastar`.

That core sets `g(start)=0`, adds `Phi(v)` when entering successor `v`, and
uses `h(v)=ChebyshevDistance(v,goal)+0.001*EuclideanDistance(v,goal)`. It
uniformly samples `min(K, eligible OPEN size)` candidates without replacement
and independently of their f-values, then expands the sampled candidate with
minimum current `f=g+h`. OPEN state changes only on strict improvement;
CLOSED states are never reopened. Movement is eight-connected, diagonal
corner-cutting is prohibited, and every legal ground-truth transition costs
1.0.

Adaptive dispatch uses `s=1-exp(-D_M/D95)` and
`K=Kmax-floor(s*(Kmax-Kmin))`, computed once per search and held fixed. This
continuous score is a normalized distance, not a probability; for finite
distances it lies in `[0,1)`, so primary adaptive K normally ranges from 5 to
16. OOD changes only K. It does not change guidance, accumulated costs,
heuristics, moves, CLOSED handling, eligible candidates, or sample
probabilities.

Successful corrected searches report `found_ground_truth_cost` (legal
transition count) and `found_guidance_accumulated_cost` (sum of destination
Phi values, excluding the start). These quantities are separate; guidance
cost is not path length and the method is not described as Dijkstra-optimal.
Failed or unreachable searches use `None` for both costs in benchmark rows.
Optional traces record eligible OPEN candidates, sampled candidates and their
current f-values, selected candidates, and the node actually expanded.

`run_ood_adaptive_seeastar` remains the conventional unit-cost adaptive
baseline for compatibility and is also exported as
`run_legacy_unit_cost_adaptive_seeastar`.

## Staged workflow

```powershell
py -3.10 -m benchmarks write-config --output results/prompt8/benchmark_config.json
py -3.10 -m benchmarks prepare-data --config results/prompt8/benchmark_config.json
py -3.10 -m benchmarks train-models --config results/prompt8/benchmark_config.json
py -3.10 -m benchmarks run-pilot --config results/prompt8/benchmark_config.json
```

The pilot is not the full benchmark. It uses three maps from each distribution
and sampling seeds 11 and 23, tests all configured search arms (37 stochastic
arms plus seven deterministic methods, 1,458 searches total under the current
defaults), and freezes the per-search timeout at the configured pilot rule.
Model training/calibration does not load ID holdout or OOD test maps;
evaluation caches are available only after the model-ready gate is written.

After reviewing the pilot report and approving the full run, launch:

```powershell
py -3.10 -m benchmarks run-full --config results/prompt8/benchmark_config.json --confirm-full-run
```

The planned full run has 34,560 searches under the current defaults. The full
command requires a frozen pilot timeout and the explicit confirmation flag. It
is not invoked by `prepare-data`, `train-models`, or `run-pilot`.

## Artifacts

- `results/prompt8/data/`: split NPZs and layout/distribution manifest.
- `results/prompt8/models/`: H1/H2/H3, Neural A*, OOD encoder, Mahalanobis detector, training histories and checkpoints.
- `results/prompt8/evaluation/`: per-map raw predictions, inference timing, heuristic-error metrics and OOD scores.
- `results/prompt8/pilot/`: pilot raw rows and frozen timeout/extrapolation report.
- `results/prompt8/full/`: full raw search rows, map-level paired statistics, OOD/reliability analyses and figures.

Conventional searches use the shared movement and unit transition-cost model.
Corrected Neural-guidance SeeA* uses the same legal moves while accumulating
destination-cell `Phi`; its guidance path cost remains separate from the
ground-truth cost in result rows. Search-worker round-trip time is recorded
apart from each algorithm's own search timer.
