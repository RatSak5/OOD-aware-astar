# Neural-guidance fixed-K and OOD-adaptive SeeA*

## Method separation

The project now exposes three distinct search APIs:

- `seeastar.uniform.uniform_seeastar` is conventional uniform SeeA*. It
  accumulates the project ground-truth unit transition cost and accepts a
  conventional heuristic.
- `seeastar.run_neural_guidance_seeastar` is fixed-K Neural-guidance SeeA*.
- `adaptive.run_ood_adaptive_neural_guidance_seeastar` is adaptive-K
  Neural-guidance SeeA*.

The corrected fixed and adaptive variants share one implementation in
`seeastar.neural_guidance.run_guidance_uniform_seeastar`. The adaptive wrapper
computes K once and delegates through the fixed-K entry point to that core.
H1/H2/H3 outputs remain conventional heuristic predictions and are not used as
Phi. Original Neural A*, Original SeeA*, and the conventional uniform search
retain their existing implementations and identifiers.

For compatibility, `adaptive.run_ood_adaptive_seeastar` retains its historic
conventional unit-cost behavior. The same callable is explicitly exported as
`adaptive.run_legacy_unit_cost_adaptive_seeastar`. It does not implement the
corrected Neural-guidance method.

## Guidance search

The core takes an obstacle map, start, goal, Neural A* guidance map `Phi`,
configured K, and sampling seed. It validates that `Phi` has the exact map
shape and contains only finite nonnegative values. Values above one are
accepted. Phi is copied without clipping, normalization, or replacement.

The start is initialized with `g(start)=0`; its Phi value is not charged. For
each legal transition from `u` to `v`:

```text
g_candidate(v) = g(u) + Phi(v)
h(v) = ChebyshevDistance(v, goal) + 0.001 * EuclideanDistance(v, goal)
f(v) = g(v) + h(v)
```

A newly discovered node receives its candidate guidance cost and parent. An
OPEN node's cost, parent, and current priority are changed only when the new
candidate cost is strictly smaller. Equal-cost alternatives preserve the
existing parent. CLOSED nodes are excluded from updates and never reopened.
The core stores current priorities directly, so it has no stale priority-queue
entries to expand.

At each expansion, eligible OPEN candidates are sampled uniformly without
replacement, independently of f-values. The sample size is
`min(configured_K, eligible_OPEN_size)`. The sampled candidate with the lowest
current f is selected. The configured K is retained in the result even when
the OPEN set is smaller. The standard-library seeded sampler and stable
coordinate ordering make the draw reproducible.

Movement uses the common eight-connected successor generator, with diagonal
corner-cutting prohibited. Every legal transition costs 1.0 for ground-truth
path measurement, including diagonal transitions. No sqrt(2) transition cost
is used.

## OOD score and adaptive K

The corrected adaptive entry point receives a precomputed raw Mahalanobis
distance from the independent OOD detector, calibrated ID `D95`, the Neural
A* Phi map, K bounds, and a search sampling seed. It does not extract features,
fit a detector, or calibrate D95.

It validates that `D_M` is finite and nonnegative and that `D95` is finite and
strictly positive. It computes:

```text
s = 1 - exp(-D_M / D95)
K = Kmax - floor(s * (Kmax - Kmin))
```

The primary bounds are `Kmin=4`, `Kmax=16`. The continuous score is a
monotonic normalized distance, not a calibrated probability. For finite
distances it is in `[0,1)`, so primary K normally ranges from 5 through 16;
the formula is not altered to force K=4. Floating-point saturation for a very
large finite ratio is represented by the nearest value below one, retaining
that same boundary behavior.

The score and K are calculated once for each adaptive search and K remains
fixed while expanding nodes. OOD affects only K. It does not modify Phi,
guidance accumulation, parent updates, the heuristic, movement, CLOSED
handling, the eligible OPEN population, sample probabilities, or minimum-f
selection.

## Costs and failure records

Successful corrected results expose separate values:

- `found_ground_truth_cost`: count of legal path transitions, each costing
  1.0.
- `found_guidance_cost` / benchmark
  `found_guidance_accumulated_cost`: sum of Phi for each path node except the
  start.

Guidance accumulated cost is neither path length nor ground-truth cost. The
algorithm is not claimed to be Dijkstra-optimal, and the heuristic is not
claimed to be admissible or consistent. When a search fails or is unreachable,
benchmark rows use `None` for both found-path cost fields; low-level result
objects use positive infinity with `path=None` to match their existing search
result convention.

Optional guidance traces record eligible OPEN nodes, sampled nodes, current
sampled f-values, selected node, and the actual expanded node. The goal can be
selected to terminate successfully without being counted as an expansion.
Tracing is off by default in worker runs unless requested.

## Experiment and benchmark wiring

`experiments.run_neural_guidance_seeastar_experiment` accepts cached Phi maps,
independent detector distances, and D95. It performs no inference or fitting,
and passes the same map-level Phi to every paired fixed and adaptive arm. The
older `run_adaptive_seeastar_experiment` remains the conventional-cost
experiment; `run_conventional_adaptive_seeastar_experiment` explicitly names
that baseline.

The benchmark worker identifiers are `neural_guidance_fixed` and
`neural_guidance_adaptive`. Both require the cached Neural A* Phi map. Adaptive
dispatch additionally requires raw `D_M`, calibrated `D95`, Kmin/Kmax, and a
sampling seed; it invokes the corrected adaptive wrapper. The configuration
uses fixed K values 4, 5, 10, and 16 and adaptive bounds 4–16. H1/H2/H3
conventional baselines, Dijkstra, conventional A*, Original Neural A*,
Original SeeA*, and conventional uniform SeeA* remain separately identifiable.

Result rows preserve existing fields and include both path costs for
successful corrected searches. No datasets, checkpoints, saved benchmark
configurations, or experiment results are regenerated by these integrations.
No training, calibration, timing pilot, or full benchmark is claimed here.
