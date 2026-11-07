# Prompt 8 benchmark plan

## Scope and gating

This is a configuration-driven benchmark of the existing search modules. It calls the existing implementations and does not change search mechanics, the adaptive K formula, or the uniform sampler. The full benchmark is not started by preparation commands; only dataset generation, training/calibration, a representative timing pilot, and analysis validation run before a separate approval.

## Approved data and experimental arms

- Map size: 32 × 32; base seed: `20261106`.
- Training: 300 maps, 100 each from D1–D3.
- Validation/calibration: 90 independent maps, 30 each from D1–D3.
- ID holdout: 90 independent maps, 30 each from D1–D3.
- OOD test: 90 independent maps, 30 each from D4–D6.
- Sampling seeds: 11, 23, 47, 89, 131, paired across stochastic methods.
- Primary bounds: (4,16); fixed K values: 4, 5, 10, 16; smooth proposed, reverse-direction and clipped-score policies.
- H1 additionally uses bounds (4,8) and (8,32), with fixed minimum, midpoint and maximum references and smooth adaptation. H2 and H3 are sensitivity analyses at primary bounds.
- Corrected fixed-K Neural-guidance SeeA* arms use the Neural A* Phi map at K = 4, 5, 10, and 16. The corrected adaptive-K Neural-guidance arm uses the same Phi-based core and bounds (4,16).
- Conventional `uniform_H1/H2/H3_*` unit-transition-cost SeeA* baselines remain separately labelled and do not represent the corrected Neural-guidance algorithms.
- Original SeeA* strategy-specific controls (uniform, cluster, UCT), Original Neural A*, Dijkstra and conventional A* remain separately labelled.

The full planned search count is 34,560: 180 maps × 37 stochastic arms × five sampling seeds, plus 180 maps × seven deterministic methods. The pilot uses three maps from each of D1–D6 and seeds 11 and 23 for sampled methods (1,458 searches). These are planned counts, not evidence that the runs have been performed.

## Split protection

Obstacle-layout fingerprints, which ignore endpoint coordinates, are checked across train, validation/calibration, ID holdout, and OOD test. Model training and D95 calibration read only train and validation/calibration. The ID holdout and OOD archive loaders require a model-ready manifest before releasing evaluation maps. Map-generation structural diagnostics are computed to verify that distributions differ; they do not tune models or adaptive policies.

## Models and inference

The existing H1, H2, H3, Neural A*, independent contrastive feature encoder and train-only Ledoit–Wolf Mahalanobis detector are trained with their configured defaults. No epoch counts or model settings are reduced silently. A cached per-map inference stage records raw H1–H3 predictions, Neural A* guidance maps, raw Mahalanobis distances, smooth scores, and per-model inference times. H1 is the primary OOD-reliability hypothesis; H2/H3 are secondary sensitivities. D95 remains the 95th percentile of the independent ID validation/calibration distances.

## Search accounting

The worker dispatches to Dijkstra, conventional A*, Original Neural A*,
Original SeeA*, conventional unit-cost uniform SeeA*, and corrected
Neural-guidance SeeA* through explicit method identifiers. The corrected
fixed-K and adaptive-K arms both receive the cached Neural A* Phi map. Adaptive
dispatch also receives raw Mahalanobis distance, calibrated ID D95, Kmin/Kmax,
and the search sampling seed; it computes K once for that search and invokes
the same underlying guidance-based search core as fixed-K. H1/H2/H3 prediction
maps are used only by conventional heuristic baselines.

The corrected core uses `g(start)=0` and `g_candidate(v)=g(u)+Phi(v)`, with
`h(v)=ChebyshevDistance(v,goal)+0.001*EuclideanDistance(v,goal)`. It uniformly
samples `min(configured_K, eligible_OPEN_size)` nodes without replacement and
independently of f-values, then selects the sampled candidate with the minimum
current f. OPEN state changes only on strict guidance-cost improvement, and
CLOSED states are not reopened. The adaptive calculation is
`s=1-exp(-D_M/D95)` followed by
`K=Kmax-floor(s*(Kmax-Kmin))`; OOD changes K only. Movement uses the common
eight-connected graph with no diagonal corner-cutting and unit ground-truth
cost per legal transition.

Successful corrected-search rows separately store ground-truth transition
count and guidance accumulated cost (destination-cell Phi sum excluding the
start). Failed or unreachable searches record `None` for both found-path
costs. Guidance accumulated cost is not path length, and no Dijkstra-optimality
claim is made for Neural-guidance SeeA*.

Original Neural A* tracing wraps the same selector and neighbor helpers, returns their original values, and then runs an uninstrumented reference search. Any mismatch in selected order, final path, guidance map, accumulated guidance costs, goal guidance cost, or expanded mask becomes an error record; metrics from that run are unavailable. Timings use the uninstrumented pass. Corrected Neural-guidance SeeA* can optionally record each eligible OPEN set, uniform sample, sampled current f-values, selected candidate, and actual expanded node; worker tracing is off by default to avoid unnecessary result size and timing overhead.

## Statistical analysis

The map is the independent unit. The five sampling repetitions are retained as raw rows and averaged within each map before paired inference. Pairwise comparisons use map-matched differences, a percentile bootstrap confidence interval, a two-sided sign-flip test, and signed rank-biserial effect size. Holm correction is applied within metric families. Success and timeout rates are reported separately from runtime; runtime comparisons use finite paired records and show the number of maps contributing.

## Timing pilot and timeout policy

The pilot spans all six distributions and all configured search arms. A hard timeout is frozen from pilot search-time p99 as `max(60 seconds, 10 × p99)`, rounded upward to a 10-second boundary. The full-run configuration cannot run without this frozen timeout. Pilot failures remain in the raw CSV. The pilot summary extrapolates per-method search time and uses cached per-map inference timing; model loading is separately recorded.

## Confounds and limits

- CPU-only execution with four configured threads and no CUDA; timings may be longer than a GPU setup.
- Search-worker serialization/round-trip time is recorded separately from each algorithm's internal runtime.
- Learned H1/H2/H3 predictions need not be admissible or consistent. Raw violation rates are computed on all valid reachable cells and directed legal edges.
- Mahalanobis score is a normalized distance, not a probability or established predictor of heuristic error. Correlation analyses are direct tests of that hypothesis.
- SeeA* variants may return suboptimal paths; path costs are remeasured under the same unit transition model, and optimality gap is checked against Dijkstra ground truth.
- Neural-guidance SeeA* also reports its separate accumulated Phi cost; that value is not compared as a ground-truth path length.
- KeeA* uses heuristic-informed, path-aware sampling and cluster scouting; this project keeps the sampling distribution uniform and adapts only K. It is included as related work, but is not a direct baseline because changing the sampling distribution would confound the K-only comparison and the published evaluations target synthesis/retrosynthesis search tasks.

## Related work: KeeA*

Zhao, Dengwei; Tu, Shikui; Sun, Yanan; and Xu, Lei. “KeeA*: Epistemic Exploratory A* Search via Knowledge Calibration.” *Advances in Neural Information Processing Systems 38* (2025), DOI 10.52202/085713-1936. The paper's exploratory mechanism uses heuristic-related knowledge calibration, cluster scouting, and path-aware candidate selection. The present proposal instead draws candidates uniformly and changes only the sample count K based on a continuous Mahalanobis score. This is a related exploration-and-search method, not a direct experimental baseline for isolating the effect of adaptive K.

Sources: [NeurIPS 2025 proceedings record](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5393eaf0b723278738788520ec54e4b4-Abstract-Conference.html), [paper PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/5393eaf0b723278738788520ec54e4b4-Paper-Conference.pdf).
