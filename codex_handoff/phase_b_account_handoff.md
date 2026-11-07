# Research Project Handoff — Phase A2 Complete, Phase B Incomplete

## Purpose

This file transfers the essential research context from the previous ChatGPT/Codex account. The goal is to continue the unfinished Phase B work without repeating completed Phase A2 work or overwriting existing artifacts.

**This is a handoff summary, not proof that every proposed Phase B change was implemented. The current files in the project are authoritative. The new Codex session must inspect them before deciding what remains.**

---

## 1. Authoritative project location

Active project folder:

`C:\Users\ratul\OneDrive\Documents\AI_project_backup`

Original project folder:

`C:\Users\ratul\OneDrive\Documents\AI project`

Modify only `AI_project_backup`. Do not modify the original project.

The user already opened a new Codex project with `AI_project_backup` as its primary folder. Verify the active working directory before doing anything.

## 2. Research objective

The project investigates whether adapting the number of uniformly sampled OPEN candidates using a continuous Mahalanobis OOD score improves the efficiency or robustness of SeeA* when accumulated path costs come from Neural A* guidance maps.

The primary experiment uses 64×64 maps with randomly positioned, axis-aligned rectangular obstacles. Rectangles overlap; obstacle density is calculated from the union of occupied cells.

The experiment compares the corrected Neural-guidance SeeA* methods against conventional search baselines on identical maps, endpoints and ground-truth movement rules.

## 3. Phase A2 — completed

Phase A2 was implemented in the backup and reviewed. Its final report stated:

- Active directory was `AI_project_backup`.
- Fixed-K API: `run_neural_guidance_seeastar`.
- Adaptive-K API: `run_ood_adaptive_neural_guidance_seeastar`.
- Both use the shared underlying `run_guidance_uniform_seeastar` search core.
- Benchmark arms use explicit `neural_guidance_fixed` and `neural_guidance_adaptive` identifiers.
- Corrected benchmark arms receive Neural A* Phi; H1/H2/H3 remain separate conventional heuristic baselines.
- The older conventional unit-cost adaptive API remains available separately as `run_legacy_unit_cost_adaptive_seeastar`.
- Successful corrected benchmark rows record `found_ground_truth_cost` and `found_guidance_accumulated_cost` separately.
- New test file: `tests/test_neural_guidance_seeastar.py`.

The reported local test results using Python 3.10.2 were:

- Targeted guidance/adaptive/benchmark tests: 54 passed.
- Existing A*, Dijkstra, Neural A*, SeeA* and OOD tests: 63 passed.
- Full CPU regression suite: **146 passed, 0 failed, 0 skipped**.

Preserve the Phase A2 mathematical specification:

1. `g(start) = 0`.
2. On entering successor `v`, `g_candidate(v) = g(u) + Phi(v)`. The start cell's Phi is not charged.
3. Update an OPEN node's g and parent only on strict improvement. Equal alternatives preserve the existing parent.
4. CLOSED nodes are not updated or reopened.
5. Phi must have the correct shape and finite nonnegative values. Values greater than 1 are allowed; do not clip or normalize.
6. Movement is eight-connected with no diagonal corner-cutting. Every legal transition, including a diagonal, has ground-truth cost 1.0.
7. Modified search heuristic: `h(v) = ChebyshevDistance(v, goal) + 0.001 * EuclideanDistance(v, goal)`; `f(v) = g(v) + h(v)`.
8. Uniformly sample K eligible OPEN candidates without replacement, independently of f, then expand the sampled candidate with minimum current f.
9. Effective K is `min(configured_K, eligible_OPEN_size)`. Do not alter configured K.
10. OOD score is `s = 1 - exp(-D_M / D95)` and adaptive K is `K = Kmax - floor(s * (Kmax - Kmin))`, with primary bounds Kmin=4, Kmax=16. For finite distances primary adaptive K normally ranges 5–16.
11. OOD may affect only K. It must not change Phi, g, parent updates, heuristic, movement, CLOSED handling, eligible OPEN candidates or sampling distribution.
12. For each path, report ground-truth transition cost separately from guidance accumulated cost `sum Phi(v)` excluding start. Do not claim optimality for learned-guidance search without proof.

Do not rewrite or mathematically change Phase A2.

## 4. Dataset — generated and validated

The complete intended dataset has 1,560 maps, all 64×64:

| Split | D1 | D2 | D3 | D4 | D5 | D6 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training | 300 | 300 | 300 | 0 | 0 | 0 | 900 |
| Validation/calibration | 60 | 60 | 60 | 0 | 0 | 0 | 180 |
| ID pilot | 20 | 20 | 20 | 0 | 0 | 0 | 60 |
| ID holdout | 60 | 60 | 60 | 0 | 0 | 0 | 180 |
| OOD pilot | 0 | 0 | 0 | 20 | 20 | 20 | 60 |
| Sealed OOD test | 0 | 0 | 0 | 60 | 60 | 60 | 180 |
| **Total** | **440** | **440** | **440** | **80** | **80** | **80** | **1560** |

Density ranges:

- D1: 10–16% (training/ID)
- D2: 17–23% (training/ID)
- D3: 24–30% (training/ID)
- D4: 3–8% (lower-density OOD)
- D5: 33–37% (higher-density OOD)
- D6: 38–42% (highest-density OOD)

Rectangles have independently sampled integer width and height from 2 to 14 cells, remain within the grid and may overlap. Density is occupied-cell union / 4096. Generation seed base: `20261106`.

Movement and labels use eight-connected movement, no diagonal corner-cutting, unit transition cost including diagonals, float64 Dijkstra distances and explicit reachability masks.

The following three archives independently passed validation:

- `data/64x64_rectangles_v1/training.npz`: 900 maps (300 per D1/D2/D3).
- `data/64x64_rectangles_v1/validation_calibration.npz`: 180 maps (60 per D1/D2/D3).
- `data/64x64_rectangles_v1/id_pilot.npz`: 60 maps (20 per D1/D2/D3).

Validated tensors have shape `(N, 3, 64, 64)` and channel order obstacle occupancy, start one-hot, goal one-hot. The report confirmed map dimensions, endpoints, density metadata, rectangle reconstruction/overlap metadata, paths, ground-truth distance maps and unique obstacle layouts across the 1,140 records.

The ID-holdout and sealed OOD-test archives must not be accessed for the pilot or training. They are reserved for a later final benchmark requiring separate approval.

## 5. OOD pilot handling note

The separate 60-map OOD-pilot archive was accidentally opened for header inspection during an earlier read-only verification, before its intended release gate. The user was informed. According to the report, it was not evaluated, used for training, used for covariance fitting, used for D95 calibration or used in search. This deviation was documented in the OOD bundle README/manifests.

Do not inspect the OOD-pilot archive again during a continuation audit. It may be used only after the intended models, detector and D95 calibration, and ID-only integration checks are complete, with the user's approval as required by the current plan.

## 6. Kaggle bundles and status

Initial private upload ZIP:

`results/64x64_rectangles_v1/64x64_rectangles_v1_kaggle_upload.zip`

Reported SHA-256:

`1bc52b8e783c7d498f6c8c08de30c46bddcbb9eb888342fa843042a273fbfbc5`

It contains training, validation/calibration and ID-pilot data (1,140 records), filtered manifest, source package, notebook, configs and documentation. It excludes OOD pilot, ID holdout and sealed OOD test.

Separate OOD-pilot ZIP:

`results/64x64_rectangles_v1/64x64_rectangles_v1_ood_pilot.zip`

Reported SHA-256:

`4f31c60927a644b1496130b2e59fd0a004e0f687f88516baf8a2a268765bbc1c`

It contains only the OOD-pilot archive and its separate 60-record manifest/metadata. It must remain separate until the release gate.

These hashes describe the bundles at the time they were reported. Recompute and verify hashes if any contents have since changed.

**Kaggle status:** No dataset was uploaded. No Kaggle job was run. The browser automation attempt timed out while attaching to Kaggle, and the signed-in account could not be verified. The intended account is the one associated with `emanarlawar@gmail.com`. The user must authenticate manually; never request, store or enter passwords, OTPs, cookies or API credentials.

## 7. Notebook and files

Existing notebook:

`notebooks/kaggle_experiments_64x64_rectangles.ipynb`

Other relevant files:

- `experiments/rectangle_pipeline.py`
- `benchmarks/rectangle_pilot.py`
- `experiments/prepare_kaggle_bundle.py`
- `results/64x64_rectangles_v1/reports/complete_bundle_manifest.md`
- `results/64x64_rectangles_v1/reports/complete_bundle_manifest.json`
- `results/64x64_rectangles_v1/reports/kaggle_upload_review.md` (if present)

A local correction report stated that the notebook was updated to require only the three initial archives, remove its dataset-generation fallback, enforce CUDA explicitly (`cuda:0`), check forward/backward/optimizer execution on CUDA for five models, and gate OOD-pilot loading. Those checks were static/implemented but not executed on Kaggle.

The latest subsequent readiness audit still found execution blockers in the packaged benchmark code. Therefore, do not assume that all corrections were completed.

## 8. Latest known Phase B blockers

The latest audit before the Codex usage limit reported:

1. `train_models()` trains H1, H2 and H3 for 50 epochs each, even though this experiment only requires Neural A* and the independent OOD encoder as primary training. H1/H2/H3 should be optional secondary baselines, not mandatory dependencies.
2. The pilot runner requires H1/H2/H3 heuristic maps and Original SeeA* inputs. The mandatory experiment must run without these optional models/inputs.
3. Conventional A* with Euclidean heuristic was missing.
4. The pilot did not generate all required deliverables: `search_results.csv`, `per_map_comparison.csv`, `algorithm_comparison.csv`, plots, human-readable report and `phase_b_results.zip`.
5. Raw results omitted important metrics: maximum OPEN/CLOSED sizes, re-expansions, move counts, geometric path length and path-cost ratio.
6. Output used a fixed directory rather than a unique run-specific directory.

A prompt was prepared to fix these issues, but the old Codex account ran out of usage before completion could be confirmed. Some edits may have been saved. The next account must perform a read-only file-level audit to identify the actual state before editing.

## 9. Intended primary experimental comparisons

Mandatory algorithms:

1. Dijkstra (ground-truth reference).
2. Conventional A* with Chebyshev heuristic.
3. Conventional A* with Euclidean heuristic (label as potentially non-admissible because diagonal transitions cost 1.0).
4. Fixed-K Neural-guidance SeeA*, K=4.
5. Fixed-K Neural-guidance SeeA*, K=5.
6. Fixed-K Neural-guidance SeeA*, K=10.
7. Fixed-K Neural-guidance SeeA*, K=16.
8. OOD-adaptive-K Neural-guidance SeeA*.

Original Neural A* and Original SeeA* can be optional if their actual inputs and implementations are available. H1/H2/H3 are optional secondary baselines. Their absence must not block the primary experiment.

Use identical maps, start and goal pairs and the same Phi for fixed-K/adaptive-K comparisons. Use sampling seeds `11, 23, 47, 89, 131` for stochastic algorithms. Map layout is the independent statistical unit; repeated seeds on a map are repeated measurements.

## 10. Required search metrics

For every map and algorithm, collect:

- Expanded nodes (define consistently whether goal counts as expanded).
- Generated unique nodes (re-relaxation is not new generation).
- Re-expansions.
- Maximum OPEN size.
- Maximum CLOSED size.
- Search-only runtime using a monotonic high-resolution timer.
- Configured/effective K and sampling-operation statistics where applicable.
- Success/failure/timeout/termination reason.
- Ground-truth path cost (number of legal transitions, each cost 1.0).
- Dijkstra optimal path cost.
- Absolute path-cost gap.
- Path-cost ratio.
- Number of moves.
- Orthogonal and diagonal move counts.
- Geometric path length (`orthogonal + sqrt(2) * diagonal`) as a separate descriptive metric.
- Guidance accumulated path cost (`sum Phi(v)` excluding start) for modified Neural-guidance SeeA*.

Keep neural inference time, OOD inference time, Mahalanobis calculation time, graph-search-only time and end-to-end time separate. Cache Phi and OOD score per map/problem instance and reuse them across sampling seeds/configurations; account for inference overhead in end-to-end reporting.

Do not claim fewer expansions necessarily means lower runtime. Do not compare Phi pointwise with Dijkstra distances as if they were the same quantity. Do not claim modified guidance-based SeeA* is optimal without proof.

## 11. Training and OOD pipeline

Primary training:

- Neural A*: up to 10 epochs on the 900 D1-D3 training maps.
- Independent OOD feature extractor: up to 40 epochs on D1-D3 training maps.
- H1/H2/H3: optional; do not train by default for this run.

Validation/calibration maps may be used for validation, scheduler decisions and checkpoint selection, but not optimizer gradient-training batches.

Fit the pooled Gaussian from training embeddings only, using the approved Ledoit-Wolf identity-target covariance shrinkage and recorded ridge. Calibrate D95 from validation/calibration embeddings only. Do not use ID pilot or OOD pilot for fitting/calibration.

GPU is mandatory for neural training/inference in Kaggle. Verify CUDA and actual model/batch/forward/backward/optimizer device placement. Never silently fall back to CPU training. H3 memory checks may be relevant only if H3 is actually included as an optional baseline.

## 12. Desired result files

Under `results/64x64_rectangles_v1/`, produce at least:

- `pilot/raw/search_results.csv`
- `pilot/per_map/per_map_comparison.csv`
- `pilot/aggregate/algorithm_comparison.csv`
- Training histories and checkpoints for Neural A* and OOD encoder.
- Detector/calibration artifacts.
- Plots of expansions, generated nodes, runtime, path costs/ratios, success rates, ID/OOD density-stratified results, OOD score correlations and training curves.
- `reports/phase_b_experimental_report.md`
- `phase_b_results.zip`
- `README_results.md`

Save a completed Kaggle notebook version with outputs. Results must be downloadable, not merely printed in notebook output. Use unique run directories; do not overwrite prior runs or Prompt 8 results.

## 13. Approval and safety restrictions

- Do not modify the original project.
- Do not alter Phase A2 mathematical rules.
- Do not regenerate existing datasets.
- Do not modify old checkpoints/histories/results without explicit need and approval.
- Do not access ID-holdout or sealed OOD-test data during training/pilot.
- Keep OOD pilot separate until models, detector, D95 and ID-only integration are finalized.
- Do not upload until account owner and Private visibility can be verified.
- Do not put credentials in prompts or handoff files; user signs in manually.
- If Kaggle browser automation is unsupported, stop and give manual steps rather than repeatedly retrying.
- Do not launch the final 360-map benchmark (180 ID holdout + 180 sealed OOD test) without separate explicit approval.

## 14. Immediate instruction for the new Codex account

First perform a READ-ONLY audit of the current `AI_project_backup` project. Read this handoff file and inspect the actual notebook, `experiments/rectangle_pipeline.py`, `benchmarks/rectangle_pilot.py`, `experiments/prepare_kaggle_bundle.py`, Phase A2 implementations, bundle manifests and results directories.

Reconcile this summary with the actual files. Determine precisely which corrections are complete, incomplete or missing. Do not assume previous proposed changes were completed. Do not repeat Phase A2. Do not modify files, upload to Kaggle, train models, calibrate D95, execute benchmarks or regenerate data during this first audit. Do not access OOD-pilot, ID-holdout or sealed OOD-test archives.

Return a file-level continuation report and wait for the user's approval before making changes.
