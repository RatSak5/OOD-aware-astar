# Phase B report — `64x64_rectangles_v1`

**Status: dataset and local preparation complete; Kaggle training, detector calibration, end-to-end integration, and pilot are pending Kaggle authentication.** The final benchmark was not started.

The working directory was `C:\Users\ratul\OneDrive\Documents\AI_project_backup`, as requested. The similarly named original project at `C:\Users\ratul\OneDrive\Documents\AI project` was not used. Existing Prompt 8 datasets, checkpoints, reports, benchmark configurations, and Phase A2 implementation files were left in place.

## A. Environment

The local environment record is saved in [`environment.json`](../environment.json). It reports Python 3.10.2, PyTorch 2.14.1+cpu, NumPy 1.24.1, Windows 10, 4 logical CPUs, about 8.4 GB total RAM with about 1.26 GB available at capture time, and no CUDA/GPU. The Kaggle environment was not launched or measured: Kaggle opened signed out, Kaggle CLI is not installed, and neither `KAGGLE_USERNAME` nor `KAGGLE_KEY` is configured.

The prepared notebook and upload package are ready for an authenticated Kaggle session. A local bootstrap simulation passed before the review gate; the current filtered-manifest package has not been uploaded or executed. Current Kaggle documentation describes CPU/GPU notebook sessions up to 12 hours, 20 GB saved working output, and a variable weekly GPU quota; these are platform limits, not measurements or guarantees for this account ([Kaggle Notebooks](https://www.kaggle.com/docs/notebooks), [Efficient GPU Usage](https://www.kaggle.com/docs/efficient-gpu-usage)). The local machine is not suitable for the planned training run with no GPU and low free RAM.

## B. Dataset generation and verification

The new generator is in [`data/rectangular_dataset.py`](../../../data/rectangular_dataset.py). It uses the base seed `20261106`; each split/density group has a separately derived and recorded seed, and each map stores its own generation seed. Rectangle width and height are sampled independently and uniformly from integers 2–14. Positions are sampled uniformly from valid top-left coordinates, rectangles may overlap, and density is calculated from the occupied-cell union. All six density groups use the same generator and endpoint/movement rules.

The canonical input is a `uint8` tensor with shape `3×64×64`, in channel order obstacle occupancy, start one-hot, goal one-hot. Density is metadata, not a model input. Endpoint selection chooses a legally connected free-space component, weights eligible components by size, chooses an axis with span at least 24, and samples endpoints from opposite extreme edges; this same rule is used for D1–D6. Ground-truth distance maps use float64 Dijkstra costs, every legal move costs 1.0, diagonal corner cutting is prohibited, and unreachable cells are represented by infinity plus an explicit reachability mask.

| Density group | Assigned interval | Maps | Actual min–max % | Mean % |
|---|---:|---:|---:|---:|
| D1 | 10–16% | 440 | 10.0098–14.5996 | 11.0711 |
| D2 | 17–23% | 440 | 17.0166–21.3867 | 17.9410 |
| D3 | 24–30% | 440 | 24.0234–28.1006 | 24.9785 |
| D4 | 3–8% | 80 | 3.0029–7.3975 | 4.2380 |
| D5 | 33–37% | 80 | 33.0078–35.9375 | 33.8004 |
| D6 | 38–42% | 80 | 38.0127–41.5283 | 38.8623 |

These ranges and means are density summaries for all splits; the one-percent-bin histograms are in [`generation_report.json`](../data/generation_report.json), with a compact table in [`density_summary.csv`](../data/density_summary.csv).

| Split | D1 | D2 | D3 | D4 | D5 | D6 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training | 300 | 300 | 300 | 0 | 0 | 0 | 900 |
| Validation/calibration | 60 | 60 | 60 | 0 | 0 | 0 | 180 |
| ID pilot | 20 | 20 | 20 | 0 | 0 | 0 | 60 |
| ID holdout | 60 | 60 | 60 | 0 | 0 | 0 | 180 |
| OOD pilot | 0 | 0 | 0 | 20 | 20 | 20 | 60 |
| Sealed OOD test | 0 | 0 | 0 | 60 | 60 | 60 | 180 |

All 1,560 layouts and all 1,560 endpoint instances are unique. The generator validation report records 64×64 dimensions, assigned-density membership, shared generation settings, minimum endpoint separation, legal-path reachability, disjoint ID/OOD pilot and sealed-test layouts, and float64 label/mask checks as passing. Start-goal Chebyshev separation is 63 cells for all generated pairs; mean shortest-path lengths in unit moves are D1 64.00, D2 65.07, D3 67.11, D4 63.25, D5 70.44, and D6 77.38. This separation is consistent across density groups; obstacle detours account for the varying path lengths.

Generation accepted every requested map except one out-of-range D6 sealed-test candidate: 1,561 attempts, 1,560 accepted, one density-target rejection, zero connectivity failures, zero endpoint-selection failures, and zero duplicate-layout failures. The lowest group acceptance rate was D6 sealed test at 60/61 (98.36%); all other groups were 100%.

The sealed archive was created read-only after in-memory generation validation. Its seal record is [`ood_test_seal.json`](../../../data/64x64_rectangles_v1/ood_test_seal.json), with SHA-256 `50b01617def47bc27e09e27d45f2e48e62f0b7690e7cb5950db5a7776f521e6a`. Its contents were not loaded after sealing for training, calibration, integration, or pilot work. Both the sealed archive and its per-map manifest records are excluded from the Kaggle upload bundle. The ID holdout archive and metadata are also excluded from that bundle.

## C. Model status

Shape checks on the current architectures produced 64×64 outputs for H CNN `(1,1,64,64)`, Neural A* U-Net `(1,1,64,64)`, and OOD feature extractor `(1,128)`. These models are fully convolutional for this input size; no fixed 32×32 flattening or resizing is used. Neural A* keeps its existing preprocessing from the canonical three channels to passability and combined endpoints. No architecture or objective was redesigned.

| Model | Primary 64×64 status | Epochs / metrics | Artifact status |
|---|---|---|---|
| H1 | Fresh-from-scratch run prepared; legacy 32×32 weights not reused | 0 of up to 50 | No Phase B checkpoint yet |
| H2 | Fresh-from-scratch run prepared | 0 of up to 50 | No Phase B checkpoint yet |
| H3 | Fresh-from-scratch run prepared | 0 of up to 50 | No Phase B checkpoint yet |
| Neural A* | Existing architecture/objective compatible; training prepared | 0 of up to 10 | No Phase B checkpoint yet |
| Independent OOD encoder | 64×64 contrastive training prepared | 0 of up to 40 | No Phase B checkpoint yet |

Training histories, validation performance, convergence, and inference-time quality cannot be assessed until the Kaggle stages run. Resume-aware checkpoint/history support was added to the existing training functions and is covered by the current test suite.

## D. OOD detector

The prepared pipeline trains an independent 128-dimensional OOD encoder, fits one pooled Gaussian from D1–D3 training embeddings only using Ledoit–Wolf covariance shrinkage with identity target plus recorded ridge regularization, and calibrates `D95` as the 95th percentile of D1–D3 validation/calibration distances. The continuous score remains `s = 1 - exp(-DM / D95)` and is documented as a normalized distance, not a probability.

No encoder was trained, no covariance was fit, and no D95 value is available yet. The pipeline contains checks for finite positive D95 and the score properties at `DM=0` and `DM=D95`. No pilot, holdout, or sealed-test embedding is used for fitting or calibration.

## E. Search integration

The Phase A2 preflight matched [`ood_adaptive_seeastar_implementation_note.md`](../../../reports/ood_adaptive_seeastar_implementation_note.md) and benchmark dispatch. The fixed-K and adaptive-K Neural-guidance SeeA* methods use the same search core. The verified rule is `g(start)=0`, then `g_candidate(v)=g(u)+Phi(v)`; the start Phi is not charged. They use strict improvements, no CLOSED reopening, the approved Chebyshev-plus-epsilon-Euclidean `h`, and uniform OPEN sampling without replacement followed by minimum-f selection. The guidance map `Phi` is produced by Neural A*, never by H1/H2/H3. OOD affects only the once-per-search K choice.

The complete pre-existing and current test suite passed: **150 tests, 62.074 seconds, OK**. This includes the explicit 64×64 overlapping-rectangle search test. Pipeline and pilot module imports passed. The newly written full neural-inference→OOD→search integration stage has not yet run because its trained checkpoints and detector do not exist; therefore these results establish source/test correctness, not end-to-end Phase B execution.

Primary adaptive bounds are `(4,16)`, with `K=Kmax-floor(s*(Kmax-Kmin))`; configured K remains fixed during each search. The existing code specifies the clipped-score ablation exactly as `min(DM/D95,1)`. Bounds `(4,8)`, `(8,32)`, reverse mapping, and clipped mapping are recorded separately in [`adaptive_sensitivity_ablations.json`](../configs/adaptive_sensitivity_ablations.json). None replaces the primary method.

## F. Pilot results

**No pilot has run, so there are no ID/OOD performance metrics to report. OOD-pilot NPZ headers and array shapes were inspected prematurely earlier; no map evaluation, model training, detector fitting, or D95 calibration used it. This correction treated all NPZ files as opaque bytes.** The prepared runner uses 60 ID-pilot maps and 60 OOD-pilot maps, split evenly by their respective density groups, with paired seeds `11, 23, 47, 89, 131`. The primary pilot design is 47 search rows per map: seven deterministic methods, three original SeeA* variants over five seeds, four fixed-K configurations over five seeds, and adaptive K over five seeds (5,640 rows total). It records status, success, both path costs, expanded/generated nodes, search time, K, raw distance, continuous score, neural/heuristic inference time, timeout, and worker resource measures.

The runner derives and freezes a per-search timeout before pilot searches from the slowest ID-only integration search, with a 30-second minimum and 10× margin. Timeout is failure; automatic retry is disabled. Paired aggregation treats map/layout as the independent unit and sampling seeds as repeated measurements. Pilot results are for feasibility and preliminary robustness only.

## G. Feasibility

The generated maps total 1,560; the later final evaluation would contain 360 maps. The primary pilot plans 5,640 rows; the final primary search would plan 16,920 rows, before any later approved ablations. The model inference and CPU graph-search paths will be timed separately by the runner. The local generation completed, but no model-training or search runtime estimate is defensible without Kaggle measurements. Kaggle currently documents 12-hour sessions and 20 GB saved working output; account quota and accelerator availability vary. The upload bundle is approximately 2.1 MB and contains no holdout or sealed-test archive.

## H. Recommendation and stop condition

The dataset, Phase A2 re-verification, 64×64 architecture inspection, Kaggle notebook, modular training/calibration/integration/pilot code, source/data upload bundle, and report are ready for review. The research pipeline is **not yet ready for the final benchmark** because training, D95 calibration, end-to-end integration, and the pilot have not run. Complete those authorized stages in Kaggle, then update this report with measured metrics and resource estimates. The sealed OOD test remains unused after generation/validation and remains excluded from training, calibration, and pilot paths.

The project-specific Kaggle upload bundle is [`64x64_rectangles_v1_kaggle_upload.zip`](../64x64_rectangles_v1_kaggle_upload.zip); the notebook is [`kaggle_experiments_64x64_rectangles.ipynb`](../../../notebooks/kaggle_experiments_64x64_rectangles.ipynb). The uploaded manifest is filtered to 1,200 records from only the four permitted training/calibration/pilot splits. The complete outer and nested file inventory, SHA-256 hashes, and credential/path scan results are in [`complete_bundle_manifest.md`](complete_bundle_manifest.md) and [`complete_bundle_manifest.json`](complete_bundle_manifest.json). Kaggle authentication is the only current access blocker. Upload and execution remain gated on your review and approval. Do not run the 360-map final benchmark without separate explicit approval.

### Verified related work

These references are included to distinguish the method from prior search and OOD work; they do not imply theoretical superiority:

1. Yonetani et al., “Path Planning using Neural A* Search,” ICML 2021, PMLR 139, 12029–12039. [PMLR](https://proceedings.mlr.press/v139/yonetani21a.html)
2. Zhao, Tu, and Xu, “SeeA*: Efficient Exploration-Enhanced A* Search by Selective Sampling,” NeurIPS 2024. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/bc8b2058fd96978a4146f18298cb2d39-Abstract-Conference.html)
3. Zhao, Tu, Sun, and Xu, “KeeA*: Epistemic Exploratory A* Search via Knowledge Calibration,” NeurIPS 2025. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5393eaf0b723278738788520ec54e4b4-Abstract-Conference.html)
4. Lee et al., “A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks,” NeurIPS 2018. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2018/hash/abdeb6f575ac5c6676b747bca8d09cc2-Abstract.html)
5. Ledoit and Wolf, “A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices,” *Journal of Multivariate Analysis* 88(2), 365–411, 2004. [DOI](https://doi.org/10.1016/S0047-259X(03)00096-4)
6. Futuhi and Sturtevant, “Learning Admissible Heuristics for A*: Theory and Practice,” ICLR 2026. [OpenReview paper](https://openreview.net/pdf/a2ec379ef8af5d8ddc561a995d9dc45580a6d816.pdf)

The proposed method uses uniform sampling without replacement and changes only sample budget `K` with a continuous Mahalanobis score. KeeA* uses heuristic-related knowledge calibration with cluster scouting and path-aware sampling. These are related exploration/search methods, but the sampling mechanisms differ. Here D1–D3 define ID training/validation and D4–D6 are unseen density distributions, while rectangle size, placement, overlap, movement, and endpoint rules are shared. Results from these rectangular maps would not establish generalization to arbitrary map families.
