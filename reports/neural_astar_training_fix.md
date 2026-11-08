# Neural A* training fix: CPU evidence and notebook changes

Branch: `fix/neural-astar-collapse`  
Runtime: Windows, PyTorch 2.14.1+cpu, CUDA unavailable, 2 CPU threads. All model experiments and tests used CPU. No full 10-epoch run was performed.

## Outcome

The evidence supports activation/logit explosion followed by sigmoid saturation as the cause of the Kaggle collapse. Adding GroupNorm(8) after each convolution prevents that early collapse and meets the 180-update training-trend check on a balanced 60-map subset. A one-map overfit experiment also learned a shaped map that beat the best constant. However, on the 30-map local validation subset the 180-update GroupNorm model still lost to constant Phi=0.25 (`0.03311` vs `0.02557`). A separate 300-update continuation also lost (`0.03922`). The notebook now records this as a nonblocking health failure and continues; the full-data Kaggle model may still fail the constant-control check. This is a remaining generalization/optimization risk; it is not reported as solved.

The selected production change is GroupNorm(8). On the 180-step run, the loss-trend, nonzero between-map variation, mean-absolute-logit, and no-guard checks passed; the exact fraction of cells in `[0.02, 0.98]` was not recorded, so that threshold is unverified. RMSprop, its learning rate and settings, default initialization, sigmoid Phi, two input channels, loss, search semantics, temperature, seeds, data and epoch budget were retained. No loss, search, heuristic, cost, Phi-range, detector, or OOD-training behavior was changed.

## Step 0: baseline and root-cause evidence

I used only copied `training`, `validation_calibration`, and `id_pilot` files in `results/_tmp_neural_astar_diag_20261108/work_data`. The baseline trace used the Kaggle seed `20261306`, the unnormalized model, RMSprop at `1e-3`, and the original differentiable-search settings. It collapsed at optimizer step 15 on CPU, within the reported 10–20-step window. I did not run other baseline seeds, so seed-to-seed reproducibility of the collapse remains unmeasured.

| Step | Observation |
|---:|---|
| 1 | Phi mean `0.483`, spatial std `0.00065`; max ReLU activation `1.30`; global gradient norm `3.73e-6`; no dead ReLU layer. |
| 9 | Max ReLU activation `6.70`; max logit `1.34`. |
| 11 | Max ReLU activation `52.2`; max logit `4.06`; Phi mean `0.952`. |
| 12 | Max logit `38.5`; gradients were near zero. |
| 14 | Max ReLU activation `6,764`; max logit remained about `38.5`. |
| 15 | Phi was exactly 1 across the map; logits ranged about `329` to `7,693`; gradients and parameter updates were zero. No ReLU layer was fully dead. |

Activation growth precedes the saturated output and vanished gradient. This supports H2 and argues against H1 as the initiating cause. The captured original trace averaged `0.063 steps/s` through collapse; this is a short diagnostic trace, not epoch throughput.

The step-1 head gradient norm was `3.28e-7`, head bias/weight norms were `.0657/.568`, and mean absolute parameter update was only `1.82e-6` even though the largest individual update was `.00841`. At step 11 the maximum activation was `52.2`, the global/head gradient norms were `.000610/.000291`, and the maximum update was `.00998`. By step 12 the maximum logit had reached `38.5` while the global gradient fell to `1.21e-7`; by step 15 both gradient and update norms were zero. Per-layer mean-absolute activations, maxima, and active fractions were recorded in `original_steps.csv`: the minimum active fraction was `.365` at step 1 and `.310` at step 15, with zero completely dead ReLU layers throughout. Thus the ReLU stacks remained active as their activations exploded; the saturated sigmoid then removed the gradient.

### Real differentiable-loss controls

These means use the actual differentiable loss on 30 training maps (10 each from D1–D3), not the deterministic K=512 proxy. Seven arms took `95.96 s` total on CPU.

| Phi | User proxy reference | Real differentiable loss |
|---:|---:|---:|
| 0.00 | 0.0242 | 0.02965 |
| 0.05 | 0.0243 | 0.03828 |
| 0.25 | 0.0250 | 0.04161 |
| 0.50 | 0.0290 | 0.05208 |
| 0.75 | 0.0413 | 0.06193 |
| 1.00 | 0.0824 | 0.10764 |
| Oracle: 0.05 on optimal path, 1 elsewhere | 0.0058 | 0.01818 |

The reference and local numbers differ because the maps and evaluation procedure differ. Their main ordering agrees: constant Phi=1 is worst, and the shaped oracle is better than constants. H3 is not supported.

### Relaxed-gradient checks

At initialization, the mean gradient for a global Phi offset over 30 maps was `-1.16e-11` (mean absolute magnitude `2.44e-11`); 29 of 30 map values had absolute magnitude at most `1e-10`. There is no measured upward offset direction. This scalar direction is effectively zero and does not explain the initial movement toward high Phi.

For nine selected cells, finite differences with epsilon `1e-3` had five nonzero values; autograd and finite-difference signs agreed for 3/5 (`60%`). Mean absolute autograd magnitude was `6.89e-5`, versus `2.18` for finite differences. The hard-argmax forward pass is discontinuous and piecewise constant, while the backward pass uses the specified identity straight-through estimator and relaxed search path. Finite differences can therefore be zero or jump across a discrete search change; this check is not a classical smooth-gradient validation. Search behavior was left unchanged.

## Step 2: candidates and overfit evidence

| Candidate / evidence | Updates | Loss trend | Phi / saturation / logits | Dead ReLU layers | CPU time | Decision |
|---|---:|---|---|---:|---:|---|
| Original, no normalization, RMSprop `1e-3`, seed `20261306` | 15 | Collapsed by step 15 | Phi exactly 1; logits `329–7,693` | 0 before collapse | `0.063 steps/s` through short trace | Reject; reproduced failure. |
| GroupNorm(8), RMSprop `1e-3`, balanced 60 train / 30 val | 180 | First 50 mean `0.07718`; last 50 `0.06837` (`−11.4%`) | No guard or collapse. Phi std at steps 10/30/100: `.172/.268/.397`; saturation `0%/8.9%/9.8%`; mean-absolute logits `1.23/1.56/2.80`. | Not measured in this 180-step run | `796 s`, `0.226 steps/s` | Selected to prevent measured explosion and for the required downward training trend. Validation caveat below. |
| GroupNorm(8) plus small head init `Normal(0,.001)`, zero bias, RMSprop `1e-3`, same 60/30 split and seed | 100 | First 50 `0.07008`; last 50 `0.06137` (`−12.4%`) | Final probe Phi std `.308`, saturation `2.35%`, mean-absolute logit `1.88`, max-absolute logit `7.38` | Not measured | `334 s`, `0.299 steps/s` | Not selected: validation loss `.03773` at step 50 then `.05057` at step 100, both above constant `.02557`; head-init change did not fix generalization. |
| GroupNorm(8), RMSprop `1e-4`, one-map overfit probe | 68 (guard stopped) | Final loss `.03882` vs initial `.02612`; did not beat the same-map best constant `.03271` | Phi std `.277`; saturation `0%`; mean-absolute logit `1.56` | Not measured | `235 s`, `.289 steps/s` | Not selected; narrower probe than the matched 60-map candidate test. |
| Earlier `lr=1e-5` exploratory trace | 100 | First 50 `.04902`; last 50 `.03835` (`−21.8%`) | Final Phi std `.00265` (effectively spatially constant); no saturation; max absolute logit `.079` | 0 observed | `.182 steps/s` | Not selected. Its sample/seed provenance was not verified as matched to the 60-map run. |
| Earlier `lr=1e-4` exploratory trace | 100 | First 50 `.06354`; last 50 `.08002` (`+25.9%`) | Final Phi std `.0985`, mean Phi `.887`, saturation `.9285`, max absolute logit `3.28` | 0 observed | `.104 steps/s` | Not selected. Its sample/seed provenance was not verified as matched to the 60-map run. |

Warmup, gradient clipping, and gradient accumulation were not run as 100-update matched candidates. They were not added to the production configuration. The candidate matrix is therefore incomplete; the only adopted optimization deviation is the GroupNorm change, which has direct before/after evidence. The first-step baseline gradient norm was only `3.73e-6`; ordinary clipping thresholds would not clip that measured first update, but that observation is not a substitute for a clipping experiment.

### One-map overfit

On the selected D1 training map (seed `14404694994049986883`), the initial loss was `.06812`; the best of six same-map constants was `.02319`. The normal trainer stopped at step 31 because both train and one-map validation losses repeated to six decimals. For the requested 200-step diagnostic, I continued the saved model and RMSprop state offline, without changing production guard behavior. At step 200 loss was `.015625` (`22.9%` of initial), below the best constant, with Phi spatial std `.2433`. The `<=50%` initial-loss, below-constant, and nonconstant-Phi criteria all passed. The first/last 50 means in the 200-step continuation were `.00673` and `.00756`; the discrete loss was not monotonic. This demonstrates single-map capacity, not generalization.

### Validation control and 300-step check

On the first 30 validation maps, after 180 GroupNorm updates, learned Phi loss was `.033105`; constant 0, `.25`, and 1 were `.025854`, `.025570`, and `.051961`. Learned Phi was shaped (mean spatial std `.3968`, across-map mean std `.00665`) and not broadly saturated (`6.55%`), but it did not beat the best constant. Continuing the same optimizer/model state to 300 updates did not fix this: learned `.039217` vs constant `.025570`; Phi remained shaped (mean spatial std `.3734`) but saturation rose to `19.36%`, mean-absolute logit `3.51`, and maximum absolute logit `17.65`. The 300-update run did not trip the configured guard, but the validation objective remains a concern.

## Chosen configuration and guard

The pipeline uses RMSprop `lr=1e-3`, alpha `.99`, epsilon `1e-8`, zero momentum/weight decay, uncentered; batch size 1, accumulation 1, no clipping, no warmup; GroupNorm(8) after every encoder/decoder convolution; PyTorch default initialization; temperature `sqrt(64)=8` via the search default; tie-break epsilon `.001`; seed `20261306`; 10 epochs. The mean-L1 objective, sigmoid Phi, two inputs, adaptive-K, detector, data gates and CUDA requirement remain as before.

The sole deviation from the Kaggle baseline is GroupNorm(8) after each convolution. It is recorded in `training_config.json`, `training_complete.json`, and `training_summary.json`. Resume rejects old or mismatched configurations/checkpoints instead of loading them silently. `train_models()` still resolves the requested CUDA device and raises if unavailable; local CPU training called the lower-level `train_neural_astar()` API.

`train_neural_astar()` writes `step_health.csv` separately from the unchanged four-column `losses.csv`; it logs every optimizer step’s loss/gradient norm and fills Phi/logit metrics on health probes at steps 10, 20, 30, 50, every 100, and epoch end. It prints probe progress and raises `NeuralAStarCollapseError` for spatial collapse, over-95% saturation, mean absolute logit above 8, or train and validation losses unchanged to six decimals at adjacent epoch ends. Exceptions propagate through `train_models()`.

Code review also found that the optional accumulation path stepped by the effective batch size but sliced only `batch_size` samples. I corrected the slice to consume the full effective batch and added a two-map, one-update CPU regression test. The Kaggle/default config remains accumulation 1; this correction does not change the selected run.

## Notebook changes

In `notebooks/kaggle_experiments_64x64_rectangles.ipynb`:

- Kept the self-contained clone-and-run notebook, its `GITHUB_REPO_URL` cell, map regeneration/copy flow, provenance JSON, and cleared outputs. Default repository is `https://github.com/RatSak5/OOD-aware-astar.git`; default ref is `fix/neural-astar-collapse`; placeholder safety checks remain.
- Added intro text for the guard, Phi health check, and constant controls; added step-health curves for Phi spatial std, mean-absolute logits, and loss.
- Added the post-training validation health table using the first 60 validation maps, learned Phi and constant Phi 0, `.25`, and 1, with spatial variation and saturation checks. `REQUIRE_LEARNED_PHI=True` by default; any failure now writes `phi_health.json`, prints a loud warning, and adds a `HEALTH CHECK FAILED` report banner without aborting later notebook cells.
- Added fixed-K=10 constant-Phi controls `seeastar_const_phi1_k10` and `seeastar_const_phi025_k10`, sharing the existing five seeds; retained existing column names/arms and added `phi_source` (`learned`/`constant`). Updated labels, colors, tables, figures, and paired comparisons for fixed-K and adaptive vs each control with the existing bootstrap/permutation/Holm method.
- Added the health table, final training configuration, and health status banner to `report.md`. The OOD gate prints the exact skip message and does no prerequisite-file reads or OOD-pilot loads while approval is false; prerequisite checks and release stay inside the approved branch. The following refresh cell only rebuilds tables/figures when OOD results were added. Data gates still load only training, validation_calibration and id_pilot.

## Local verification

- Final full unittest suite before these notebook-only changes: 160 tests ran; 159 passed and one unrelated test failed: `test_search_metrics_are_populated` asserts `closed_size == expanded_nodes`, but generic A* counts the goal as expanded and does not add the goal to `closed_states` (8 vs 7). I checked out base commit `95eb754` in a detached worktree and ran that exact test there; it failed identically with `7 != 8`. The branch has not changed A* behavior, so I left the unrelated metric/search code untouched.
- All seven Neural A* health/config tests passed across targeted runs, including the accumulation regression, constant and saturated guard cases, healthy varying Phi, mismatched resume rejection, and no-model-argument checkpoint inference round trip.
- Final notebook harness passed all 23 code cells on CPU with 24 training, 8 validation and 6 pilot maps. It asserted the exact OOD skip output `OOD pilot NOT released: skipping this section...` while the integration prerequisite was unmet, and verified no OOD-pilot archive was copied or loaded. It also verified `phi_health.json`, a `HEALTH CHECK FAILED` banner in `report.md`, `results_manifest.json`, and all three inside `phase_b_results.zip`; both constant search arms and `phi_source` values were present. The harness keeps `REQUIRE_LEARNED_PHI=True`; its health table measured learned `.08112` vs best constant `.03699`, warned, and still completed every later cell.
- The local harness produced only `training`, `validation_calibration`, and `id_pilot` loaded split names. Full notebook training was not run locally.

## Kaggle first-epoch expectations and limits

On the CPU 60-map trace, probes were healthy at steps 10/30/100: Phi spatial std `.172/.268/.397`, saturated fraction `0%/8.9%/9.8%`, and mean-absolute logits `1.23/1.56/2.80`. Epoch 1 averaged `.0751` train loss and `.0368` validation loss; the second and third train means were `.0694` and `.0681`. Provisionally expect a first Kaggle epoch mean in the `.06–.08` training range and roughly `.03–.05` validation range, with a downward training trend; those are small-subset observations, not a T4 guarantee. A healthy Kaggle map should show spatial structure and a visibly nonflat colorbar spanning at least about `.3`; a flat image at 1.00 is the old failure. Expect step-health rows at 10, 30, and 100 to have nonzero spatial std, mean-absolute logits below 5, and saturation well below 95%; loss should move downward over the first epoch rather than become identical on successive epochs. The post-training table should show meaningful within-map Phi std, nonzero across-map variation, and learned validation loss below the best of constants 0, `.25`, and 1. The local model did not meet that last condition.

The local 180-update epoch on 60 maps took roughly 4.4 minutes on this CPU (`0.226 steps/s` overall). The previous Kaggle baseline took 7,566 seconds for 10 epochs (`12.6 min/epoch`) on a T4. GroupNorm adds work, and there is no T4 measurement for this configuration. Budget provisionally about 13–16 minutes per Kaggle epoch, with low confidence; judge the run first by the 10/30/100-step health log and the first-epoch loss/validation table. The validation control result above means a healthy-looking Phi may still fail the learned-vs-constant check, so inspect that table before spending the remaining epoch budget.

## Remaining risks

The required early collapse is reproduced and prevented by GroupNorm in the tested 180-step subset, and a shaped Phi can overfit one map. The evidence does not establish that the 10-epoch full-data model will beat constant Phi on validation; local 180- and 300-update validation checks did not. The notebook intentionally stops by default when it does not. The optimizer-candidate matrix is incomplete for warmup/clipping/accumulation, and finite differences are not a smooth check of the straight-through search derivative. No changes were made to address these concerns by altering the frozen loss or search.
