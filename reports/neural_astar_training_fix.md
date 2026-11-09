# Neural A* training fix: CPU evidence and notebook changes

Branch: `fix/neural-astar-collapse`  
Runtime: Windows, PyTorch 2.14.1+cpu, CUDA unavailable, 2 CPU threads. All model experiments and tests used CPU. No full 20-epoch run was performed locally.

## Outcome

GroupNorm(8) prevented the original immediate activation blow-up on CPU, but the pasted Kaggle `step_health.csv` shows that `lr=1e-3` still develops a later saturation failure: the saturated fraction jumps to `64.7%` at step 500 and is generally `72–86%` from steps 700–3500. At step 3600, `88.0%` of probe cells are saturated, mean absolute logit is `8.24`, and the training guard stops. After changing to 20 epochs and `lr=3e-4`, the user's subsequent Kaggle run reached about epoch 13–14 before the mean absolute logit again crossed 8 and the guard stopped. This confirms that `3e-4` delayed but did not eliminate longer-run logit drift.

A matched 300-update CPU comparison supports trying RMSprop at `1e-4` after the 20-epoch `3e-4` Kaggle run stopped at epoch 13–14. On the same 60/30 maps and seed, `1e-4` kept logits and saturation lower through 300 updates and produced a stronger downward training-loss trend than `3e-4`; its best validation loss was worse (`.03874` vs `.02815`). I selected `1e-4` for the next run as a stability trial, not as a demonstrated validation improvement. It has only been measured locally for 300 updates, so it does not prove that all 20 Kaggle epochs will finish or that learned Phi will beat constant Phi `.25`.

The one-epoch batch-4 comparison was slower and had worse train and validation loss at the same 60-map exposure, so batch size remains 1. The user-approved next run uses 20 Neural A* epochs and learning rate `1e-4`; it keeps the 900 training maps, 180 validation maps, batch size 1, seeds, objective, search, heuristic, cost weighting, Phi range, inputs, detector and OOD training unchanged.

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
| GroupNorm(8), RMSprop `1e-3`, balanced 60 train / 30 val | 180 | First 50 mean `0.07718`; last 50 `0.06837` (`−11.4%`) | No early guard or collapse. Phi std at steps 10/30/100: `.172/.268/.397`; saturation `0%/8.9%/9.8%`; mean-absolute logits `1.23/1.56/2.80`. | Not measured in this 180-step run | `796 s`, `0.226 steps/s` | Prevents the original immediate blow-up, but the Kaggle trace later saturates at step 500+. |
| GroupNorm(8) plus small head init `Normal(0,.001)`, zero bias, RMSprop `1e-3`, same 60/30 split and seed | 100 | First 50 `0.07008`; last 50 `0.06137` (`−12.4%`) | Final probe Phi std `.308`, saturation `2.35%`, mean-absolute logit `1.88`, max-absolute logit `7.38` | Not measured | `334 s`, `0.299 steps/s` | Not selected: validation loss `.03773` at step 50 then `.05057` at step 100, both above constant `.02557`; head-init change did not fix generalization. |
| GroupNorm(8), RMSprop `1e-4`, one-map overfit probe | 68 (guard stopped) | Final loss `.03882` vs initial `.02612`; did not beat the same-map best constant `.03271` | Phi std `.277`; saturation `0%`; mean-absolute logit `1.56` | Not measured | `235 s`, `.289 steps/s` | Not selected; narrower probe than the matched 60-map candidate test. |
| Earlier `lr=1e-5` exploratory trace | 100 | First 50 `.04902`; last 50 `.03835` (`−21.8%`) | Final Phi std `.00265` (effectively spatially constant); no saturation; max absolute logit `.079` | 0 observed | `.182 steps/s` | Not selected. Its sample/seed provenance was not verified as matched to the 60-map run. |
| Earlier `lr=1e-4` exploratory trace | 100 | First 50 `.06354`; last 50 `.08002` (`+25.9%`) | Final Phi std `.0985`, mean Phi `.887`, saturation `.9285`, max absolute logit `3.28` | 0 observed | `.104 steps/s` | Not selected. Its sample/seed provenance was not verified as matched to the 60-map run. |
| GroupNorm(8), RMSprop `3e-4`, matched 60/30, batch 1 | 300 | First 50 `.06362`; last 50 `.05403` (`−15.1%`) | Step 300: Phi std `.360`, saturation `6.09%`, mean/max absolute logits `2.86/8.79`; no guard | Not measured | `1122 s`, `.267 steps/s` | Better validation than `1e-4` locally, but the full Kaggle run later tripped the logit guard at epoch 13–14. |
| GroupNorm(8), RMSprop `1e-4`, matched 60/30, batch 1 | 300 | First 50 `.07489`; last 50 `.05713` (`−23.7%`) | Step 300: Phi std `.354`, saturation `.12%`, mean/max absolute logits `2.22/4.91`; no guard | Not measured | `1678 s`, `.179 steps/s` | Selected for the next Kaggle trial for lower logit/saturation drift; validation is worse than `3e-4` and remains above constant `.02557`. |
| GroupNorm(8), RMSprop `1e-3`, matched 60/30, batch 4 | 15 | One epoch train `.08216`; validation `.05171` | Epoch-end Phi std `.295`, saturation `0%`, mean absolute logit `2.60` | Not measured | `569 s`, `.026 updates/s`, `.105 map/s` | Rejected: batch 1 had lower train/validation loss and nearly 2x map throughput at the same exposure. |

### Follow-up: Kaggle late saturation, learning rate, and mini-batch

The attached Kaggle trace begins healthy: at step 100, mean absolute logit was `2.85` and saturation was `0.16%`. The deterioration starts later: saturation is `9.3%` at step 400, then `64.7%` at step 500; from step 700 through step 3500 it is usually between `72%` and `86%`. At the failing step 3600, the trace reports loss `.02246`, mean spatial Phi std `.4675`, across-map std `.00676`, saturation `.8799`, mean absolute logit `8.242`, max absolute logit `18.535`, and nonzero gradient norm `3.55e-5`. The guard's mean-logit threshold, rather than its 95% saturation threshold, caused the reported stop. The trace records output/logit health, not intermediate activations, so it does not establish hidden-layer explosion in this later run. It does establish that the model became mostly binary before the guard stopped it.

The matched `3e-4` run used seed `20261306`, 60 training maps (20 each from D1–D3), 30 validation maps, GroupNorm(8), RMSprop, and batch size 1. It ran 300 updates over five local epochs. At steps 100/200/300, mean absolute logits were `1.96/2.65/2.86`, saturated fractions were `.74%/5.45%/6.09%`, and mean spatial standard deviations were `.355/.367/.360`. First-50 versus last-50 training loss was `.06362` versus `.05403` (`−15.1%`). Its best validation checkpoint was epoch 4 (`.02815`); the final value was `.03569`. Runtime was `1122 s` on CPU (`.267 updates/s`); CUDA was unavailable.

The matched `1e-4` run used the same seed, maps, architecture, optimizer, and batch size, changing only the learning rate from `3e-4` to `1e-4`. At steps 100/200/300, mean absolute logits were `1.96/1.99/2.22`, saturated fractions were `0%/.34%/.12%`, and mean spatial standard deviations were `.313/.314/.354`. First-50 versus last-50 training loss was `.07489` versus `.05713` (`−23.7%`). Epoch train losses were `.07981, .07255, .07761, .06629, .06408`; validation losses were `.04873, .05578, .03874, .04637, .05007`. Its best validation loss at epoch 3 (`.03874`) remained above constant Phi `.25` (`.02557`) and was worse than the best `3e-4` checkpoint. The run took `1678 s` on CPU (`.179 updates/s`); CUDA was unavailable. This supports the lower rate's stability benefit but exposes a validation tradeoff.

The existing matched `1e-3` batch-1 run had a first/last-50 training trend of `−11.4%`, validation loss `.03311` at 180 updates, and `.03922` after continuation to 300 updates. At 180 updates its health probe reported mean absolute logits `3.44` and saturation `6.96%`; at 300 updates the continuation reported `3.51` and `19.36%`. These comparisons support `3e-4` as the more stable setting over the measured 300 updates, though the validation result remains above the best constant.

For mini-batch size 4, the matched run used the same 60/30 maps, seed, GroupNorm, and `1e-3` learning rate, and completed one epoch (15 optimizer updates for 60 map presentations). It ended with train loss `.08216`, validation loss `.05171`, Phi spatial std `.295`, no saturated cells at the epoch-end probe, and mean absolute logit `2.60`. The batch-1 `1e-3` first epoch had train loss `.07512` and validation loss `.03678` after the same 60 map presentations; its epoch took `304 s`, while batch 4 took `569 s` (`.198` vs `.105` map presentations/s on this CPU). Batch 4 therefore did not improve the measured learning or CPU throughput. Because it had only 15 optimizer updates, it does not test whether it prevents the later step-500-plus saturation; I did not extend this slower candidate locally.

Warmup, gradient clipping, and gradient accumulation were not run as 100-update matched candidates. They were not added to the production configuration. The candidate matrix is therefore incomplete; the adopted optimization deviations are GroupNorm and the learning-rate reduction, each tied to measured activation/health or loss-trend results above. The first-step baseline gradient norm was only `3.73e-6`; ordinary clipping thresholds would not clip that measured first update, but that observation is not a substitute for a clipping experiment.

### One-map overfit

On the selected D1 training map (seed `14404694994049986883`), the initial loss was `.06812`; the best of six same-map constants was `.02319`. The normal trainer stopped at step 31 because both train and one-map validation losses repeated to six decimals. For the requested 200-step diagnostic, I continued the saved model and RMSprop state offline, without changing production guard behavior. At step 200 loss was `.015625` (`22.9%` of initial), below the best constant, with Phi spatial std `.2433`. The `<=50%` initial-loss, below-constant, and nonconstant-Phi criteria all passed. The first/last 50 means in the 200-step continuation were `.00673` and `.00756`; the discrete loss was not monotonic. This demonstrates single-map capacity, not generalization.

### Validation control and 300-step check

On the first 30 validation maps, after 180 GroupNorm updates, learned Phi loss was `.033105`; constant 0, `.25`, and 1 were `.025854`, `.025570`, and `.051961`. Learned Phi was shaped (mean spatial std `.3968`, across-map mean std `.00665`) and not broadly saturated (`6.55%`), but it did not beat the best constant. Continuing the same optimizer/model state to 300 updates did not fix this: learned `.039217` vs constant `.025570`; Phi remained shaped (mean spatial std `.3734`) but saturation rose to `19.36%`, mean-absolute logit `3.51`, and maximum absolute logit `17.65`. The 300-update run did not trip the configured guard, but the validation objective remains a concern.

## Chosen configuration and guard

The next pipeline run uses RMSprop `lr=1e-4`, alpha `.99`, epsilon `1e-8`, zero momentum/weight decay, uncentered; batch size 1, accumulation 1, no clipping, no warmup; GroupNorm(8) after every encoder/decoder convolution; PyTorch default initialization; temperature `sqrt(64)=8` via the search default; tie-break epsilon `.001`; seed `20261306`; 20 epochs. The mean-L1 objective, sigmoid Phi, two inputs, adaptive-K, detector, data gates and CUDA requirement remain as before.

The deviations from the original Kaggle baseline are GroupNorm(8) after each convolution and RMSprop learning rate `1e-4` instead of `1e-3`; the training budget is now 20 epochs. They are recorded in `training_config.json`, `training_complete.json`, and `training_summary.json`. Resume rejects old or mismatched configurations/checkpoints instead of loading them silently; use a fresh result directory for the new Kaggle configuration. `train_models()` still resolves the requested CUDA device and raises if unavailable; local CPU training called the lower-level `train_neural_astar()` API. Batch size remains 1 because the measured batch-4 run did not help.

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

- The full suite passed after the `1e-4`, 20-epoch configuration change: 161 tests. The config regression test checks 20 epochs, batch size 1, and learning rate `1e-4`. The notebook passed JSON and nbformat validation with 46 cells; it was not executed. A previous version of `test_search_metrics_are_populated` expected `closed_size == expanded_nodes` and failed 7 vs 8; that mismatch was reproduced at base commit `95eb754`, then the assertion was corrected to reflect existing A* behavior (the goal counts as expanded but is not added to CLOSED). A* search behavior itself remains unchanged.
- All seven Neural A* health/config tests passed across targeted runs, including the accumulation regression, constant and saturated guard cases, healthy varying Phi, mismatched resume rejection, and no-model-argument checkpoint inference round trip.
- Final notebook harness passed all 23 code cells on CPU with 24 training, 8 validation and 6 pilot maps. It asserted the exact OOD skip output `OOD pilot NOT released: skipping this section...` while the integration prerequisite was unmet, and verified no OOD-pilot archive was copied or loaded. It also verified `phi_health.json`, a `HEALTH CHECK FAILED` banner in `report.md`, `results_manifest.json`, and all three inside `phase_b_results.zip`; both constant search arms and `phi_source` values were present. The harness keeps `REQUIRE_LEARNED_PHI=True`; its health table measured learned `.08112` vs best constant `.03699`, warned, and still completed every later cell.
- The local harness produced only `training`, `validation_calibration`, and `id_pilot` loaded split names. Full notebook training was not run locally.

## Kaggle first-epoch expectations and limits

On the CPU 60-map `1e-4` trace, step-health probes at 10/30/100 had Phi spatial std `.224/.251/.313`, saturated fractions `0%/0%/0%`, and mean-absolute logits `1.08/1.67/1.96`. Epoch 1 averaged `.0798` train loss and `.0487` validation loss. These are small-subset CPU values, not a T4 guarantee. Expect nonzero Phi spatial std, moderate logits, and saturation far below the late-run levels seen at `3e-4`; continue monitoring each logged probe. The 1e-4 model's best local validation loss remained above constant Phi `.25`, so the post-training health table may still report a validation-control warning.

The local 180-update epoch on 60 maps took roughly 4.4 minutes on this CPU (`0.226 steps/s` overall). The previous Kaggle baseline took 7,566 seconds for 10 epochs (`12.6 min/epoch`) on a T4. GroupNorm adds work, and there is no T4 measurement for this configuration. Budget provisionally about 13–16 minutes per Kaggle epoch, with low confidence; judge the run first by the 10/30/100-step health log and the first-epoch loss/validation table. The validation control result above means a healthy-looking Phi may still fail the learned-vs-constant check, so inspect that table before spending the remaining epoch budget.

## Remaining risks

The required immediate collapse is reproduced and prevented by GroupNorm. The Kaggle run at `3e-4` nevertheless reached the logit guard around epoch 13–14. The `1e-4` candidate stayed healthy through 300 CPU updates but had worse validation loss than `3e-4` and still lost to constant Phi `.25`; full 20-epoch stability and validation behavior remain unknown. The notebook reports health-check failure without aborting later cells; only the training guard stops a collapsed run. Warmup/clipping/accumulation remain untested, and batch 4 was tested for one epoch only. No loss or search changes were made.
