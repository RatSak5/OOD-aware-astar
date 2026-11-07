# Feature-space OOD detection implementation note

## Method references and project adaptation

This note uses Lee et al., *A Simple Unified Framework for Detecting
Out-of-Distribution Samples and Adversarial Attacks* (NeurIPS 2018), including
its supplementary material, and Ledoit and Wolf, *A Well-Conditioned Estimator
for Large-Dimensional Covariance Matrices* (2004). Supplied files:

- `C:/Users/ratul/Downloads/1807.03888v2.pdf` (20 PDF pages including supplement)
- `C:/Users/ratul/Downloads/ole1a.pdf` (43 PDF pages including appendices)

Lee et al. fit class-conditional Gaussians with a shared covariance to
supervised classifier features (main paper Sec. 2.1, Eq. (1), p. 3). Their
confidence score is the negative squared Mahalanobis distance to the nearest
class mean (Eq. (2), p. 3); their full detector also uses features from several
network layers and can add input preprocessing (Algorithm 1, p. 4). This
project has no map classes and uses an independent self-supervised map encoder,
so it fits one Gaussian to the pooled D1-D3 training-map embeddings. This is an
explicit single-Gaussian, single-feature-level adaptation, not an exact
reproduction of Lee et al.'s class-conditional classifier detector.

The feature extractor is trained independently of the Prompt 4 heuristic
prediction head, using contrastive pairs formed by symmetry-preserving flips
and 180-degree rotations of training maps. This avoids using a jointly trained
heuristic head for OOD scoring and prevents heuristic-target labels from
defining the embedding statistics. The OOD score does not imply heuristic
error or a calibrated probability.

The encoder input has three channels: blocked-cell occupancy, start one-hot,
and goal one-hot. The CNN emits a fixed-dimensional pre-projector embedding
`z`; only during optional self-supervised training, a projection head maps it
to normalized contrastive vectors. Two randomly sampled horizontal/vertical
flip views form positives. The training objective is symmetric NT-Xent
(cross-entropy over cosine-similarity logits divided by temperature, with the
paired view as the target and the diagonal excluded). Detector statistics
always use `z`, never the projector output. No heuristic head features are
used, so a jointly trained versus independent feature comparison is not
required for this implementation choice.

## Ledoit-Wolf covariance shrinkage

For centered training embeddings `x_i` in `R^p`, the Ledoit-Wolf sample
covariance is

```text
S = (1/n) sum_i x_i x_i^T
mu = trace(S) / p
delta2 = ||S - mu I||_F^2
beta2 = min( (1/n^2) sum_i ||x_i x_i^T - S||_F^2, delta2 )
lambda = beta2 / delta2
Sigma_LW = (1-lambda) S + lambda mu I
```

This is the feasible shrinkage form in Ledoit and Wolf (2004), Eq. (14),
PDF p. 12, using their sample counterparts for the shrinkage target and
intensity. The terms are evaluated in float64; an explicitly recorded small
ridge is added before factorization/inversion for numerical protection,
particularly if an embedding collapses and `mu` is nearly zero. This ridge is
additional numerical regularization beyond the Ledoit-Wolf estimator.

Training maps only determine the mean and covariance. Separate, fingerprint-
checked D1-D3 validation maps determine `D95`, the empirical 95th percentile
of their raw Mahalanobis distances. D4-D6 are excluded from both steps.

## Raw distance, continuous score, and metrics

For a feature `z`,

```text
D_M(z) = sqrt((z-mu)^T Sigma^{-1} (z-mu))
s(z)   = 1 - exp(-D_M(z) / D95)
```

Implementation uses `-expm1(-D_M/D95)` for precision near zero. If finite
floating-point underflow rounds a very large finite score to exactly one, it
returns the largest representable float below one; it does not threshold at
`D95`. `D95` must be finite and greater than a configurable minimum, otherwise
calibration raises `CalibrationError` instead of silently dividing by zero.
At `D_M=D95`, `s=1-exp(-1)≈0.63212056`. It is a monotonic normalization, not a
probability or universal threshold.

Binary summaries use the explicit default operating point `s >= 1-exp(-1)`,
equivalent to `D_M >= D95`, and always report that threshold. AUROC and
trapezoidal AUPR treat OOD as positive. Map-level bootstrap resampling gives
95% percentile confidence intervals for per-distribution means, raw and
normalized ID/OOD mean gaps, AUROC, AUPR, FPR, and FNR; ID and OOD samples are
resampled separately for binary comparisons. Per-distribution score summaries
are reported for D1-D6. The score threshold is configurable, with the
`1-exp(-1)` operating point used by default.

For a separately supplied Prompt 4 model, heuristic error is calculated as
`mean(|h_pred-h_star|)` over reachable traversable cells. Pearson and Spearman
correlations are reported for both raw distance and normalized score, overall
and by distribution, with map-bootstrap confidence intervals. These are
empirical relationships only; no causal or uncertainty claim follows from
distance alone.
