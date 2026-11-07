# CNN heuristic learning and CEA implementation note

## Source and applicability

Source: Futuhi and Sturtevant, *Learning Admissible Heuristics for A*: Theory
and Practice* (ICLR 2026), supplied as
`C:/Users/ratul/Downloads/2509.22626v2.pdf` (31 PDF pages, including the
appendix). Relevant references are Assumption 4 (main-paper p. 4), the CEA
objective Eq. (5) and its ordinal-class discussion (p. 8), the training and
sampling discussion (p. 9), model details (Appendix G, PDF p. 25), and CEA
hyperparameter guidance (Appendix J, PDF pp. 26–27).

CEA assumes a finite cost-class representation and a constant edge cost
(Assumption 4). Prompt 1's graph has unit cost on every legal edge, so every
finite optimal cost-to-go is an integer. For an `H x W` map, a shortest simple
path costs at most `H*W-1`; therefore the fixed classes `0, ..., H*W-1` cover
all maps of the configured size. Obstacles and cells outside the goal's
connected component are excluded from supervised losses and constraint
statistics.

The paper's experiments use Rubik's Cube pattern databases, state encodings,
and a ResNet classifier. This project instead applies the CEA loss to spatial
per-cell logits produced from obstacle/start/goal map channels. This is a
domain and representation adaptation; it does not reproduce the paper's
Rubik's Cube experiment or claim the paper's empirical results transfer to
these maps. The paper discusses varying goals in its instance-dependent
framework (Theorem 7, p. 7); this project conditions the per-cell CNN on a
one-hot goal channel, but does not claim that theorem's sample-complexity
result for this specific architecture/data distribution.

## Published Cross-Entropy Admissibility objective

The paper treats integer heuristic values as ordinal classes and uses a
softmax distribution `p_k` for each training state. Eq. (5), p. 8, is

```text
CEA_i = -log( sum_{k=1}^{h*_i} (k / h*_i)^beta * p_k^(i) )
        + eta * ( -log p_(h*_i)^(i) )
L_CEA = (1/N) * sum_i CEA_i
```

The first term rewards probability mass on admissible classes, with lower
classes discounted by `(k/h*)^beta`; smaller `beta` prioritizes admissibility
more strongly, while larger `beta` favors heuristic strength. The second term
is a scaled cross-entropy term for the exact target class. The paper states
`eta << 1` and describes the unique per-sample global optimum as placing all
probability on the true class. Appendix J (PDF pp. 26–27) starts at
`beta=1, eta=0.1` and describes lowering both when training progress and
overestimation improvements stall. Appendix G's Table 4 uses task-specific
values. Project defaults will be configurable and will not be presented as
paper-optimal. Softmax is evaluated stably and log arguments are floored at a
configurable `epsilon` (default `1e-12`) for numerical stability.

**Zero-cost class boundary:** the printed Eq. (5) has `h*` in its denominator
and its sum starts at class 1, so it is undefined at the goal (`h*=0`), which
is part of this project's targets. We retain actual cost labels `0, ..., L`;
for a zero-cost target we define its admissible-mass term as `-log(p_0)`, the
natural boundary case in which the only admissible class is the exact zero
class with unit weight. For positive targets, the implementation uses Eq. (5)
as printed; class zero receives zero weight in its first term, and the exact
target probability is included in the second. This boundary convention is an
explicit implementation adaptation and will be covered by a unit test.

For inference, the categorical estimate is the argmax cost class, matching the
paper's reported predicted-class and overestimation analyses. It is converted
to a nonnegative integer `h_pred`. Softmax/CEA training does not guarantee that
finite trained predictions are admissible; all constraint metrics are
calculated on the uncorrected prediction.

## Project-specific H1 and H2 losses

H1 is masked MSE on reachable traversable cells:

```text
L_fit = mean((h_pred - h_star)^2)
```

H2 is the proposed composite objective, distinct from CEA:

```text
L_fit  = mean(alpha * relu(h_pred-h_star)^2
               + beta_fit * relu(h_star-h_pred)^2)
L_adm  = mean(relu(h_pred-h_star)^2)
L_cons = mean_over_valid_directed_edges(
             relu(h_pred(u) - c(u,v) - h_pred(v))^2)
L_goal = mean_over_maps(h_pred(goal)^2)
L_total = lambda_fit*L_fit + lambda_adm*L_adm
          + lambda_cons*L_cons + lambda_goal*L_goal
```

`c(u,v)` is the shared Prompt 1/2 `get_transition_cost`, equal to `1.0` for
every legal orthogonal or diagonal move. The edge set is built with the shared
eight-connected successor generator and includes both directions of every
undirected legal edge whose endpoints have finite cost-to-go. These are soft
penalties and do not establish admissibility or consistency on unseen maps.

## Predictor, masks, and evaluation

The conventional heuristic predictor is a configurable CNN with input channels
for obstacle occupancy, one-hot start, and one-hot goal. H1/H2 produce a
spatial Softplus output; H3 produces a spatial softmax distribution over
integer distance classes. `h_pred` is the heuristic consumed by conventional
A* and remains independent of Prompt 3's learned `Phi` and accumulated `G_phi`.

Losses and MAE/RMSE include only traversable cells reachable from the goal.
Consistency is checked over valid directed edges in both directions. Raw
admissibility and consistency violation rates, maximum violations, and goal
error are reported without clipping or other correction. Validation may
control early stopping and checkpoint selection using D1–D3 only. D4–D6 are
used only after model selection for per-distribution OOD evaluation.
