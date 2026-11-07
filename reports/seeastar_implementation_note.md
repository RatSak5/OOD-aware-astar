# Original and conventional unit-cost SeeA* implementation note

## Source and scope

This note is based on Zhao, Tu, and Xu, *SeeA*: Efficient Exploration-Enhanced
A* Search by Selective Sampling (NeurIPS 2024), including its supplementary
material and paper checklist, in the supplied
`C:/Users/ratul/Downloads/NeurIPS-2024-seea-efficient-exploration-enhanced-a-search-by-selective-sampling-Paper-Conference.pdf`.

This note covers the Original SeeA* strategies and the conventional
unit-transition-cost uniform SeeA* API. It does not specify the separate
Neural-guidance fixed-K or OOD-adaptive-K methods; those corrected methods are
documented in the [Neural-guidance adaptive SeeA* note](ood_adaptive_seeastar_implementation_note.md).

The paper does not define one unique sampling rule as “the” SeeA* default.
Algorithm 1 is a generic search procedure parameterized by a candidate sampler
`F`; Algorithms 2, 3, and 4 give its uniform, competitive-clustering, and
UCT-like candidate strategies. The implementation keeps this generic search
separate from those three paper strategies and reports all three in
comparisons. The research-facing `uniform_seeastar` API is a separate wrapper
with its own `UniformSeeAStarConfig`; that config exposes no cluster or UCT
options, so the Prompt 7 foundation can only run uniform sampling.

## Search setup and equations

The paper's A* evaluation is

```text
f(n) = g(n) + h(n)                                               (Eq. 1, p. 2)
```

`g` is accumulated cost from the root, `h` estimates remaining cost, and the
OPEN set contains waiting nodes; CLOSED contains expanded nodes (Sec. 3, p. 4).
Algorithm 1 (supplement, p. 15) replaces A*'s global minimum selection with:

```text
D = F(OPEN, K)
n = argmin_{x in D} f(x)
if n is goal: return n
move n from OPEN to CLOSED; add its children to OPEN
repeat until OPEN is empty or t >= Tmax
```

Thus sampling and ranking are distinct: candidate generation is followed by
selection of the candidate with minimum `f`. `K` is fixed for a search and
controls the exploitation/exploration balance; `K=1` makes the sampler choose
the expanded state, while `K >= |OPEN|` generally reduces selection to A*.
The UCT-like strategy is the exception in that it ranks candidate membership
using its exploration score, then still expands minimum-`f` within that set.

### Published candidate strategies

1. **Uniform (Algorithm 2, supplement p. 15; Sec. 4.1.1, p. 5):** when
   `|OPEN| <= K`, return OPEN. Otherwise choose K distinct OPEN nodes uniformly
   without replacement; expand the candidate with minimum `f`. This behavior is
   available through the dedicated `uniform_seeastar` function as well as the
   generic paper reproduction.
2. **Competitive clustering (Algorithm 3 and 5, supplement pp. 15-16; Sec.
   4.1.2, p. 5):** maintain `Nc` clusters of OPEN nodes. Each new node is
   represented by `fh(n)`, assigned to its nearest center, and updates that
   center by `w_j <- w_j + eta * (fh(n) - w_j)`. Let
   `Kc = ceil(K/Nc)`. From each cluster, keep all nodes if it has at most Kc;
   otherwise sample Kc uniformly without replacement. The union is D, then
   minimum `f` in D is expanded. The union can exceed K because of the ceiling
   and the per-cluster rule; the implementation preserves this behavior.
3. **UCT-like (Eq. 4, Algorithm 4, supplement p. 16; Sec. 4.1.3, p. 6):**

   ```text
   E(n) = f(n) - cb * sqrt(dmax / (1 + d(n)))
   ```

   When `|OPEN| <= K`, return OPEN. Otherwise take the K nodes with the
   smallest E as D and expand the minimum-f node in D. `d(n)` is root depth and
   `dmax` is maximum OPEN depth.

The paper's clustering uses task-specific node embeddings. The grid project
has no published SeeA* node embedding, so the adaptation uses normalized map
coordinates `(row/(H-1), col/(W-1))` as `fh(n)`, with random initial centers
and the paper's online competitive update. This groups spatially nearby OPEN
cells; it does not claim to reproduce the molecule/AIG feature embeddings.

## Analysis assumptions and complexity

Assumption 4.1 (main paper p. 6) models `f` on optimal-path nodes as uniform
around a common optimal value `mu_f0` with width `2 sigma`; off-path `f` is
uniform around its true `f*`, and off-path `f*` values are iid Gaussian with
mean `mu_f1 > mu_f0` and variance `sigma_s^2`. Under that model the pairwise
ordering probability `p_sigma` falls as error grows. For `No` OPEN states,
`P_A = p_sigma^(No-1)` and uniform-sampling SeeA* has
`P_S = (K/No) * p_sigma^(K-1)` (Eqs. 5-8). Theorem 4.3 states

```text
P_S > P_A iff p_sigma < (K/No)^(1/(No-K)), for No > K >= 1.    (Eq. 9, p. 7)
```

This is a conditional probability result, not a universal runtime or
optimality guarantee. The paper notes that when `f` is exact (`p_sigma=1`),
A* is more efficient; it also derives an illustrative best K of
`-1/log(p_sigma)` (p. 7). Appendix C supplies uniform- and Gaussian-error
derivations (supplement pp. 16-21).

Per expansion, the implementation materializes OPEN in coordinate-sorted order
for deterministic behavior, costing O(|OPEN| log |OPEN|). Uniform selection
then samples K nodes and ranks them in O(K). UCT scans OPEN and deterministically
selects its K lowest exploration scores in O(|OPEN| log K) using bounded
selection, followed by candidate ranking. Clustering assigns each new node in
O(Nc * d) for embedding dimension d and samples by cluster; maintaining a
stable coordinate order in each cluster adds sorting cost. A* and Dijkstra use
priority queues with their usual
O((|V|+|E|) log |V|) upper bound on a finite graph.

The paper's path-finding appendix (N, p. 30) uses unit step cost and tests
Euclidean h plus a noisy `h_hat ~ Uniform(0, 2h)`. Here diagonals also cost one,
so Euclidean distance can overestimate. All project comparisons therefore use
the project's admissible, consistent Chebyshev heuristic by default, or the
same caller-supplied heuristic for A* and each SeeA* strategy. The project
does not silently copy the paper's Euclidean heuristic into a different
movement-cost graph.

## Project-specific graph adaptations

- For the Original and conventional-cost searches covered here, the project's
  shared eight-connected graph and no-corner-cutting rule define successors.
  Every legal transition costs 1.0, matching the project's `unit_move_count`
  convention and the paper's path-finding step cost. Dijkstra, A*, and these
  SeeA* APIs all use the same transition-cost function.
- The paper's pseudocode describes search-tree nodes and does not spell out
  duplicate handling for cyclic state spaces. Grid maps are searched as a
  graph: each coordinate is one state, duplicate discoveries are merged, a
  better g updates the parent, and a CLOSED state is reopened when needed. The
  cluster sampler keeps one cluster membership per coordinate state.
- The uniform-sampling variant uses precisely the paper's Algorithm 2 candidate
  construction and Algorithm 1 minimum-f candidate selection. The dedicated
  wrapper shares the graph-search update code with the generic reproduction,
  but its configuration only exposes K, seed, an optional expansion cap, and
  cost tolerance; clustering and UCT are unavailable through that interface.
- The cluster algorithm's supplement does not repeat the `|OPEN| <= K` guard
  shown explicitly for uniform and UCT strategies. For this project, when
  `|OPEN| <= K` the cluster strategy returns all OPEN states. This small-open
  guard avoids discarding states when the candidate budget already covers the
  graph frontier; above that case its per-cluster sampling follows Algorithm 3.
- The generic `Tmax` is an optional maximum number of non-goal expansions;
  `None` means no expansion cap. K, random seed, Nc, eta, and cb are
  configurable and fixed for the whole search in the APIs covered here. The
  separate OOD-adaptive Neural-guidance API is documented in the note linked
  above.
  Benchmark rows use matched maps and endpoints. The base seed, search
  configuration, cost model, and per-map outcomes are saved; sampler seeds are
  derived deterministically from the base seed, map seed, map index, and
  strategy index.

Reported defaults use K=50, Nc=5, eta=0.15, and cb=0.35 as configurable
starting values drawn from the paper's retrosynthesis setting (p. 8), not as
validated grid-world hyperparameters. The paper also reports different values
for logic synthesis and Sokoban (supplement pp. 27, 29). SeeA* does not promise
optimality under arbitrary h or sampling; the evaluator reports the found-cost
ratio against Dijkstra rather than assuming the ratio is one.
