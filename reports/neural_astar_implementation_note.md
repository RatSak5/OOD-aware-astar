# Neural A* implementation note

## Source and scope

This note is based on the complete supplied paper, Yonetani et al., *Path
Planning using Neural A* Search* (ICML 2021), including its appendix (the PDF is
16 pages). Main-paper page references below are the printed page numbers, which
match the PDF page numbers for the cited sections. Appendix references identify
both the appendix section and PDF page. The source file is
`C:/Users/ratul/Downloads/2009.07476v3.pdf`.

The project uses the paper's binary grid-world path-planning task and guidance
formulation. It uses the project's existing map encoding, shared eight-neighbor
successor function, and strict diagonal corner rule. These graph and software
adaptations are identified explicitly below.

## Input and guidance-map representation

For each instance, the paper's input is a map `X` plus start and goal locations
(`Q=(X, v_s, v_g)`, Sec. 2, p. 3). In its explicit binary cost-map setting,
`X(v)=1` denotes passable and `X(v)=0` denotes impassable; the ground-truth
node-entry cost is one for passable cells and infinity for obstacles (Sec. 2,
p. 3). The project's stored grid uses the inverse convention (`0` free,
`1` blocked), so the Neural A* encoder receives `X = 1 - grid`.

The encoder input is the channel concatenation `[X, V_s + V_g]`, where `V_s`
and `V_g` are one-hot endpoint maps (Sec. 3.2, p. 5). Its output is one scalar
per cell, the learned guidance-cost map `Phi`. The paper uses a fully
convolutional U-Net with a VGG-16 backbone and a final sigmoid, which bounds
`Phi` in `[0, 1]` (Sec. 4.2, p. 6). `Phi(v)` is a learned per-cell guidance
cost. It is not a movement-direction weight, ground-truth transition cost,
conventional heuristic, or physical distance.

## Differentiable node selection and guidance-cost accumulation

The paper's differentiable selector (Eq. (3), p. 4) is

```text
V* = I_max( [ exp(-(G + H) / tau) elementwise-multiplied by O ]
            / < exp(-(G + H) / tau), O > )
```

`O` is the binary OPEN mask, `G` is accumulated guidance cost, `H` is the
search heuristic, `<.,.>` is the map-wise inner product, and `I_max` returns a
one-hot argmax in the forward pass while acting as identity in backpropagation.
Thus the hard forward choice is the OPEN cell minimizing `G + H`. The
normalization is over OPEN cells only. The paper uses Chebyshev distance for
`H` and adds `0.001 * EuclideanDistance` to break ties; it sets `tau` to the
square root of map width (Sec. 4.2, p. 6; Appendix A.2, Table 4, PDF p. 12).
For this project's unit-cost eight-connected graph, Chebyshev distance is the
appropriate base heuristic. The small Euclidean term reproduces the paper's
node-ordering tie-break; it is not a ground-truth path cost or a claim of
admissibility for the full tie-broken value.

For the paper's explicit binary cost-map case, its eight-neighbor kernel `K`
has ones in a 3-by-3 window except at the center. The newly discovered OPEN
neighbors and already-OPEN neighbors are (Eq. (4), p. 4):

```text
V_nbr     = (V* convolved with K) elementwise-multiplied by X, (1-O), (1-C)
Vbar_nbr  = (V* convolved with K) elementwise-multiplied by X, O, (1-C)
```

`C` is CLOSED. Obstacles are excluded by `X`; already-open neighbors are kept
separate so their costs can be improved. The guidance cost proposed for each
neighbor is (Eq. (6), p. 4):

```text
G' = <G, V*> * 1 + Phi
```

The map-wide update (Eq. (5), p. 4) is:

```text
G = G' * V_nbr + min(G', G) * Vbar_nbr
    + G * (1 - V_nbr - Vbar_nbr)
```

Products here are elementwise. In scalar terms, entering neighbor `v` from
selected cell `u` proposes `G(u) + Phi(v)`. A newly opened neighbor receives
that value; an already OPEN neighbor keeps the lower of its previous value and
the proposal; all other cells retain their current value. Algorithm 2 (p. 5)
initializes `O=V_s`, `C=0`, `G=0`, and `Parent(v_s)=empty`. Consequently, the
start cell's guidance cost is not charged, and the guidance cost of every
entered cell, including the goal, is accumulated. The goal test occurs when
the goal is selected into CLOSED (Algorithm 2, pp. 4–5).

For stable evaluation, subtracting the minimum finite `G+H` over OPEN before
exponentiation leaves the normalized selector unchanged. The sigmoid is the
paper's output-range constraint; there is no map-wise normalization of `Phi`,
no `1-Phi` conversion, and no directional edge-weight transform.

## Training objective and gradient treatment

The supervision target `P_bar` is the binary Dijkstra shortest-path map for
explicit maps (Sec. 2, p. 3; Sec. 4.1, p. 5). The differentiable search output
is its CLOSED mask `C`. The training loss is the mean L1 discrepancy (Eq. (7),
p. 4):

```text
L = || C - P_bar ||_1 / |V|
```

This penalizes both missed ground-truth path cells and excess expansions. The
paper detaches `O`, `V_nbr`, `Vbar_nbr`, and `G` from the backpropagation chain
to stabilize recurrent training and reduce memory (Sec. 3.2, p. 4). In
mini-batch search, the goal-verification flag and mask updates in Eq. (8),
p. 5, keep a completed instance's OPEN and CLOSED masks unchanged while other
batch members continue. The paper reports RMSProp, learning rate `0.001`, and
batch size `100` in its experiments (Sec. 4.3, p. 6; Appendix A.2, Table 4,
PDF p. 12); those dataset-sized settings are configurable rather than imposed
on this project's small generated-map experiments.

## Connectivity, costs, and project adaptations

The paper uses eight-neighbor connectivity and unit cost per entered passable
cell (Sec. 2, p. 3). It does not assign `sqrt(2)` to diagonal moves. The
project's ground-truth cost remains the shared `get_transition_cost` result:
`1.0` for each legal move, orthogonal or diagonal. Dijkstra, conventional A*,
heuristic supervision, and optimal-path evaluation all use this cost. Neural
A*'s `Phi` and accumulated `G` are separate guidance quantities and do not
replace the shared ground-truth transition cost.

**Explicit graph adaptation:** the paper's 3-by-3 neighbor kernel permits a
diagonal whenever the destination cell is passable, even if the move passes
between blocked orthogonal cells. The project requires no corner-cutting. Its
Neural A* expansion therefore applies the shared `get_neighbors` rule and
omits such diagonals from both `V_nbr` and `Vbar_nbr`. This changes the graph's
legal edges, not Eqs. (3), (5), (6), or (7). The stored map uses blocked=1,
whereas the paper's explicit passability mask uses passable=1; the encoder and
search mask use `1-grid`.

**Architecture adaptation:** the implemented encoder is a project-local
U-Net with the VGG-16 encoder block counts and channel widths. Its weights are
initialized by PyTorch defaults; no pretrained VGG-16 checkpoint is assumed or
downloaded. The encoder input, sigmoid output, guidance semantics,
differentiable search, and L1 search-output objective follow the paper. This
is an explicit weight-initialization and implementation adaptation, rather
than a claim to use the authors' exact reference implementation. The decoder
uses bilinear size-aligned upsampling and ceil-mode pooling to support odd map
dimensions; this local decoder detail is also an implementation adaptation.

**Training-loop adaptation:** variable-length searches are run one map at a
time within each configured mini-batch, and their Eq. (7) losses are averaged
before a single optimizer update. Each search stops independently on selecting
its goal, as in Eq. (8). This preserves the stated objective and per-instance
termination while not implementing the paper's vectorized parallel search.

## Quantities kept separate in code and reports

- `Phi`: predicted learned per-cell guidance cost in `[0,1]`.
- `G_phi`: accumulated guidance cost used by Neural A* selection.
- `get_transition_cost`: ground-truth unit cost of a legal graph edge.
- `distance_map` and `optimal_path_cost`: Dijkstra ground truth.
- `H`: Neural A* search heuristic and paper tie-break term.
- `g`, `h`, `f=g+h`: conventional A* quantities from Prompt 2.
- Physical path length and horizontal/vertical/diagonal counts: separate
  measurements, never substitutes for the search cost.
