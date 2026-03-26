# Architecture Overview

## The Core Innovation: Heterogeneous Bipartite Graphs for QFT

Traditional lattice QFT approaches (and existing ML methods like L-GCN) treat the lattice as a **homogeneous graph** where field values are embedded as features on spacetime nodes. This conflates geometry with field content.

Our architecture introduces a **heterogeneous bipartite graph** that formally separates these:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HETEROGENEOUS BIPARTITE GRAPH                    │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                    SPACETIME LAYER                          │   │
│   │                                                             │   │
│   │    (0,0)───adjacent───(1,0)───adjacent───(2,0)───adjacent───(3,0)
│   │      │                  │                  │                  │  │
│   │   adjacent           adjacent           adjacent           adjacent
│   │      │                  │                  │                  │  │
│   │    (0,1)───adjacent───(1,1)───adjacent───(2,1)───adjacent───(3,1)
│   │      │                  │                  │                  │  │
│   │   adjacent           adjacent           adjacent           adjacent
│   │      │                  │                  │                  │  │
│   │    (0,2)───adjacent───(1,2)───adjacent───(2,2)───adjacent───(3,2)
│   │      │                  │                  │                  │  │
│   │   adjacent           adjacent           adjacent           adjacent
│   │      │                  │                  │                  │  │
│   │    (0,3)───adjacent───(1,3)───adjacent───(2,3)───adjacent───(3,3)
│   │                                                             │   │
│   │    Features: [x¹, x², lattice_spacing a]                   │   │
│   │    Edges: 4-connected grid with direction vectors           │   │
│   │    (periodic BCs: edges wrap around boundaries)             │   │
│   └────────────────────────┬────────────────────────────────────┘   │
│                            │                                        │
│              ┌─────────────┼─────────────┐                          │
│              │   INHABITS EDGES (bipartite)                         │
│              │   Each field node connects to                        │
│              │   exactly one spacetime node                         │
│              │             │                                        │
│              │         inhabits                                     │
│              │         inhabits_inv                                 │
│              │             │                                        │
│   ┌──────────┴─────────────┴─────────────┴──────────────────────┐   │
│   │                     FIELD LAYER                             │   │
│   │                                                             │   │
│   │    φ(0,0)    φ(1,0)    φ(2,0)    φ(3,0)                   │   │
│   │    φ(0,1)    φ(1,1)    φ(2,1)    φ(3,1)                   │   │
│   │    φ(0,2)    φ(1,2)    φ(2,2)    φ(3,2)                   │   │
│   │    φ(0,3)    φ(1,3)    φ(2,3)    φ(3,3)                   │   │
│   │                                                             │   │
│   │    Features: [φ value] (scalar, 1 DOF per site)            │   │
│   │    No intra-field edges (fields couple through spacetime)   │   │
│   └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Bipartite?

In continuum QFT, the action functional has the form:

```
S[φ, g] = ∫ d⁴x √g [ ½ g^μν ∂_μφ ∂_νφ + ½ m²φ² + λφ⁴ ]
```

The geometry (metric `g_μν`) and field content (`φ`) are fundamentally different objects that couple through the action. Our graph mirrors this:

- **Spacetime nodes** encode geometry (coordinates, lattice spacing, metric)
- **Field nodes** encode field content (scalar values, spinor components, gauge links)
- **Inhabits edges** couple them (field at position x)
- **Adjacent edges** propagate information through the lattice (discretized derivatives)

This separation is impossible in a homogeneous graph where everything is a single node type.

---

## Graph Structure in PyG HeteroData

```
HeteroData(
  spacetime={
    x: [N², dim+1],              # [coordinates, lattice_spacing]
    num_nodes: N²
  },
  scalar={
    x: [N², 1],                  # [φ value]
    num_nodes: N²
  },
  (spacetime, adjacent, spacetime)={
    edge_index: [2, N²·2d],     # nearest-neighbor lattice edges
    edge_attr: [N²·2d, dim]     # unit direction vectors ±x̂, ±ŷ
  },
  (scalar, inhabits, spacetime)={
    edge_index: [2, N²],         # bipartite: field_i → spacetime_i
  },
  (spacetime, inhabits_inv_scalar, scalar)={
    edge_index: [2, N²],         # reverse: spacetime_i → field_i
  },
  y: scalar                      # target action S_E[φ]
)
```

For a 16×16 lattice: 256 spacetime nodes + 256 scalar nodes + 2,048 adjacency edges + 512 bipartite edges.

---

## Model Architecture: HeteroGNN

```
┌─────────────────────────────────────────────────────────────────────┐
│                          HeteroGNN                                  │
│                                                                     │
│  ┌───────────────────── ENCODING ─────────────────────┐             │
│  │                                                     │             │
│  │  Spacetime Encoder (MLP)    Field Encoder (MLP)     │             │
│  │  [x¹,x²,a] → h_st          [φ] → h_field           │             │
│  │       dim+1 → H                  1 → H              │             │
│  │                                                     │             │
│  │  Edge Encoder (MLP)                                 │             │
│  │  [±x̂, ±ŷ] → e_adj                                  │             │
│  │       dim → H                                       │             │
│  └─────────────────────────────────────────────────────┘             │
│                            │                                        │
│                            ▼                                        │
│  ┌────────── THREE-STAGE MESSAGE PASSING (× N blocks) ──────────┐   │
│  │                                                               │   │
│  │  ┌─ Stage 1: Field → Spacetime ───────────────────────────┐  │   │
│  │  │  Messages flow along INHABITS edges                     │  │   │
│  │  │  h_st += MLP([h_field ∥ h_st]) + residual + LayerNorm │  │   │
│  │  │                                                         │  │   │
│  │  │  Purpose: Aggregate field content at each geometry site │  │   │
│  │  └─────────────────────────────────────────────────────────┘  │   │
│  │                            │                                  │   │
│  │                            ▼                                  │   │
│  │  ┌─ Stage 2: Spacetime → Spacetime ───────────────────────┐  │   │
│  │  │  Messages flow along ADJACENT edges                     │  │   │
│  │  │  h_st += MLP([h_st_i ∥ h_st_j ∥ e_adj]) + res + LN   │  │   │
│  │  │                                                         │  │   │
│  │  │  Purpose: Propagate info along lattice (≈ derivatives) │  │   │
│  │  │  Each hop = one lattice spacing of receptive field      │  │   │
│  │  └─────────────────────────────────────────────────────────┘  │   │
│  │                            │                                  │   │
│  │                            ▼                                  │   │
│  │  ┌─ Stage 3: Spacetime → Field ───────────────────────────┐  │   │
│  │  │  Messages flow along INHABITS_INV edges                 │  │   │
│  │  │  h_field += MLP([h_st ∥ h_field]) + res + LayerNorm   │  │   │
│  │  │                                                         │  │   │
│  │  │  Purpose: Feed geometry back to fields (coupling)      │  │   │
│  │  └─────────────────────────────────────────────────────────┘  │   │
│  │                                                               │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                            │                                        │
│                            ▼                                        │
│  ┌───────────────────── READOUT ──────────────────────┐             │
│  │                                                     │             │
│  │  Energy Head:                                       │             │
│  │    S_E = Σ_x  MLP([h_st(x) ∥ h_field(x)]) · a^d   │             │
│  │              ↑                              ↑       │             │
│  │         per-site energy            lattice volume    │             │
│  │         density                    element           │             │
│  │                                                     │             │
│  │  Correlator Head (optional):                        │             │
│  │    G(x,y) ≈ project(h_field(x)) · project(h_field(y))           │
│  │                                                     │             │
│  └─────────────────────────────────────────────────────┘             │
│                                                                     │
│  Output: { energy: S_E[φ], local_energy: s(x), correlator: G(x,y) }│
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: End-to-End Pipeline

```
 ┌──────────────────────────────────────────────────────────────────┐
 │  1. MONTE CARLO DATA GENERATION                                  │
 │     scripts/generate_mc_data.py                                  │
 │                                                                  │
 │  LatticeConfig ──→ HypercubicLattice(16×16, periodic)           │
 │  ScalarFieldConfig ──→ Phi4Action(m²=-0.5, λ=0.5)              │
 │  MCConfig ──→ MetropolisSampler                                 │
 │                    │                                             │
 │                    ▼                                             │
 │  Thermalize (1000 sweeps) → Generate (10000 configs × 10 sweeps)│
 │                    │                                             │
 │                    ▼                                             │
 │  mc_data.pt: { configurations: [10000, 256],                    │
 │                actions: [10000],                                 │
 │                acceptance_rate: 0.45 }                           │
 └──────────────────────┬───────────────────────────────────────────┘
                        │
                        ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  2. GRAPH CONSTRUCTION                                           │
 │     HeteroGraphBuilder.build_dataset()                           │
 │                                                                  │
 │  For each configuration φ_i:                                     │
 │    Lattice geometry ──→ spacetime nodes + adjacency edges        │
 │    φ_i values ──→ scalar field nodes                             │
 │    Bipartite matching ──→ inhabits + inhabits_inv edges          │
 │                    │                                             │
 │                    ▼                                             │
 │  List[HeteroData] with target y = S_E[φ_i] from exact action   │
 └──────────────────────┬───────────────────────────────────────────┘
                        │
                        ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  3. TRAINING                                                     │
 │     scripts/train.py → Trainer                                   │
 │                                                                  │
 │  HeteroGNN learns to predict S_E[φ] from the graph structure:   │
 │                                                                  │
 │  Loss = MSE( S_GNN(φ_i), S_MC(φ_i) )                           │
 │                                                                  │
 │  Optimizer: AdamW, Scheduler: Cosine annealing                  │
 │  Checkpoints + TensorBoard logging every N epochs               │
 └──────────────────────┬───────────────────────────────────────────┘
                        │
                        ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │  4. PHYSICS ANALYSIS                                             │
 │     scripts/evaluate.py + scripts/sweep.py                       │
 │                                                                  │
 │  Coupling sweep at L = 8, 16, 32:                               │
 │    ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐ │
 │    │ Compute ξ/L │───→│ Find crossing│───→│ Extract ν from    │ │
 │    │ vs m² for   │    │ point → m²_c │    │ scaling collapse  │ │
 │    │ each L      │    └──────────────┘    │ Target: ν ≈ 1.000│ │
 │    └─────────────┘                        └───────────────────┘ │
 └──────────────────────────────────────────────────────────────────┘
```

---

## Phase Extension Architecture

The codebase uses abstract base classes so later phases add new implementations without modifying existing code:

```
Phase 1 (current)          Phase 2 (U(1)+Fermions)     Phase 3 (SU(3) 4D)
─────────────────          ───────────────────────      ──────────────────
Lattice:                   Lattice:                     Lattice:
  HypercubicLattice(2D)     HypercubicLattice(2D/3D)    HypercubicLattice(4D)

Field:                     Field:                       Field:
  ScalarField                ScalarField                  ScalarField
                           + GaugeField (U(1) links)    + GaugeField (SU(3) 3×3)
                           + FermionField (spinors)      + FermionField (4-spinor)

Action:                    Action:                      Action:
  Phi4Action                 Phi4Action                   Phi4Action
                           + WilsonGaugeAction           + WilsonGaugeAction
                           + FermionAction               + FermionAction

MC Sampler:                MC Sampler:                  MC Sampler:
  MetropolisSampler          MetropolisSampler            MetropolisSampler
                           + HMCSampler                  + HMCSampler

Message Passing:           Message Passing:             Message Passing:
  ThreeStageBlock            ThreeStageBlock              ThreeStageBlock
                           + GaugeEquivariantMP         + SU3EquivariantMP

HeteroGraphBuilder:        HeteroGraphBuilder:          HeteroGraphBuilder:
  [ScalarField]              [Scalar, Gauge, Fermion]     [Scalar, Gauge, Fermion]
  (no changes needed)        (no changes needed)          (no changes needed)
```

### Key Extension Points

| Component | How it extends | What stays the same |
|-----------|---------------|-------------------|
| `HypercubicLattice` | `dimensions=(N,N,N,N)` for 4D | Same class, same interface |
| `HeteroGraphBuilder` | Pass more `Field` objects in list | `build()` method unchanged |
| `HeteroGNN` | `field_types={"scalar":1, "gauge":18}` | Dynamic encoder creation |
| `ThreeStageBlock` | `field_dims={"scalar":H, "gauge":H}` | Same 3-stage structure |
| `EnergyHead` | Concatenates all field embeddings | Same MLP readout |

---

## Module Dependency Graph

```
config.py
    │
    ├──→ lattice/
    │      ├── base.py (ABC)
    │      ├── hypercubic.py
    │      └── boundary.py
    │
    ├──→ fields/
    │      ├── base.py (ABC)
    │      ├── scalar.py
    │      ├── gauge.py (stub)
    │      └── fermion.py (stub)
    │
    ├──→ actions/
    │      ├── base.py (ABC)
    │      └── phi4.py ──→ lattice, fields
    │
    ├──→ mc/
    │      ├── sampler.py (ABC)
    │      ├── metropolis.py ──→ actions
    │      ├── observables.py ──→ lattice
    │      └── analysis.py
    │
    ├──→ graphs/
    │      ├── node_types.py
    │      ├── edge_types.py
    │      ├── builder.py ──→ lattice, fields (produces PyG HeteroData)
    │      └── transforms.py
    │
    ├──→ models/
    │      ├── encoders/ ──→ (standalone MLPs)
    │      ├── message_passing/
    │      │     ├── field_to_st.py ──→ PyG MessagePassing
    │      │     ├── st_to_st.py ──→ PyG MessagePassing
    │      │     ├── st_to_field.py ──→ PyG MessagePassing
    │      │     └── stage.py ──→ edge_types, all 3 MP layers
    │      ├── heads/
    │      │     ├── energy.py
    │      │     └── correlator.py
    │      └── hetero_gnn.py ──→ encoders, stage, heads
    │
    ├──→ training/
    │      ├── losses.py
    │      ├── trainer.py ──→ models, losses, metrics, checkpointing
    │      ├── callbacks.py
    │      └── metrics.py
    │
    ├──→ analysis/
    │      ├── critical.py
    │      ├── correlation.py
    │      ├── phase_diagram.py
    │      └── visualization.py
    │
    └──→ utils/
           ├── reproducibility.py
           ├── checkpointing.py
           └── logging.py
```

---

## Over-Smoothing Mitigation

Deep GNNs risk **over-smoothing**: all node embeddings converge to the same vector, destroying local field information. This is especially dangerous for QFT where short-range correlations matter. Our mitigations:

1. **Residual connections** in every `ThreeStageBlock`: output = input + messages
2. **Layer normalization** per node type after each message passing stage
3. **Limited stacking depth**: 2–4 blocks (receptive field of 2–4 lattice spacings per block)
4. **Separate encoders per type**: spacetime and field representations never mix in the encoder
5. **Monitoring**: track cosine similarity between embeddings vs graph distance during training

---

## Physics Background

### Scalar φ⁴ Theory (Phase 1)

The Euclidean action on a 2D lattice:

```
S_E[φ] = Σ_x a² [ ½ Σ_μ (φ(x+μ̂) - φ(x))² / a² + ½ m² φ(x)² + λ φ(x)⁴ ]
```

- For `m² < m²_c`: ordered (ferromagnetic) phase, spontaneous symmetry breaking
- For `m² > m²_c`: disordered (paramagnetic) phase
- At `m² = m²_c`: second-order phase transition in the **Ising universality class**
- Critical exponent: **ν = 1** (exact, from conformal field theory)

### Finite-Size Scaling

The correlation length ξ diverges at the critical point: `ξ ~ |m² - m²_c|^{-ν}`.
On a finite lattice of size L, the ratio `ξ/L` is scale-invariant at criticality.
Plotting `ξ/L` vs `m²` for different L values:
- Curves **cross** at `m² = m²_c`
- The slope at crossing determines `ν`
- Data **collapse** onto a single curve when plotted vs `(m² - m²_c) · L^{1/ν}`

---

## Monte Carlo Samplers

Two sampler implementations are provided, automatically selected by `create_sampler()`:

```
┌─────────────────────────────────────────────────────────────┐
│  MetropolisSampler (sequential, L ≤ 16)                     │
│                                                             │
│  for site in random_permutation(all_sites):                 │
│      propose φ'(site) = φ(site) + N(0, step)               │
│      ΔS = delta_action(φ, site, φ')                        │
│      accept with probability min(1, exp(-ΔS))              │
│                                                             │
│  ~400 sweeps/s on 16×16, correct for any lattice           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  CheckerboardSampler (vectorized, L ≥ 32)                   │
│                                                             │
│  1. Partition sites into even/odd by parity of (x+y+...)   │
│  2. No two same-parity sites are neighbors                  │
│  3. Update ALL even sites simultaneously (vectorized numpy) │
│  4. Update ALL odd sites simultaneously                     │
│                                                             │
│  10-50x faster than sequential for large lattices           │
│  Essential for 64×64 (4096 sites)                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Correlation Length Estimators

Two methods for extracting ξ from the two-point function G(r):

### Log-slope (simple, noisy)
Fit log(G(r)) = log(A) - r/ξ via linear regression.
Fragile: fails when G(r) hits noise floor at large r.

### Second-moment (robust, preferred)
Uses the Fourier transform of G(r):

```
ξ = (1 / 2sin(π/L)) × √( G̃(0)/G̃(k_min) - 1 )
```

where k_min = 2π/L is the smallest non-zero momentum.
This is the standard lattice field theory estimator — it uses all
data points and doesn't require fitting. Implemented as
`ObservableSet.correlation_length_second_moment()`.
