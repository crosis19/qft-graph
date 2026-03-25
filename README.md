# QFT-Graph: Heterogeneous Graph Neural Networks for Quantum Field Theory

A novel computational framework that uses **heterogeneous graph neural networks** to model quantum field theories on the lattice. The key innovation is a **bipartite graph structure** that separates spacetime geometry from field content — a physically motivated architecture absent from published literature.

## Motivation

Lattice quantum field theory (QFT) is the primary non-perturbative tool for studying strongly coupled quantum systems, but faces fundamental computational bottlenecks: critical slowing down, the fermion sign problem, and the inability to directly simulate real-time dynamics. Existing ML approaches embed field values as features on spacetime nodes, treating the lattice as a homogeneous graph.

This project introduces a **heterogeneous bipartite graph** where spacetime nodes and field nodes are distinct types connected by typed edges. This mirrors the geometry–matter separation fundamental to continuum QFT and enables:

- Cleaner multi-field handling (scalar, spinor, gauge representations at the same site)
- Dynamic geometry (learnable spacetime positions for adaptive discretization)
- Natural extension to curved spacetime (metric as edge features)
- Renormalization group flow as graph coarsening

## Project Phases

| Phase | Duration | Theory | Goal |
|-------|----------|--------|------|
| **I** (current) | 3 months | Scalar φ⁴ in 2D | Recover Ising critical exponent ν ≈ 1 |
| II | 4 months | U(1) Gauge + Fermions | Gauge-equivariant message passing |
| III | 6 months | SU(3) Yang–Mills in 4D | Benchmark against lattice QCD |
| IV | 4 months | Wick Rotation Bridge | Euclidean → Minkowski spectral functions |
| V | 6 months | Native Minkowski | Real-time dynamics via complex Langevin |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/qft-graph.git
cd qft-graph

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev]"
```

### Dependencies

- Python ≥ 3.10
- PyTorch ≥ 2.1
- PyTorch Geometric ≥ 2.4
- NumPy, SciPy, OmegaConf, Matplotlib, TensorBoard

### Generate Monte Carlo Data

```bash
python scripts/generate_mc_data.py \
    --dimensions 16 16 \
    --mass_squared -0.5 \
    --coupling 0.5 \
    --n_configs 10000
```

### Train the Model

```bash
python scripts/train.py \
    --data data/mc_configs/phi4_16x16_m2=-0.5_lam=0.5/mc_data.pt \
    --config configs/defaults.yaml \
    --experiment_name phi4_16x16_run1
```

### Run Coupling Sweep (for critical exponents)

```bash
python scripts/sweep.py \
    --dimensions 8 8 \
    --m2_min -1.0 --m2_max 0.0 --m2_steps 20 \
    --n_configs 5000

python scripts/sweep.py \
    --dimensions 16 16 \
    --m2_min -1.0 --m2_max 0.0 --m2_steps 20 \
    --n_configs 5000

python scripts/sweep.py \
    --dimensions 32 32 \
    --m2_min -1.0 --m2_max 0.0 --m2_steps 20 \
    --n_configs 5000
```

### Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint experiments/runs/phi4_16x16_run1/checkpoint_final.pt \
    --data data/mc_configs/phi4_16x16_m2=-0.5_lam=0.5/mc_data.pt
```

## Project Structure

```
qft_graph/
├── configs/                    # YAML configuration files
│   ├── defaults.yaml           # Global defaults
│   ├── lattice/                # Lattice size configs (8x8, 16x16, 32x32)
│   ├── model/                  # Model architecture configs
│   ├── training/               # Training hyperparameters
│   └── mc/                     # Monte Carlo sampler configs
├── src/qft_graph/              # Main package
│   ├── lattice/                # Spacetime geometry (N-dim hypercubic)
│   ├── fields/                 # Quantum fields (scalar, gauge*, fermion*)
│   ├── graphs/                 # Heterogeneous graph construction
│   ├── actions/                # Lattice action functionals (φ⁴)
│   ├── mc/                     # Monte Carlo sampling + observables
│   ├── models/                 # GNN architecture
│   │   ├── encoders/           # Type-specific node/edge encoders
│   │   ├── message_passing/    # 3-stage message passing blocks
│   │   └── heads/              # Energy + correlator readout heads
│   ├── training/               # Training loop, losses, metrics
│   ├── analysis/               # Critical exponents, correlations, plots
│   └── utils/                  # Reproducibility, checkpointing, logging
├── scripts/                    # CLI entry points
├── tests/                      # Test suite (pytest)
├── notebooks/                  # Jupyter exploration notebooks
├── data/                       # Generated data (gitignored)
└── experiments/                # Run logs and checkpoints (gitignored)
```

*\* Stubs for Phase 2+*

## Running Tests

```bash
pytest tests/ -v
```

## Configuration

All parameters are managed via hierarchical dataclass configs loaded from YAML with [OmegaConf](https://omegaconf.readthedocs.io/). Override any parameter from the command line or by composing YAML files:

```yaml
# configs/defaults.yaml
lattice:
  dimensions: [16, 16]
  spacing: 1.0
  boundary: periodic

field:
  mass_squared: -0.5
  coupling: 0.5

model:
  hidden_dim: 64
  n_mp_blocks: 3
  activation: gelu

training:
  epochs: 200
  batch_size: 32
  lr: 0.001
  loss: energy_matching
```

## Phase 1 Validation Targets

| Observable | Benchmark | Method |
|-----------|-----------|--------|
| Critical exponent ν | 1.000 ± 0.001 | Finite-size scaling at L = 8, 16, 32 |
| Two-point function G(r) | MC baseline | Direct comparison |
| Phase transition location | Known m²_c | Susceptibility peak / ξ/L crossing |

## Key References

- Bachtis, Aarts & Lucini (2021) — Closest prior work (different bipartition)
- Favoni et al. (2022) — L-GCN lattice gauge architecture
- Kanwar et al. (2020) — Equivariant normalizing flows for lattice QFT
- Boyda et al. (2021) — SU(N) gauge-equivariant sampling

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed overview of the heterogeneous graph structure and model architecture.
