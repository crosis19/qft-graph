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
| **I** (complete — φ⁴ paper in v2 journal revision, MLST) | 3 months | Scalar φ⁴ in 2D | Recover Ising critical exponent ν ≈ 1 |
| **II** (IIa gauge study complete 2026-07; IIb fermions pending) | 4 months | U(1) Gauge + Fermions | Gauge-equivariant message passing |
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

The sampler automatically selects the best algorithm: sequential Metropolis for small lattices (≤16×16) and vectorized checkerboard decomposition for large lattices (≥32×32), giving 10–50x speedups. For critical-region sweeps, a Wolff/Brower–Tamayo embedded-cluster sampler (`src/qft_graph/mc/cluster.py`) removes critical slowing down (τ_int ≈ 185 → ≈ 8–9 at L = 64, up to 21x faster decorrelation); enable it with `scripts/sweep.py --sampler cluster`.

```bash
# Small lattice (sequential sampler)
python scripts/generate_mc_data.py \
    --dimensions 16 16 \
    --mass_squared -0.5 \
    --coupling 0.5 \
    --n_configs 10000

# Large lattice (auto-selects checkerboard sampler)
python scripts/generate_mc_data.py \
    --dimensions 64 64 \
    --mass_squared -0.5 \
    --coupling 0.5 \
    --n_configs 5000
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
    --m2_min -2.5 --m2_max 0.0 --m2_steps 20 \
    --n_configs 5000

python scripts/sweep.py \
    --dimensions 16 16 \
    --m2_min -2.5 --m2_max 0.0 --m2_steps 20 \
    --n_configs 5000

python scripts/sweep.py \
    --dimensions 32 32 \
    --m2_min -2.5 --m2_max 0.0 --m2_steps 20 \
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
│   ├── fields/                 # Quantum fields (scalar, U(1) gauge, fermion*)
│   ├── graphs/                 # Heterogeneous graph construction
│   ├── actions/                # Lattice action functionals (φ⁴, U(1) Wilson)
│   ├── mc/                     # Monte Carlo sampling + observables
│   ├── models/                 # GNN architecture
│   │   ├── encoders/           # Type-specific node/edge encoders
│   │   ├── message_passing/    # 3-stage message passing blocks
│   │   └── heads/              # Action (S_E) + correlator readout heads (output key: `energy`)
│   ├── training/               # Training loop, losses, metrics
│   ├── analysis/               # Critical exponents, correlations, plots
│   └── utils/                  # Reproducibility, checkpointing, logging
├── scripts/                    # CLI entry points
├── tests/                      # Test suite (pytest)
├── notebooks/                  # Jupyter exploration notebooks
├── paper/                      # Phase I paper (LaTeX; `make -C paper`, `make -C paper submission`)
├── paper_gauge/                # Phase IIa paper draft (learned vs. exact gauge invariance)
├── docs/                       # Decision records and plans (phase2_decisions.md, ...)
├── results/                    # Committed analysis + provenance JSONs (fss_analysis_cluster.json, cluster_fss/ snapshot)
├── data/                       # Generated data (gitignored)
└── experiments/                # Run logs and checkpoints (gitignored)
```

*\* `fermion.py` is a stub for Phase IIb; gauge fields are fully implemented (Phase IIa).*

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

## Phase 1 Results

All entries are mean ± std over 5 seeds (3 seeds for ablations); every number
traces to a committed script + `results/*.json` provenance record.

| Observable | Result | Method |
|-----------|--------|--------|
| Action prediction (8×8 / 16×16) | r = 0.99999 / 0.99998 | GNN vs exact action, held-out sets |
| Action prediction (64×64) | r = 0.981 ± 0.036 (4/5 seeds ≥ 0.9989) | one slow-converging seed, see paper |
| Size transfer (train 16×16 → eval 8–64) | r = 1.0000 at all sizes | single model, no retraining |
| Critical exponent ν | 1.04 ± 0.08 | pseudo-critical-shift fits, cluster ensembles, 7 sizes L = 16–128 (exact: ν = 1) |
| Critical point m²_c | −2.217 ± 0.003 (λ=0.5) | χ-peak pseudo-critical extrapolation, cluster ensembles |
| Susceptibility scaling γ/ν | 1.732 ± 0.008 | ln χ_max vs ln L, Wolff/Brower–Tamayo cluster sampler (exact: 1.75); local-Metropolis comparison 1.60 ± 0.03 is biased by critical slowing down |
| Over-smoothing ablation | no-skip homogeneous GNN collapses by B=6 | depth study, 3 variants × 5 depths |

### Reproducing every figure and table

```bash
# Table I (action prediction; one 5-seed run per lattice size)
python scripts/generate_mc_data.py --dimensions 16 16 --mass_squared -0.5 --coupling 0.5 --n_configs 5000
python scripts/train_baselines.py --data data/mc_configs/phi4_16x16_m2=-0.5_lam=0.5/mc_data.pt \
    --config configs/paper/phase1_train_16x16.yaml --models HeteroGNN --seeds 0 1 2 3 4 \
    --output results/baseline_16x16_v2.json      # repeat per size with matching config

# Table II (architecture comparison at 16x16, all models)
python scripts/train_baselines.py --data data/mc_configs/phi4_16x16_m2=-0.5_lam=0.5/mc_data.pt \
    --config configs/paper/phase1_train_16x16.yaml --seeds 0 1 2 3 4

# Table III' (multi-coupling generalization)
python scripts/generate_multicoupling_data.py
python scripts/train_multicoupling.py --seeds 0 1 2 3 4

# Fig. 2 + FSS numbers (cluster-sampler pipeline, 7 sizes L=16..128; the local run
# survives only as the comparison arm, results/fss_analysis_v5.json)
python scripts/run_cluster_fss.py            # base + chi-peak cluster sweeps -> data/sweep_cluster{,_peak} (gitignored)
#   (peak grids were later refined to uniform dense scans via scripts/dense_peak_scan.py;
#    the committed point-data snapshot results/cluster_fss/{base,peak} already reflects
#    this, so the analysis below reproduces the committed numbers without re-running sweeps)
python scripts/analyze_fss.py --sweep_dir results/cluster_fss/base \
    --peak_dir results/cluster_fss/peak --sizes 16 24 32 48 64 96 128 \
    --peak_n_points 9 --output results/fss_analysis_cluster.json
python scripts/tau_int_comparison.py         # matched local-vs-cluster autocorrelation run
python scripts/assemble_fss_comparison.py    # -> results/fss_local_vs_cluster.json
python scripts/make_fss_numbers.py           # -> paper/fss_numbers.tex (every FSS number in the paper)

# Depth ablation figure / size-transfer table
python scripts/run_depth_ablation.py --data data/mc_configs/phi4_16x16_m2=-0.5_lam=0.5/mc_data.pt \
    --config configs/paper/phase1_train_16x16.yaml
python scripts/run_size_transfer.py

# Assemble LaTeX table bodies + all figures, then build the PDF
python scripts/make_paper_tables.py
python paper/generate_figures.py
make -C paper
```

GPU acceleration: every training script takes `--device cuda`;
`notebooks/07_ws1_experiments_colab.ipynb` runs the full suite on Colab.
Exact analysis-environment versions: `paper/requirements-paper.txt`.

### Phase IIa (complete 2026-07)

A-1–A-5 done: the A/B gauge-null finding is measured, Variant C
(gauge-invariant inputs) is ratified as the winning graph variant, and the
study is written up as a standalone paper draft — see
`docs/phase2_decisions.md` for the decision record.

U(1) gauge ensembles + labels (heatbath validated against torus-exact
character expansion):

```bash
python scripts/generate_u1_data.py     # 15 ensembles -> data/u1_configs/*.h5
python scripts/label_u1_data.py        # action, Wilson loops, Q + area-law checks
```

## Key References

- Bachtis, Aarts & Lucini (2021) — Closest prior work (different bipartition)
- Favoni et al. (2022) — L-GCN lattice gauge architecture
- Kanwar et al. (2020) — Equivariant normalizing flows for lattice QFT
- Boyda et al. (2021) — SU(N) gauge-equivariant sampling

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed overview of the heterogeneous graph structure and model architecture.
