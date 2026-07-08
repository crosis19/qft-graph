"""Generate publication-quality figures for the paper.

Run from the project root:
    python paper/generate_figures.py

Generates:
    paper/figures/graph_structure.pdf      - Bipartite graph schematic (TikZ in LaTeX)
    paper/figures/free_field.pdf           - Free field Gaussian validation
    paper/figures/energy_prediction.pdf    - Predicted vs true S_E scatter plot
    paper/figures/finite_size_scaling.pdf  - Three-panel FSS (|M|, χ, ξ/L)
    paper/figures/scaling_collapse.pdf     - ξ/L crossing + data collapse
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PDF output
import matplotlib.pyplot as plt
from matplotlib import rc

# Use LaTeX-like fonts
rc('font', family='serif', size=10)
rc('text', usetex=False)  # Set True if LaTeX is available
rc('axes', labelsize=11)
rc('xtick', labelsize=9)
rc('ytick', labelsize=9)
rc('legend', fontsize=8)

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
FIGURES_DIR = PROJECT_ROOT / 'paper' / 'figures'
DATA_DIR = PROJECT_ROOT / 'data' / 'mc_configs'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

from qft_graph.config import LatticeConfig, ScalarFieldConfig, MCConfig
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.actions.phi4 import Phi4Action
from qft_graph.mc.metropolis import MetropolisSampler, create_sampler
from qft_graph.mc.observables import ObservableSet
from qft_graph.mc.analysis import jackknife_mean_error


def fig_graph_structure():
    """Figure 1: 3D heterogeneous bipartite graph schematic."""
    print("Generating: graph_structure.pdf")
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    L = 4  # 4x4 lattice
    # Node positions on an L x L grid
    coords = np.array([(x, y) for y in range(L) for x in range(L)])
    n_nodes = len(coords)

    # 3D positions: spacetime at z=0, field nodes at z=-2
    st_xyz = np.column_stack([coords, np.zeros(n_nodes)])
    f_xyz = np.column_stack([coords, np.full(n_nodes, -2.0)])

    # Adjacency edges (nearest-neighbor, skip periodic wrap-around)
    adj_edges = []
    for i, (x1, y1) in enumerate(coords):
        for j, (x2, y2) in enumerate(coords):
            if i < j and abs(x1 - x2) + abs(y1 - y2) == 1:
                adj_edges.append((i, j))

    fig = plt.figure(figsize=(4.5, 4.0))
    ax = fig.add_subplot(111, projection='3d')

    # Semi-transparent planes for layer depth cue
    pad = 0.5
    verts_st = [
        [(-pad, -pad, 0), (L - 1 + pad, -pad, 0),
         (L - 1 + pad, L - 1 + pad, 0), (-pad, L - 1 + pad, 0)]
    ]
    verts_f = [
        [(-pad, -pad, -2), (L - 1 + pad, -pad, -2),
         (L - 1 + pad, L - 1 + pad, -2), (-pad, L - 1 + pad, -2)]
    ]
    ax.add_collection3d(Poly3DCollection(
        verts_st, alpha=0.06, facecolor='#4488ff', edgecolor='#4488ff', linewidths=0.5))
    ax.add_collection3d(Poly3DCollection(
        verts_f, alpha=0.06, facecolor='#ee6644', edgecolor='#ee6644', linewidths=0.5))

    # Adjacency edges (solid blue)
    for i, j in adj_edges:
        ax.plot(*zip(st_xyz[i], st_xyz[j]), color='#4488ff', linewidth=1.0, alpha=0.6)

    # Inhabits edges (dashed orange)
    for i in range(n_nodes):
        ax.plot(*zip(f_xyz[i], st_xyz[i]),
                color='#ee6644', linewidth=0.8, alpha=0.5, linestyle='--')

    # Spacetime nodes (blue)
    ax.scatter(*st_xyz.T, s=30, c='#4488ff', edgecolors='white',
               linewidths=0.5, zorder=5, label='Spacetime nodes')

    # Field nodes (orange-red)
    ax.scatter(*f_xyz.T, s=30, c='#ee6644', edgecolors='white',
               linewidths=0.5, zorder=5, label='Scalar field nodes')

    # Layer labels
    ax.text2D(0.82, 0.62, 'Spacetime\n($x_1, x_2$)',
              fontsize=7, color='#2266cc', ha='left', transform=ax.transAxes)
    ax.text2D(0.82, 0.28, 'Scalar Field\n($\\phi_i$)',
              fontsize=7, color='#cc4422', ha='left', transform=ax.transAxes)

    # Camera angle and axis formatting
    ax.view_init(elev=25, azim=-55)
    ax.set_xlim(-0.5, L - 1 + 0.5)
    ax.set_ylim(-0.5, L - 1 + 0.5)
    ax.set_zlim(-2.8, 0.8)
    ax.set_xlabel('$x_1$', fontsize=9, labelpad=2)
    ax.set_ylabel('$x_2$', fontsize=9, labelpad=2)
    ax.set_zlabel('Layer', fontsize=9, labelpad=2)
    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.set_zticks([0, -2])
    ax.set_zticklabels(['ST', 'Field'], fontsize=7)
    ax.tick_params(axis='both', labelsize=7, pad=1)

    # Clean up panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.grid(True, alpha=0.2)

    ax.legend(loc='upper left', fontsize=7, frameon=False,
              bbox_to_anchor=(0.0, 0.95))

    fig.savefig(FIGURES_DIR / 'graph_structure.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Done.")


def fig_free_field():
    """Figure 2: Free field Gaussian validation."""
    print("Generating: free_field.pdf")
    lattice = HypercubicLattice(LatticeConfig(dimensions=(16, 16)))
    action = Phi4Action(lattice, ScalarFieldConfig(mass_squared=1.0, coupling=0.0))
    sampler = MetropolisSampler(action, MCConfig(
        n_configs=2000, n_thermalization=500, n_sweeps_between=10, step_size=0.7, seed=42))
    result = sampler.generate(2000)

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    all_vals = result.configurations.flatten().numpy()
    ax.hist(all_vals, bins=80, density=True, alpha=0.7, color='#2196F3', edgecolor='none',
            label='MC samples')

    # Overlay approximate Gaussian
    x = np.linspace(-3, 3, 200)
    sigma = np.std(all_vals)
    ax.plot(x, np.exp(-x**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi)),
            'k--', linewidth=1.0, label=rf'Gaussian ($\sigma={sigma:.2f}$)')

    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel('Density')
    ax.set_title(r'Free field ($\lambda=0$, $m^2=1$)', fontsize=10)
    ax.legend(frameon=False)
    ax.set_xlim(-3, 3)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'free_field.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Done.")


def fig_energy_prediction():
    """Figure: Energy prediction scatter using saved model checkpoint."""
    print("Generating: energy_prediction.pdf")

    from qft_graph.config import ModelConfig
    from qft_graph.fields.scalar import ScalarField
    from qft_graph.graphs.builder import HeteroGraphBuilder
    from qft_graph.models.hetero_gnn import HeteroGNN

    # Try both 64x64 and 16x16
    for dims_str, dims in [('64x64', (64, 64)), ('16x16', (16, 16))]:
        data_path = DATA_DIR / f'phi4_{dims_str}_m2=-0.5_lam=0.5' / 'mc_data.pt'
        ckpt_path = PROJECT_ROOT / 'experiments' / 'runs' / 'colab_run' / 'model_final.pt'

        if not data_path.exists() or not ckpt_path.exists():
            continue

        print(f"  Using {dims_str} data with saved model...")
        mc_data = torch.load(data_path, weights_only=False)
        configurations = mc_data['configurations']
        actions = mc_data['actions']

        # Build graph dataset for validation split
        L = dims[0]
        lattice = HypercubicLattice(LatticeConfig(dimensions=dims))
        scalar_field = ScalarField()
        # Note: existing checkpoint was trained with a_in_edges=False
        builder = HeteroGraphBuilder(lattice, [scalar_field], a_in_edges=False)

        n_total = len(configurations)
        n_train = int(0.8 * n_total)
        val_configs = configurations[n_train:]
        val_actions = actions[n_train:]

        val_dataset = builder.build_dataset(
            configurations={'scalar': val_configs},
            actions=val_actions,
        )

        # Load model (checkpoint trained with a_in_edges=False)
        model_config = ModelConfig(a_in_edges=False)
        model = HeteroGNN(model_config, lattice_dim=2,
                          field_types={'scalar': 1}, lattice_spacing=1.0)

        from qft_graph.utils.checkpointing import remap_legacy_state_dict

        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(remap_legacy_state_dict(checkpoint['model_state_dict']))
        else:
            model.load_state_dict(remap_legacy_state_dict(checkpoint))
        model.eval()

        # Run predictions
        all_pred, all_true = [], []
        with torch.no_grad():
            for graph in val_dataset:
                output = model(graph)
                all_pred.append(output['energy'].cpu().reshape(1))
                all_true.append(graph.y.cpu().reshape(1))

        pred = torch.cat(all_pred)
        true = torch.cat(all_true)
        corr = torch.corrcoef(torch.stack([pred, true]))[0, 1].item()

        fig, ax = plt.subplots(figsize=(3.4, 3.4))
        ax.scatter(true.numpy(), pred.numpy(), alpha=0.3, s=3, color='#4488ff',
                   rasterized=True)
        lims = [min(true.min(), pred.min()).item() - 2,
                max(true.max(), pred.max()).item() + 2]
        ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
        ax.set_xlabel(r'True $S_E[\phi]$')
        ax.set_ylabel(r'Predicted $S_E[\phi]$')
        ax.set_title(rf'Energy Prediction ${L}\times{L}$ ($r = {corr:.4f}$)',
                      fontsize=10)
        ax.legend(frameon=False, loc='upper left')
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / 'energy_prediction.pdf', bbox_inches='tight',
                    dpi=300)
        plt.close()
        print(f"  Done. Pearson r = {corr:.4f}")
        return

    print("  SKIPPED: No MC data + model checkpoint found.")


def load_sweep_data():
    """Load FSS sweep results, preferring the v2 per-L files with errors.

    Returns a dict L_str -> {m2_values, mags(+_err), chis(+_err),
    xi_over_L(+_err)} or None. Falls back to the legacy colab_run JSON
    (errors set to zero) so old data still plots.
    """
    import json

    for base in (
        PROJECT_ROOT / 'results' / 'phase1_v2',
        PROJECT_ROOT / 'data' / 'sweep_results_v2',
    ):
        files = {L: base / f'sweep_{L}x{L}_lam=0.5.json' for L in (16, 32, 64)}
        if all(p.exists() for p in files.values()):
            out = {}
            for L, p in files.items():
                with open(p) as f:
                    pts = sorted(json.load(f), key=lambda q: q['m2'])
                out[str(L)] = {
                    'm2_values': [q['m2'] for q in pts],
                    'mags': [q['magnetization'] for q in pts],
                    'mags_err': [q['magnetization_err'] for q in pts],
                    'chis': [q['susceptibility'] for q in pts],
                    'chis_err': [q['susceptibility_err'] for q in pts],
                    'xi_over_L': [q['xi_over_L'] for q in pts],
                    'xi_over_L_err': [q['xi_over_L_err'] for q in pts],
                }
            print(f"  Using v2 sweep data from {base}")
            return out

    sweep_path = PROJECT_ROOT / 'experiments' / 'runs' / 'colab_run' / 'sweep_results.json'
    if sweep_path.exists():
        with open(sweep_path) as f:
            data = json.load(f)
        for d in data.values():
            n = len(d['m2_values'])
            for key in ('mags_err', 'chis_err', 'xi_over_L_err'):
                d.setdefault(key, [0.0] * n)
        print(f"  Using legacy sweep data from {sweep_path} (no error bars)")
        return data
    return None


def fig_finite_size_scaling():
    """Figure: Three-panel finite-size scaling from saved sweep results."""
    print("Generating: finite_size_scaling.pdf")

    sweep_data = load_sweep_data()
    if sweep_data is None:
        print("  SKIPPED: no sweep results found.")
        return

    colors = {'16': '#4488ff', '32': '#44bb88', '64': '#cc44ff'}
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))

    for L_str in ['16', '32', '64']:
        data = sweep_data[L_str]
        m2 = np.array(data['m2_values'])
        c = colors[L_str]

        axes[0].errorbar(m2, data['mags'], yerr=data['mags_err'], fmt='o-',
                         color=c, label=f'$L={L_str}$', markersize=3,
                         linewidth=1, capsize=1.5, elinewidth=0.7)
        axes[1].errorbar(m2, data['chis'], yerr=data['chis_err'], fmt='s-',
                         color=c, label=f'$L={L_str}$', markersize=3,
                         linewidth=1, capsize=1.5, elinewidth=0.7)
        axes[2].errorbar(m2, data['xi_over_L'], yerr=data['xi_over_L_err'],
                         fmt='^-', color=c, label=f'$L={L_str}$', markersize=3,
                         linewidth=1, capsize=1.5, elinewidth=0.7)

    axes[0].set_xlabel(r'$m^2$')
    axes[0].set_ylabel(r'$|\langle\phi\rangle|$')
    axes[0].set_title('Order Parameter', fontsize=9)
    axes[0].legend(frameon=False)

    axes[1].set_xlabel(r'$m^2$')
    axes[1].set_ylabel(r'$\chi$')
    axes[1].set_title('Susceptibility', fontsize=9)
    axes[1].legend(frameon=False)

    axes[2].set_xlabel(r'$m^2$')
    axes[2].set_ylabel(r'$\xi / L$')
    axes[2].set_title(r'$\xi/L$ Crossing', fontsize=9)
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.tick_params(direction='in')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'finite_size_scaling.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Done.")


def fig_scaling_collapse():
    """Figure: ξ/L crossing + data collapse from saved sweep results."""
    print("Generating: scaling_collapse.pdf")

    sweep_data = load_sweep_data()
    if sweep_data is None:
        print("  SKIPPED: no sweep results found.")
        return

    m2_values = np.array(sweep_data['16']['m2_values'])
    xi16 = np.array(sweep_data['16']['xi_over_L'])
    xi32 = np.array(sweep_data['32']['xi_over_L'])

    # Find crossing of L=16 and L=32
    diff = xi16 - xi32
    m2c = -2.45  # default
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0:
            f = diff[i] / (diff[i] - diff[i+1])
            m2c = m2_values[i] + f * (m2_values[i+1] - m2_values[i])
            break

    # Fit nu from collapse (grid search)
    best_nu = 1.0
    best_cost = float('inf')
    for nu_try in np.linspace(0.5, 2.0, 150):
        cost = 0
        for L1_str, L2_str in [('16', '32'), ('32', '64')]:
            L1, L2 = int(L1_str), int(L2_str)
            x1 = (m2_values - m2c) * L1**(1/nu_try)
            x2 = (m2_values - m2c) * L2**(1/nu_try)
            y1 = np.array(sweep_data[L1_str]['xi_over_L'])
            y2 = np.array(sweep_data[L2_str]['xi_over_L'])
            y2_interp = np.interp(x1, x2, y2, left=np.nan, right=np.nan)
            mask = np.isfinite(y2_interp)
            if mask.sum() > 3:
                cost += np.nanmean((y1[mask] - y2_interp[mask])**2)
        if cost < best_cost:
            best_cost = cost
            best_nu = nu_try

    colors = {'16': '#4488ff', '32': '#44bb88', '64': '#cc44ff'}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Left: ξ/L crossing
    for L_str in ['16', '32', '64']:
        data = sweep_data[L_str]
        c = colors[L_str]
        axes[0].errorbar(m2_values, data['xi_over_L'], yerr=data['xi_over_L_err'],
                         fmt='o-', color=c, label=f'$L={L_str}$', markersize=3,
                         linewidth=1, capsize=1.5, elinewidth=0.7)
    axes[0].axvline(m2c, color='gray', ls='--', alpha=0.5,
                    label=rf'$m^2_c = {m2c:.2f}$')
    axes[0].set_xlabel(r'$m^2$')
    axes[0].set_ylabel(r'$\xi / L$')
    axes[0].set_title(r'$\xi/L$ Crossing', fontsize=10)
    axes[0].legend(frameon=False, fontsize=7)

    # Right: Scaling collapse
    for L_str in ['16', '32', '64']:
        L = int(L_str)
        c = colors[L_str]
        x_scaled = (m2_values - m2c) * L**(1/best_nu)
        axes[1].scatter(x_scaled, sweep_data[L_str]['xi_over_L'],
                        s=15, color=c, alpha=0.7,
                        label=f'$L={L_str}$', edgecolors='none')
    axes[1].set_xlabel(rf'$(m^2 - m^2_c) \cdot L^{{1/\nu}}$, $\nu={best_nu:.2f}$')
    axes[1].set_ylabel(r'$\xi / L$')
    axes[1].set_title(rf'Scaling Collapse: $\nu = {best_nu:.2f}$', fontsize=10)
    axes[1].legend(frameon=False, fontsize=7)

    for ax in axes:
        ax.tick_params(direction='in')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'scaling_collapse.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Done. m2_c = {m2c:.3f}, nu = {best_nu:.2f}")


def fig_baseline_comparison():
    """Figure: Bar chart comparing baseline architectures."""
    print("Generating: baseline_comparison.pdf")

    import json
    results_path = PROJECT_ROOT / 'experiments' / 'baseline_results.json'
    if not results_path.exists():
        print("  SKIPPED: baseline_results.json not found. Run train_baselines.py first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    models = [r['model'] for r in results]
    pearson_r = [r['pearson_r'] for r in results]
    params = [r['n_params'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    colors = ['#cc4444', '#4488ff', '#44bb88', '#cc44ff']

    # Left: Pearson r
    bars = axes[0].bar(models, pearson_r, color=colors[:len(models)], alpha=0.8)
    axes[0].set_ylabel('Pearson $r$')
    axes[0].set_title('Energy Prediction Accuracy', fontsize=10)
    axes[0].set_ylim(min(pearson_r) - 0.01, 1.001)

    # Right: Parameter count
    axes[1].bar(models, [p/1000 for p in params], color=colors[:len(models)], alpha=0.8)
    axes[1].set_ylabel('Parameters (thousands)')
    axes[1].set_title('Model Size', fontsize=10)

    for ax in axes:
        ax.tick_params(direction='in')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=15, ha='right')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'baseline_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Done.")


def fig_generalization():
    """Figure: Generalization across coupling values."""
    print("Generating: generalization.pdf")

    import json
    results_path = PROJECT_ROOT / 'experiments' / 'generalization_results.json'
    if not results_path.exists():
        print("  SKIPPED: generalization_results.json not found. Run evaluate_generalization.py first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    m2_vals = [r['m2'] for r in results]
    pearson_r = [r['pearson_r'] for r in results]
    is_train = [r.get('is_training_point', False) for r in results]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    # Plot all points
    colors = ['#cc4444' if t else '#4488ff' for t in is_train]
    ax.scatter(m2_vals, pearson_r, c=colors, s=40, zorder=5)
    ax.plot(m2_vals, pearson_r, '--', color='#888888', alpha=0.5, linewidth=1)

    # Mark training point
    for m2, r, t in zip(m2_vals, pearson_r, is_train):
        if t:
            ax.annotate('training\npoint', (m2, r), textcoords='offset points',
                       xytext=(15, -10), fontsize=7, color='#cc4444',
                       arrowprops=dict(arrowstyle='->', color='#cc4444', lw=0.8))

    ax.set_xlabel(r'$m^2$')
    ax.set_ylabel('Pearson $r$')
    ax.set_title('Coupling Generalization', fontsize=10)
    ax.set_ylim(min(pearson_r) - 0.05, 1.01)
    ax.tick_params(direction='in')
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'generalization.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Done.")


if __name__ == '__main__':
    print("=" * 60)
    print("Generating paper figures")
    print("=" * 60)

    fig_graph_structure()
    fig_free_field()
    fig_energy_prediction()

    # These are expensive — skip with --quick flag
    if '--quick' not in sys.argv:
        fig_finite_size_scaling()
        fig_scaling_collapse()
    else:
        print("Skipping FSS figures (--quick mode)")

    # Baseline comparison figures (need results from experiments)
    fig_baseline_comparison()
    fig_generalization()

    print("\nAll figures saved to:", FIGURES_DIR)
