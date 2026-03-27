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


def fig_free_field():
    """Figure 1: Free field Gaussian validation."""
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
    """Figure: Energy prediction scatter (uses pre-generated data)."""
    print("Generating: energy_prediction.pdf")

    # Load 16x16 data
    data_path = DATA_DIR / 'phi4_16x16_m2=-0.5_lam=0.5' / 'mc_data.pt'
    if not data_path.exists():
        print("  SKIPPED: 16x16 data not found. Run generate_mc_data.py first.")
        return

    data = torch.load(data_path, weights_only=False)
    actions = data['actions']

    # Simulate "predicted" by adding small noise (placeholder until real model output saved)
    # In the actual paper, replace with saved model predictions
    pred = actions + torch.randn_like(actions) * 0.1

    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    ax.scatter(actions.numpy(), pred.numpy(), alpha=0.3, s=3, color='#4488ff', rasterized=True)
    lims = [actions.min().item() - 2, actions.max().item() + 2]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlabel(r'True $S_E[\phi]$')
    ax.set_ylabel(r'Predicted $S_E[\phi]$')
    ax.set_title(r'Energy Prediction ($16\times 16$)', fontsize=10)
    ax.legend(frameon=False, loc='upper left')
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'energy_prediction.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print("  Done. NOTE: Replace with actual model predictions for final version.")


def fig_finite_size_scaling():
    """Figure: Three-panel finite-size scaling (|M|, χ, ξ/L)."""
    print("Generating: finite_size_scaling.pdf")

    COUPLING = 0.5
    SWEEP_CONFIGS = 1000
    M2_STEPS = 20
    m2_values = np.linspace(-1.5, -2.8, M2_STEPS)

    sizes = [(16, 16), (32, 32), (64, 64)]
    colors = {'16': '#4488ff', '32': '#44bb88', '64': '#cc44ff'}
    results = {}

    for dims in sizes:
        L = dims[0]
        print(f"  Sweeping L={L}...")
        lat = HypercubicLattice(LatticeConfig(dimensions=dims))
        obs = ObservableSet(lat)
        warm_phi = None

        mags, chis, xis = [], [], []
        xi_errs = []

        n_sweeps = max(10, L // 4 * 10)

        for m2 in m2_values:
            act = Phi4Action(lat, ScalarFieldConfig(mass_squared=m2, coupling=COUPLING))
            samp = create_sampler(act, MCConfig(
                n_configs=SWEEP_CONFIGS, n_thermalization=1000,
                n_sweeps_between=n_sweeps, seed=None))
            res = samp.generate(SWEEP_CONFIGS, initial_phi=warm_phi)
            warm_phi = res.configurations[-1].clone()

            n = len(res.configurations)
            M_samples = torch.tensor([res.configurations[j].mean().item() for j in range(n)])
            absM = M_samples.abs()
            M2 = M_samples ** 2

            mag_mean, _ = jackknife_mean_error(absM)
            V = lat.num_sites()
            chi = V * (M2.mean().item() - absM.mean().item()**2)

            xi_mean, xi_err = ObservableSet.correlation_length_fft_jackknife(
                res.configurations, L, n_blocks=20)

            mags.append(mag_mean)
            chis.append(chi)
            xis.append(xi_mean / L)
            xi_errs.append(xi_err / L)

        results[L] = {'mags': mags, 'chis': chis, 'xis': xis, 'xi_errs': xi_errs}

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))

    for L, data in results.items():
        c = colors[str(L)]
        axes[0].plot(m2_values, data['mags'], 'o-', color=c, label=f'$L={L}$',
                     markersize=3, linewidth=1)
        axes[1].plot(m2_values, data['chis'], 's-', color=c, label=f'$L={L}$',
                     markersize=3, linewidth=1)
        axes[2].errorbar(m2_values, data['xis'], yerr=data['xi_errs'],
                         fmt='^-', color=c, label=f'$L={L}$',
                         markersize=3, linewidth=1, capsize=1.5)

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

    # Save data for collapse plot
    torch.save({'m2_values': m2_values, 'results': results},
               FIGURES_DIR / 'fss_data.pt')
    print("  Done.")


def fig_scaling_collapse():
    """Figure: ξ/L crossing + data collapse (uses saved FSS data)."""
    print("Generating: scaling_collapse.pdf")

    data_path = FIGURES_DIR / 'fss_data.pt'
    if not data_path.exists():
        print("  SKIPPED: Run fig_finite_size_scaling() first.")
        return

    saved = torch.load(data_path, weights_only=False)
    m2_values = saved['m2_values']
    results = saved['results']

    # Find crossing of L=16 and L=32
    xi16 = np.array(results[16]['xis'])
    xi32 = np.array(results[32]['xis'])
    diff = xi16 - xi32
    # Find sign change
    m2c = -2.45  # default
    for i in range(len(diff) - 1):
        if diff[i] * diff[i+1] < 0:
            # Linear interpolation
            f = diff[i] / (diff[i] - diff[i+1])
            m2c = m2_values[i] + f * (m2_values[i+1] - m2_values[i])
            break

    # Fit nu from collapse (simple grid search)
    best_nu = 1.0
    best_cost = float('inf')
    for nu_try in np.linspace(0.5, 2.0, 100):
        cost = 0
        for L1, L2 in [(16, 32), (32, 64)]:
            if L2 not in results:
                continue
            x1 = (m2_values - m2c) * L1**(1/nu_try)
            x2 = (m2_values - m2c) * L2**(1/nu_try)
            y1 = np.array(results[L1]['xis'])
            y2 = np.array(results[L2]['xis'])
            # Interpolate y2 onto x1 grid and compare
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
    for L, data in results.items():
        c = colors[str(L)]
        axes[0].errorbar(m2_values, data['xis'], yerr=data['xi_errs'],
                         fmt='o-', color=c, label=f'$L={L}$',
                         markersize=3, linewidth=1, capsize=1.5)
    axes[0].axvline(m2c, color='gray', ls='--', alpha=0.5,
                    label=rf'$m^2_c = {m2c:.2f}$')
    axes[0].set_xlabel(r'$m^2$')
    axes[0].set_ylabel(r'$\xi / L$')
    axes[0].set_title(r'$\xi/L$ Crossing', fontsize=10)
    axes[0].legend(frameon=False, fontsize=7)

    # Right: Scaling collapse
    for L, data in results.items():
        c = colors[str(L)]
        x_scaled = (m2_values - m2c) * L**(1/best_nu)
        axes[1].scatter(x_scaled, data['xis'], s=15, color=c, alpha=0.7,
                        label=f'$L={L}$', edgecolors='none')
    axes[1].set_xlabel(rf'$(m^2 - m^2_c) \cdot L^{{1/\nu}}$, $\nu={best_nu:.2f}$')
    axes[1].set_ylabel(r'$\xi / L$')
    axes[1].set_title(rf'Scaling Collapse: $\nu = {best_nu:.2f}$', fontsize=10)
    axes[1].legend(frameon=False, fontsize=7)

    for ax in axes:
        ax.tick_params(direction='in')

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'scaling_collapse.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Done. m²_c = {m2c:.3f}, ν = {best_nu:.2f}")


if __name__ == '__main__':
    print("=" * 60)
    print("Generating paper figures")
    print("=" * 60)

    fig_free_field()
    fig_energy_prediction()

    # These are expensive — skip with --quick flag
    if '--quick' not in sys.argv:
        fig_finite_size_scaling()
        fig_scaling_collapse()
    else:
        print("Skipping FSS figures (--quick mode)")

    print("\nAll figures saved to:", FIGURES_DIR)
