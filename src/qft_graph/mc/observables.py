"""Physical observable measurements from field configurations."""

from __future__ import annotations

import numpy as np
import torch

from qft_graph.lattice.hypercubic import HypercubicLattice


class ObservableSet:
    """Computes physical observables from scalar field configurations.

    All methods operate on single configurations. Ensemble averaging
    should be done externally over multiple configurations.
    """

    def __init__(self, lattice: HypercubicLattice) -> None:
        self.lattice = lattice
        self._nsites = lattice.num_sites()
        self._dims = lattice.shape

    def magnetization(self, phi: torch.Tensor) -> float:
        """Order parameter: <phi> = (1/V) sum_x phi(x).

        Args:
            phi: Configuration of shape (num_sites,).
        """
        return phi.mean().item()

    def abs_magnetization(self, phi: torch.Tensor) -> float:
        """Absolute magnetization |<phi>|."""
        return abs(phi.mean().item())

    def magnetization_squared(self, phi: torch.Tensor) -> float:
        """M^2 for a single configuration, M = (1/V) sum_x phi(x)."""
        m = phi.mean().item()
        return m * m

    def susceptibility_term(self, phi: torch.Tensor) -> float:
        """Site-averaged phi^2 = (1/V) sum_x phi(x)^2 for one configuration.

        NOTE: this is NOT an ingredient of the magnetic susceptibility
        (it differs from M^2 by the x != y correlations). For chi use
        ObservableSet.susceptibility, which implements the frozen
        convention chi = V * (<M^2> - <|M|>^2).
        """
        return (phi**2).mean().item()

    @staticmethod
    def susceptibility(
        phi_configs: torch.Tensor,
        convention: str = "abs",
    ) -> float:
        """Ensemble magnetic susceptibility chi.

        Frozen convention (implementation plan, task P1-2):

            chi = V * ( <M^2> - <|M|>^2 ),   M = (1/V) sum_x phi(x)

        The |M| subtraction removes the spontaneous-magnetization
        contribution in the broken phase, where the Var(M) form is
        contaminated by rare tunneling events.

        Args:
            phi_configs: Field configurations, shape (n_configs, n_sites).
            convention: "abs" (frozen, default) uses <|M|>^2;
                "var" uses <M>^2 (V*Var(M)) — sensitivity checks only.

        Returns:
            Susceptibility chi.
        """
        n_sites = phi_configs.shape[1]
        M = phi_configs.mean(dim=1)
        if convention == "abs":
            sub = M.abs().mean().pow(2)
        elif convention == "var":
            sub = M.mean().pow(2)
        else:
            raise ValueError(f"Unknown chi convention: {convention!r}")
        return float(n_sites * (M.pow(2).mean() - sub))

    def two_point_function(
        self, phi: torch.Tensor, connected: bool = True
    ) -> torch.Tensor:
        """Two-point correlation function G(r).

        Computes G(r) = <phi(x) phi(x+r)> averaged over x and all
        equivalent displacement vectors at distance r.

        If connected=True (default), subtracts the disconnected piece:
            G_c(r) = <phi(x) phi(x+r)> - <phi>^2

        For a 2D lattice of size L, returns correlator for r = 0, 1, ..., L//2.

        Args:
            phi: Configuration of shape (num_sites,).
            connected: If True, subtract <phi>^2 (disconnected part).

        Returns:
            Tensor of shape (L//2 + 1,).
        """
        L = self._dims[0]  # Assume square lattice
        phi_grid = phi.reshape(self._dims)
        mean_phi = phi.mean()

        max_r = L // 2 + 1
        G_r = torch.zeros(max_r)
        counts = torch.zeros(max_r)

        for dx in range(max_r):
            for dim in range(self.lattice.dimension()):
                # Shift along dimension `dim` by `dx`
                shifted = torch.roll(phi_grid, shifts=-dx, dims=dim)
                corr = (phi_grid * shifted).mean()
                if connected:
                    corr = corr - mean_phi**2
                G_r[dx] += corr
                counts[dx] += 1

        G_r /= counts.clamp(min=1)
        return G_r

    def two_point_function_full(self, phi: torch.Tensor) -> torch.Tensor:
        """Full (disconnected) two-point function.

        G(r) = <phi(x) phi(x+r)> without subtracting <phi>^2.
        Useful when computing connected correlator from ensemble averages.

        Args:
            phi: Configuration of shape (num_sites,).

        Returns:
            Tensor of shape (L//2 + 1,).
        """
        return self.two_point_function(phi, connected=False)

    def energy_density(self, action_value: float) -> float:
        """Energy density e = S_E / V.

        Args:
            action_value: Total Euclidean action S_E.
        """
        return action_value / self.lattice.volume()

    @staticmethod
    def correlation_length(G_r: torch.Tensor, lattice_spacing: float = 1.0) -> float:
        """Extract correlation length xi from exponential fit to G(r).

        Fits G(r) ~ A * exp(-r/xi) for r > 0 where G(r) > 0.

        Args:
            G_r: Two-point function from two_point_function().
            lattice_spacing: Physical lattice spacing.

        Returns:
            Correlation length xi in lattice units.
        """
        # Use log-linear fit: log(G(r)) = log(A) - r/xi
        r_vals = []
        log_G = []
        for r in range(1, len(G_r)):
            if G_r[r].item() > 0:
                r_vals.append(r * lattice_spacing)
                log_G.append(torch.log(G_r[r]).item())

        if len(r_vals) < 2:
            return 0.0

        # Simple linear regression
        r_arr = torch.tensor(r_vals)
        log_G_arr = torch.tensor(log_G)
        r_mean = r_arr.mean()
        log_mean = log_G_arr.mean()
        slope = ((r_arr - r_mean) * (log_G_arr - log_mean)).sum() / (
            (r_arr - r_mean) ** 2
        ).sum()

        if slope.item() >= 0:
            return float("inf")  # No decay = infinite correlation length

        xi = -1.0 / slope.item()
        return xi

    @staticmethod
    def correlation_length_second_moment(
        G_r: torch.Tensor,
        L: int,
        lattice_spacing: float = 1.0,
    ) -> float:
        """Extract correlation length using the second-moment method (1D).

        Uses the 1D Fourier transform of the angle-averaged correlator.
        For small lattices this is adequate, but for precise finite-size
        scaling use correlation_length_fft() which uses the full 2D FFT.

        Args:
            G_r: Two-point function from two_point_function(), shape (L//2+1,).
            L: Linear lattice size.
            lattice_spacing: Physical lattice spacing.

        Returns:
            Correlation length xi in lattice units.
        """
        max_r = len(G_r)  # L//2 + 1
        k_min = 2.0 * np.pi / L

        # Reconstruct full 1D DFT using symmetry G(r) = G(L-r)
        G_tilde_0 = G_r[0].item()
        for r in range(1, max_r - 1):
            G_tilde_0 += 2.0 * G_r[r].item()
        if max_r > 1:
            G_tilde_0 += G_r[max_r - 1].item()

        G_tilde_k = G_r[0].item()
        for r in range(1, max_r - 1):
            G_tilde_k += 2.0 * G_r[r].item() * np.cos(k_min * r)
        if max_r > 1:
            G_tilde_k += G_r[max_r - 1].item() * np.cos(k_min * (max_r - 1))

        if G_tilde_k <= 0 or G_tilde_0 <= 0:
            return 0.0

        ratio = G_tilde_0 / G_tilde_k - 1.0
        if ratio <= 0:
            return 0.0

        xi = lattice_spacing / (2.0 * np.sin(np.pi / L)) * np.sqrt(ratio)
        return float(xi) if np.isfinite(xi) else 0.0

    @staticmethod
    def correlation_length_fft(
        phi_configs: torch.Tensor,
        L: int,
        lattice_spacing: float = 1.0,
        chi_convention: str = "abs",
    ) -> float:
        """Extract correlation length using the full 2D FFT method.

        Uses the standard second-moment estimator in momentum space:

            ξ = (1 / 2sin(π/L)) × sqrt( G̃(0)/G̃(k_min) - 1 )

        with the frozen convention (plan task P1-2):

            G̃(0)     = χ = V · (<M²> - <|M|>²)
            G̃(k_min) = <|φ̃(k_min, 0)|²> / V     (propagator at k_min)

        χ is computed from magnetization fluctuations (not from the k=0
        FFT mode), while G̃(k_min) comes from the k≠0 FFT modes which are
        unaffected by any mean subtraction.

        Args:
            phi_configs: Field configurations, shape (n_configs, n_sites).
            L: Linear lattice size (assumes square L×L lattice).
            lattice_spacing: Physical lattice spacing.
            chi_convention: "abs" (frozen, default) or "var" — passed to
                ObservableSet.susceptibility; "var" for sensitivity checks.

        Returns:
            Correlation length xi in lattice units.
        """
        n_configs = phi_configs.shape[0]
        V = L * L

        # Reshape to 2D grids
        phi_grids = phi_configs.reshape(n_configs, L, L)

        # --- Susceptibility from magnetization fluctuations ---
        M = phi_grids.mean(dim=(1, 2))  # (n_configs,)
        chi = ObservableSet.susceptibility(
            phi_configs.reshape(n_configs, -1), convention=chi_convention
        )

        # --- Propagator at k_min from FFT ---
        # Subtract per-config spatial mean so k=0 mode is zero.
        # This doesn't affect k≠0 modes.
        per_config_mean = M.reshape(n_configs, 1, 1)
        phi_centered = phi_grids - per_config_mean

        # 2D FFT, then ensemble-average the power spectrum at k_min
        phi_fft = torch.fft.fft2(phi_centered)  # (n_configs, L, L)
        # F = <|φ̃(k_min, 0)|²> / V  where k_min index = 1
        F = (phi_fft[:, 1, 0].real**2 + phi_fft[:, 1, 0].imag**2).mean().item() / V

        if F <= 0 or chi <= 0:
            return 0.0

        ratio = chi / F - 1.0
        if ratio <= 0:
            return 0.0

        xi = lattice_spacing / (2.0 * np.sin(np.pi / L)) * np.sqrt(ratio)
        return float(xi) if np.isfinite(xi) else 0.0

    @staticmethod
    def correlation_length_fft_jackknife(
        phi_configs: torch.Tensor,
        L: int,
        n_blocks: int = 20,
        lattice_spacing: float = 1.0,
        chi_convention: str = "abs",
    ) -> tuple[float, float]:
        """Extract ξ with binned jackknife error estimate.

        Splits configurations into n_blocks jackknife blocks, computes ξ
        from each leave-one-block-out subset, and returns the mean and
        standard error. Choose n_blocks so the block size exceeds
        2·tau_int of the configuration series.

        Args:
            phi_configs: Field configurations, shape (n_configs, n_sites).
            L: Linear lattice size.
            n_blocks: Number of jackknife blocks.
            lattice_spacing: Physical lattice spacing.
            chi_convention: "abs" (frozen) or "var", forwarded to
                correlation_length_fft.

        Returns:
            (xi_mean, xi_err) tuple.
        """
        n_configs = phi_configs.shape[0]
        block_size = n_configs // n_blocks
        # Trim to exact multiple of block_size
        n_use = block_size * n_blocks
        configs = phi_configs[:n_use]

        xi_jack = []
        for b in range(n_blocks):
            # Leave out block b
            mask = torch.ones(n_use, dtype=torch.bool)
            mask[b * block_size : (b + 1) * block_size] = False
            subset = configs[mask]
            xi_b = ObservableSet.correlation_length_fft(
                subset, L, lattice_spacing, chi_convention=chi_convention
            )
            xi_jack.append(xi_b)

        xi_arr = np.array(xi_jack)
        xi_mean = xi_arr.mean()
        # Jackknife error formula: sqrt((n-1)/n * Σ(xi_b - xi_mean)²)
        xi_err = np.sqrt((n_blocks - 1) / n_blocks * np.sum((xi_arr - xi_mean)**2))

        return float(xi_mean), float(xi_err)
