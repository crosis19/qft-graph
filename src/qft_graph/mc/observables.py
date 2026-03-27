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
        """<phi>^2 for susceptibility computation."""
        m = phi.mean().item()
        return m * m

    def susceptibility_term(self, phi: torch.Tensor) -> float:
        """V * <phi^2> contribution. Combine with <phi>^2 for chi = V*(<phi^2> - <phi>^2)."""
        return (phi**2).mean().item()

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
    ) -> float:
        """Extract correlation length using the full 2D FFT method.

        This is the correct implementation for finite-size scaling.
        It uses the power spectrum of the field configurations:

            G̃(k) = <|φ̃(k)|²> / V

        and then:
            ξ = (1 / 2sin(π/L)) × sqrt( G̃(0,0)/G̃(k_min,0) - 1 )

        where k_min = (2π/L, 0). This uses ALL spatial correlations
        (not just on-axis), giving correct results for all lattice sizes.

        The FFT approach is also faster than computing G(r) in real space.

        Args:
            phi_configs: Field configurations, shape (n_configs, n_sites).
            L: Linear lattice size (assumes square L×L lattice).
            lattice_spacing: Physical lattice spacing.

        Returns:
            Correlation length xi in lattice units.
        """
        n_configs = phi_configs.shape[0]

        # Reshape to 2D grid and compute ensemble-averaged power spectrum
        phi_grids = phi_configs.reshape(n_configs, L, L)

        # Subtract per-config mean to get connected correlator
        # (equivalent to setting k=0 mode to zero, but we need G̃(0) from
        #  the ensemble, not per-config, so we subtract ensemble mean)
        ensemble_mean = phi_grids.mean().item()
        phi_centered = phi_grids - ensemble_mean

        # 2D FFT of each config, then average |φ̃(k)|²
        phi_fft = torch.fft.fft2(phi_centered)  # (n_configs, L, L) complex
        power_spectrum = (phi_fft.real**2 + phi_fft.imag**2).mean(dim=0)  # (L, L)
        power_spectrum /= (L * L)  # Normalize: G̃(k) = <|φ̃(k)|²> / V

        # G̃(0,0): zero-momentum mode (susceptibility)
        G_tilde_0 = power_spectrum[0, 0].item()

        # G̃(k_min, 0): smallest non-zero momentum along x
        # k_min = 2π/L corresponds to FFT index 1
        G_tilde_k = power_spectrum[1, 0].item()

        if G_tilde_k <= 0 or G_tilde_0 <= 0:
            return 0.0

        ratio = G_tilde_0 / G_tilde_k - 1.0
        if ratio <= 0:
            return 0.0

        xi = lattice_spacing / (2.0 * np.sin(np.pi / L)) * np.sqrt(ratio)
        return float(xi) if np.isfinite(xi) else 0.0
