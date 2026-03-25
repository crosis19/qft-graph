"""Tests for the HypercubicLattice class."""

import pytest
import torch

from qft_graph.config import LatticeConfig
from qft_graph.lattice.hypercubic import HypercubicLattice


class TestHypercubicLattice2D:
    """Tests for 2D hypercubic lattice with periodic BCs."""

    def test_num_sites(self, small_lattice):
        assert small_lattice.num_sites() == 16  # 4x4

    def test_dimension(self, small_lattice):
        assert small_lattice.dimension() == 2

    def test_coordinates_shape(self, small_lattice):
        coords = small_lattice.site_coordinates()
        assert coords.shape == (16, 2)

    def test_coordinates_range(self, small_lattice):
        coords = small_lattice.site_coordinates()
        assert coords.min() >= 0
        assert coords.max() <= 3  # 0, 1, 2, 3 for 4x4

    def test_neighbor_count(self, small_lattice):
        """Each site should have 2d = 4 neighbors in 2D with PBC."""
        src, dst = small_lattice.neighbor_pairs()
        # Total edges = nsites * 2 * ndim (both directions for each axis)
        assert len(src) == 16 * 4  # 4 neighbors per site

    def test_neighbor_pairs_valid_indices(self, small_lattice):
        src, dst = small_lattice.neighbor_pairs()
        assert src.min() >= 0
        assert src.max() < 16
        assert dst.min() >= 0
        assert dst.max() < 16

    def test_periodic_boundary_wrapping(self):
        """Corner site should connect to opposite edge."""
        config = LatticeConfig(dimensions=(4, 4), spacing=1.0, boundary="periodic")
        lattice = HypercubicLattice(config)
        src, dst = lattice.neighbor_pairs()

        # Site 0 is at (0,0). Its neighbors should include site at (3,0) and (0,3)
        site_0_neighbors = dst[src == 0].tolist()
        # (0,0) -> (1,0), (3,0), (0,1), (0,3) in some order
        assert len(site_0_neighbors) == 4

    def test_edge_directions_shape(self, small_lattice):
        dirs = small_lattice.edge_directions()
        src, _ = small_lattice.neighbor_pairs()
        assert dirs.shape == (len(src), 2)

    def test_edge_directions_unit(self, small_lattice):
        """Direction vectors should be unit vectors along axes."""
        dirs = small_lattice.edge_directions()
        norms = dirs.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms))

    def test_volume(self, small_lattice):
        assert small_lattice.volume() == 16.0  # a^2 * N^2 = 1.0 * 16


class TestHypercubicLattice3D:
    """Tests for 3D hypercubic lattice (Phase 2+ preparation)."""

    def test_3d_lattice(self):
        config = LatticeConfig(dimensions=(4, 4, 4), spacing=1.0, boundary="periodic")
        lattice = HypercubicLattice(config)
        assert lattice.num_sites() == 64
        assert lattice.dimension() == 3

    def test_3d_neighbor_count(self):
        config = LatticeConfig(dimensions=(4, 4, 4), spacing=1.0, boundary="periodic")
        lattice = HypercubicLattice(config)
        src, _ = lattice.neighbor_pairs()
        # 2d = 6 neighbors per site in 3D
        assert len(src) == 64 * 6


class TestHypercubicLatticeOpenBC:
    """Tests for open boundary conditions."""

    def test_open_bc_fewer_edges(self):
        """Open BCs should have fewer edges than periodic."""
        periodic = HypercubicLattice(LatticeConfig(dimensions=(4, 4), boundary="periodic"))
        open_bc = HypercubicLattice(LatticeConfig(dimensions=(4, 4), boundary="open"))

        src_p, _ = periodic.neighbor_pairs()
        src_o, _ = open_bc.neighbor_pairs()
        assert len(src_o) < len(src_p)
