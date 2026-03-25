"""Shared test fixtures for qft_graph tests."""

from __future__ import annotations

import pytest
import torch

from qft_graph.config import LatticeConfig, MCConfig, ModelConfig, ScalarFieldConfig
from qft_graph.actions.phi4 import Phi4Action
from qft_graph.fields.scalar import ScalarField
from qft_graph.graphs.builder import HeteroGraphBuilder
from qft_graph.lattice.hypercubic import HypercubicLattice


@pytest.fixture
def small_lattice_config():
    """4x4 lattice for fast unit tests."""
    return LatticeConfig(dimensions=(4, 4), spacing=1.0, boundary="periodic")


@pytest.fixture
def medium_lattice_config():
    """8x8 lattice for integration tests."""
    return LatticeConfig(dimensions=(8, 8), spacing=1.0, boundary="periodic")


@pytest.fixture
def small_lattice(small_lattice_config):
    return HypercubicLattice(small_lattice_config)


@pytest.fixture
def medium_lattice(medium_lattice_config):
    return HypercubicLattice(medium_lattice_config)


@pytest.fixture
def scalar_field():
    return ScalarField()


@pytest.fixture
def field_config():
    return ScalarFieldConfig(mass_squared=-0.5, coupling=0.5)


@pytest.fixture
def free_field_config():
    """Free field: lambda=0, m^2=1 (Gaussian theory with exact results)."""
    return ScalarFieldConfig(mass_squared=1.0, coupling=0.0)


@pytest.fixture
def phi4_action(small_lattice, field_config):
    return Phi4Action(small_lattice, field_config)


@pytest.fixture
def graph_builder(small_lattice, scalar_field):
    return HeteroGraphBuilder(small_lattice, [scalar_field])


@pytest.fixture
def sample_config(small_lattice, scalar_field):
    """A random scalar field configuration on the small lattice."""
    torch.manual_seed(42)
    return scalar_field.initialize(small_lattice.num_sites(), mode="hot")


@pytest.fixture
def model_config():
    return ModelConfig(
        hidden_dim=16,
        n_mp_blocks=2,
        encoder_layers=1,
        dropout=0.0,
        activation="gelu",
        readout="energy",
    )


@pytest.fixture
def mc_config():
    return MCConfig(
        n_configs=10,
        n_thermalization=50,
        n_sweeps_between=2,
        step_size=1.0,
        seed=42,
    )
