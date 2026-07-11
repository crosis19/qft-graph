"""Tests for the three U(1) gauge graph variants (task A-3).

Variant A ("link_nodes"): links are gauge nodes attached to both endpoints.
Variant B ("edge_features"): links ride on st-st adjacency edge features.
Variant C ("invariant_oracle"): plaquette trig on spacetime nodes.

The A-3 oracle: a gauge transform of the input configs leaves Variant C
inputs bit-identical. This holds because the builder computes trig in
float64 (transform noise ~1e-15) and emits float32 features (resolution
~6e-8); the test asserts exact equality, not a tolerance.
"""

import numpy as np
import pytest
import torch

from qft_graph.fields.gauge import (
    plaquette_angles,
    random_gauge_transform,
    wrap_angle,
)
from qft_graph.graphs.builder import U1GaugeGraphBuilder
from qft_graph.graphs.edge_types import (
    ADJACENT,
    GAUGE_ENDS_AT,
    GAUGE_STARTS_AT,
    ORIGIN_OF,
    TARGET_OF,
)
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.config import LatticeConfig

BETA = 2.0
L = 4
N_SITES = L * L
N_LINKS = 2 * N_SITES
N_ADJ = 4 * N_SITES


@pytest.fixture
def theta():
    """Random float64 link angles on the 4x4 lattice."""
    rng = np.random.default_rng(7)
    return rng.uniform(-np.pi, np.pi, size=(2, L, L))


@pytest.fixture
def theta_t(theta):
    return torch.from_numpy(theta)


def make_builder(small_lattice, variant):
    return U1GaugeGraphBuilder(small_lattice, beta=BETA, variant=variant)


def make_lattice(dims):
    return HypercubicLattice(
        LatticeConfig(dimensions=dims, spacing=1.0, boundary="periodic")
    )


ALL_VARIANTS = ["link_nodes", "edge_features", "invariant_oracle"]
# (4, 6) kills axis-transposition mutants that a square lattice cannot see
DIMS = [(4, 4), (4, 6)]


class TestVariantA:
    def test_shapes_and_types(self, small_lattice, theta_t):
        data = make_builder(small_lattice, "link_nodes").build({"gauge": theta_t})
        assert set(data.node_types) == {"spacetime", "gauge"}
        assert data["spacetime"].x.shape == (N_SITES, 1)
        assert torch.all(data["spacetime"].x == 1.0)  # coordinate-free (decision 1)
        assert data["gauge"].x.shape == (N_LINKS, 5)
        assert data["gauge"].x.dtype == torch.float32
        assert data[ADJACENT].edge_index.shape == (2, N_ADJ)
        assert data[ADJACENT].edge_attr.shape == (N_ADJ, 2)
        for etype in (GAUGE_STARTS_AT, GAUGE_ENDS_AT):
            assert data[etype].edge_index.shape == (2, N_LINKS)
        for etype in (ORIGIN_OF, TARGET_OF):
            assert data[etype].edge_index.shape == (2, N_LINKS)
        data.validate(raise_on_error=True)

    def test_feature_roundtrip(self, small_lattice, theta, theta_t):
        """[cos, sin] recovers the wrapped link angle; onehot and beta correct."""
        data = make_builder(small_lattice, "link_nodes").build({"gauge": theta_t})
        x = data["gauge"].x.double()

        rec = torch.atan2(x[:, 1], x[:, 0]).numpy().reshape(2, L, L)
        np.testing.assert_allclose(rec, wrap_angle(theta), atol=1e-6)

        onehot = x[:, 2:4]
        assert torch.all(onehot[:N_SITES] == torch.tensor([1.0, 0.0], dtype=x.dtype))
        assert torch.all(onehot[N_SITES:] == torch.tensor([0.0, 1.0], dtype=x.dtype))
        assert torch.all(x[:, 4] == BETA)

    def test_link_endpoints(self, small_lattice, theta_t):
        """Each gauge node runs from x to x + e_mu with mu given by its onehot."""
        data = make_builder(small_lattice, "link_nodes").build({"gauge": theta_t})
        coords = small_lattice.site_coordinates()
        start = data[GAUGE_STARTS_AT].edge_index
        end = data[GAUGE_ENDS_AT].edge_index

        # Source side enumerates gauge nodes in order for both edge types
        assert torch.all(start[0] == torch.arange(N_LINKS))
        assert torch.all(end[0] == torch.arange(N_LINKS))

        delta = (coords[end[1]] - coords[start[1]]) % L
        mu = data["gauge"].x[:, 2:4].argmax(dim=-1)
        expected = torch.eye(2)[mu]
        assert torch.all(delta == expected)

    def test_reverse_edges_flip(self, small_lattice, theta_t):
        data = make_builder(small_lattice, "link_nodes").build({"gauge": theta_t})
        assert torch.all(
            data[ORIGIN_OF].edge_index == data[GAUGE_STARTS_AT].edge_index.flip(0)
        )
        assert torch.all(
            data[TARGET_OF].edge_index == data[GAUGE_ENDS_AT].edge_index.flip(0)
        )


class TestVariantB:
    def test_shapes_and_types(self, small_lattice, theta_t):
        data = make_builder(small_lattice, "edge_features").build({"gauge": theta_t})
        assert set(data.node_types) == {"spacetime"}
        assert data["spacetime"].x.shape == (N_SITES, 2)
        assert torch.all(data["spacetime"].x[:, 0] == 1.0)
        assert torch.all(data["spacetime"].x[:, 1] == BETA)
        assert data[ADJACENT].edge_attr.shape == (N_ADJ, 4)
        data.validate(raise_on_error=True)

    def test_forward_edges_carry_links(self, small_lattice, theta, theta_t):
        """+mu adjacency edge x -> x+e_mu carries [cos, sin] of theta_mu(x)."""
        data = make_builder(small_lattice, "edge_features").build({"gauge": theta_t})
        attr = data[ADJACENT].edge_attr.numpy()
        # Block layout: (mu=0,+),(mu=0,-),(mu=1,+),(mu=1,-), flat site order
        for mu, block in ((0, 0), (1, 2)):
            sl = slice(block * N_SITES, (block + 1) * N_SITES)
            np.testing.assert_allclose(
                attr[sl, 2], np.cos(theta[mu]).ravel(), atol=1e-7
            )
            np.testing.assert_allclose(
                attr[sl, 3], np.sin(theta[mu]).ravel(), atol=1e-7
            )

    def test_reversed_edges_carry_inverse_link(self, small_lattice, theta_t):
        """For every directed edge (i,j), edge (j,i) has the same cos and
        exactly negated sin (the inverse link U^-1)."""
        data = make_builder(small_lattice, "edge_features").build({"gauge": theta_t})
        ei = data[ADJACENT].edge_index
        attr = data[ADJACENT].edge_attr
        feat = {
            (int(ei[0, k]), int(ei[1, k])): (attr[k, 2].item(), attr[k, 3].item())
            for k in range(ei.shape[1])
        }
        assert len(feat) == N_ADJ  # no duplicate directed pairs on L=4
        for (i, j), (c, s) in feat.items():
            c_rev, s_rev = feat[(j, i)]
            assert c_rev == c
            assert s_rev == -s

    def test_consistent_with_variant_a(self, small_lattice, theta_t):
        """B's forward edge features equal A's gauge node [cos, sin]."""
        data_a = make_builder(small_lattice, "link_nodes").build({"gauge": theta_t})
        data_b = make_builder(small_lattice, "edge_features").build({"gauge": theta_t})
        attr = data_b[ADJACENT].edge_attr
        # A's gauge nodes: mu-major flat order == B's +mu blocks 0 and 2
        fwd = torch.cat(
            [attr[:N_SITES, 2:4], attr[2 * N_SITES : 3 * N_SITES, 2:4]]
        )
        assert torch.equal(fwd, data_a["gauge"].x[:, 0:2])


class TestVariantC:
    def test_shapes_and_types(self, small_lattice, theta_t):
        data = make_builder(small_lattice, "invariant_oracle").build({"gauge": theta_t})
        assert set(data.node_types) == {"spacetime"}
        assert data["spacetime"].x.shape == (N_SITES, 3)
        assert data["spacetime"].x.dtype == torch.float32
        assert data[ADJACENT].edge_attr.shape == (N_ADJ, 2)
        data.validate(raise_on_error=True)

    def test_features_are_plaquette_trig(self, small_lattice, theta, theta_t):
        data = make_builder(small_lattice, "invariant_oracle").build({"gauge": theta_t})
        x = data["spacetime"].x.numpy()
        tp = plaquette_angles(theta).ravel()
        np.testing.assert_array_equal(x[:, 0], np.cos(tp).astype(np.float32))
        np.testing.assert_array_equal(x[:, 1], np.sin(tp).astype(np.float32))
        np.testing.assert_array_equal(x[:, 2], np.float32(BETA))

    def test_gauge_invariance_bit_identical(self, small_lattice):
        """The A-3 oracle: gauge transforms leave Variant C inputs bit-identical."""
        rng = np.random.default_rng(42)
        builder = make_builder(small_lattice, "invariant_oracle")
        for _ in range(3):
            theta = rng.uniform(-np.pi, np.pi, size=(2, L, L))
            ref = builder.build({"gauge": torch.from_numpy(theta)})["spacetime"].x
            for _ in range(25):
                theta_g = random_gauge_transform(theta, rng)
                x = builder.build({"gauge": torch.from_numpy(theta_g)})["spacetime"].x
                assert torch.equal(x, ref)
                # The float64 margin backing the bitwise claim
                d = np.abs(
                    np.cos(plaquette_angles(theta_g)) - np.cos(plaquette_angles(theta))
                ).max()
                assert d < 1e-12

    def test_gauge_invariance_from_float32_storage(self, small_lattice):
        """Same oracle when links come from float32 storage (the HDF5 path):
        the stored config is fed to build() as float32 (the builder promotes
        internally), gauge copies are transformed in float64. Feeding the
        float32 tensor directly pins the builder's float64 promotion —
        without it, this fails at ~5e-7."""
        rng = np.random.default_rng(43)
        builder = make_builder(small_lattice, "invariant_oracle")
        theta32 = rng.uniform(-np.pi, np.pi, size=(2, L, L)).astype(np.float32)
        ref = builder.build({"gauge": torch.from_numpy(theta32)})["spacetime"].x
        theta = theta32.astype(np.float64)
        for _ in range(25):
            theta_g = random_gauge_transform(theta, rng)
            x = builder.build({"gauge": torch.from_numpy(theta_g)})["spacetime"].x
            assert torch.equal(x, ref)


class TestVariantsABNotInvariant:
    """Sanity: A and B inputs must CHANGE under gauge transforms — learning
    the invariance is the whole experiment (task A-5)."""

    @pytest.mark.parametrize("variant", ["link_nodes", "edge_features"])
    def test_features_change(self, small_lattice, theta, variant):
        rng = np.random.default_rng(11)
        builder = make_builder(small_lattice, variant)
        theta_g = random_gauge_transform(theta, rng)
        data = builder.build({"gauge": torch.from_numpy(theta)})
        data_g = builder.build({"gauge": torch.from_numpy(theta_g)})
        if variant == "link_nodes":
            changed = not torch.equal(data["gauge"].x, data_g["gauge"].x)
        else:
            changed = not torch.equal(
                data[ADJACENT].edge_attr, data_g[ADJACENT].edge_attr
            )
        assert changed


class TestEdgeIndexAttrBinding:
    """Pin the (edge_index, edge_attr) pairing on ADJACENT edges.

    Position-based feature checks alone cannot see a global src/dst swap
    (mutation-verified blind spot): the reversal-pairing relation is
    antisymmetric either way. These tests key every attribute off the
    endpoints actually listed in edge_index, on square AND rectangular
    lattices, so any misalignment of block order, roll direction, axis
    transposition, or edge orientation fails here.
    """

    @pytest.mark.parametrize("dims", DIMS)
    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_displacement_matches_endpoints(self, dims, variant):
        rng = np.random.default_rng(5)
        theta = rng.uniform(-np.pi, np.pi, size=(2, *dims))
        lat = make_lattice(dims)
        data = U1GaugeGraphBuilder(lat, beta=BETA, variant=variant).build(
            {"gauge": torch.from_numpy(theta)}
        )
        ei = data[ADJACENT].edge_index.numpy()
        disp = data[ADJACENT].edge_attr.numpy()[:, :2].round().astype(int)
        coords = lat.site_coordinates().numpy().astype(int)  # spacing 1.0

        # Every edge is one unit step, and the displacement feature is the
        # dst-minus-src step (mod lattice extent) for the listed endpoints.
        assert (np.abs(disp).sum(axis=1) == 1).all()
        delta = coords[ei[1]] - coords[ei[0]]
        assert (np.mod(delta - disp, np.array(dims)) == 0).all()

    @pytest.mark.parametrize("dims", DIMS)
    def test_variant_b_link_bound_to_traversal(self, dims):
        """The [cos, sin] on each directed edge is the link it traverses:
        theta_mu(src) along +mu, the inverse of theta_mu(dst) along -mu."""
        rng = np.random.default_rng(5)
        theta = rng.uniform(-np.pi, np.pi, size=(2, *dims))
        lat = make_lattice(dims)
        data = U1GaugeGraphBuilder(lat, beta=BETA, variant="edge_features").build(
            {"gauge": torch.from_numpy(theta)}
        )
        ei = data[ADJACENT].edge_index.numpy()
        attr = data[ADJACENT].edge_attr.numpy()
        disp = attr[:, :2].round().astype(int)
        coords = lat.site_coordinates().numpy().astype(int)

        for k in range(ei.shape[1]):
            mu = int(np.abs(disp[k]).argmax())
            if disp[k, mu] == 1:
                base = tuple(coords[ei[0, k]])  # forward: link lives at src
                expected = (np.cos(theta[mu][base]), np.sin(theta[mu][base]))
            else:
                base = tuple(coords[ei[1, k]])  # backward: inverse link at dst
                expected = (np.cos(theta[mu][base]), -np.sin(theta[mu][base]))
            np.testing.assert_allclose(attr[k, 2:4], expected, atol=1e-7)

    @pytest.mark.parametrize("dims", DIMS)
    def test_variant_a_endpoints(self, dims):
        """Variant A endpoint binding on square and rectangular lattices."""
        rng = np.random.default_rng(5)
        theta = rng.uniform(-np.pi, np.pi, size=(2, *dims))
        lat = make_lattice(dims)
        data = U1GaugeGraphBuilder(lat, beta=BETA, variant="link_nodes").build(
            {"gauge": torch.from_numpy(theta)}
        )
        coords = lat.site_coordinates().numpy().astype(int)
        start = data[GAUGE_STARTS_AT].edge_index[1].numpy()
        end = data[GAUGE_ENDS_AT].edge_index[1].numpy()
        mu = data["gauge"].x[:, 2:4].argmax(dim=-1).numpy()
        delta = np.mod(coords[end] - coords[start], np.array(dims))
        assert (delta == np.eye(2, dtype=int)[mu]).all()
        # and the angle bound to each gauge node is theta_mu(start site)
        rec = np.arctan2(data["gauge"].x[:, 1].numpy(), data["gauge"].x[:, 0].numpy())
        expected = theta[mu, coords[start][:, 0], coords[start][:, 1]]
        np.testing.assert_allclose(rec, wrap_angle(expected), atol=1e-6)

    @pytest.mark.parametrize("dims", DIMS)
    def test_variant_c_plaquettes(self, dims):
        """Variant C features on square and rectangular lattices."""
        rng = np.random.default_rng(5)
        theta = rng.uniform(-np.pi, np.pi, size=(2, *dims))
        lat = make_lattice(dims)
        data = U1GaugeGraphBuilder(lat, beta=BETA, variant="invariant_oracle").build(
            {"gauge": torch.from_numpy(theta)}
        )
        tp = plaquette_angles(theta).ravel()
        x = data["spacetime"].x.numpy()
        np.testing.assert_array_equal(x[:, 0], np.cos(tp).astype(np.float32))
        np.testing.assert_array_equal(x[:, 1], np.sin(tp).astype(np.float32))


class TestFloat32Promotion:
    """build() promotes to float64 internally, so float32 inputs (the HDF5
    storage dtype) yield bitwise the same features as their float64 view.
    Deleting the promotion breaks this (and the Variant C oracle) at ~5e-7."""

    @pytest.mark.parametrize("variant", ALL_VARIANTS)
    def test_float32_input_equals_float64_view(self, small_lattice, variant):
        rng = np.random.default_rng(9)
        theta32 = rng.uniform(-np.pi, np.pi, size=(2, L, L)).astype(np.float32)
        builder = make_builder(small_lattice, variant)
        d32 = builder.build({"gauge": torch.from_numpy(theta32)})
        d64 = builder.build({"gauge": torch.from_numpy(theta32.astype(np.float64))})
        assert torch.equal(d32["spacetime"].x, d64["spacetime"].x)
        assert torch.equal(d32[ADJACENT].edge_attr, d64[ADJACENT].edge_attr)
        if variant == "link_nodes":
            assert torch.equal(d32["gauge"].x, d64["gauge"].x)


class TestBuilderApi:
    @pytest.mark.parametrize("alias,canonical", [("A", "link_nodes"), ("B", "edge_features"), ("C", "invariant_oracle")])
    def test_variant_aliases(self, small_lattice, alias, canonical):
        assert make_builder(small_lattice, alias).variant == canonical

    def test_unknown_variant_raises(self, small_lattice):
        with pytest.raises(ValueError, match="Unknown variant"):
            U1GaugeGraphBuilder(small_lattice, beta=BETA, variant="D")

    def test_non_2d_lattice_raises(self):
        lat3 = HypercubicLattice(
            LatticeConfig(dimensions=(4, 4, 4), spacing=1.0, boundary="periodic")
        )
        with pytest.raises(ValueError, match="2D"):
            U1GaugeGraphBuilder(lat3, beta=BETA)

    def test_non_periodic_lattice_raises(self):
        """Frozen convention (plan §3.1): the roll-based link/plaquette
        arithmetic assumes wraparound — open BCs must fail loudly."""
        lat_open = HypercubicLattice(
            LatticeConfig(dimensions=(4, 4), spacing=1.0, boundary="open")
        )
        with pytest.raises(ValueError, match="periodic"):
            U1GaugeGraphBuilder(lat_open, beta=BETA)

    def test_missing_config_raises(self, small_lattice):
        with pytest.raises(ValueError, match="Missing configuration"):
            make_builder(small_lattice, "link_nodes").build({})

    def test_wrong_shape_raises(self, small_lattice):
        bad = torch.zeros(2, L, L + 1)
        with pytest.raises(ValueError, match="shape"):
            make_builder(small_lattice, "link_nodes").build({"gauge": bad})

    @pytest.mark.parametrize("variant", ["link_nodes", "edge_features", "invariant_oracle"])
    def test_globals_carry_beta(self, small_lattice, theta_t, variant):
        data = make_builder(small_lattice, variant).build({"gauge": theta_t})
        assert torch.equal(data.globals, torch.tensor([[BETA]]))

    @pytest.mark.parametrize("variant", ["link_nodes", "edge_features", "invariant_oracle"])
    def test_build_dataset(self, small_lattice, variant):
        rng = np.random.default_rng(3)
        n = 5
        thetas = torch.from_numpy(rng.uniform(-np.pi, np.pi, size=(n, 2, L, L)))
        actions = torch.randn(n)
        dataset = make_builder(small_lattice, variant).build_dataset(
            {"gauge": thetas}, actions
        )
        assert len(dataset) == n
        assert dataset[0].y.shape == (1,)
        assert torch.equal(dataset[2].y, actions[2].unsqueeze(0))
