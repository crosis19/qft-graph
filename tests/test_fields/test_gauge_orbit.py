"""Tests for gauge_orbit (task A-5): seeded gauge copies in float64.

The A-5 eps_gauge protocol rests on two properties pinned here:
(1) copies are generated in float64 from (possibly float32-stored) links,
    so gauge-invariant quantities agree to ~1e-12 across the orbit; and
(2) the orbit is reproducible from a seed, so every model evaluated on
    the same ensemble sees identical copies (paired comparison).
"""

import numpy as np
import pytest

from qft_graph.fields.gauge import (
    gauge_orbit,
    plaquette_angles,
    topological_charge,
    wilson_loop,
)

L = 6
K = 4


@pytest.fixture
def theta32():
    """Float32 links, as stored in the A-1/A-2 HDF5 files."""
    rng = np.random.default_rng(21)
    return rng.uniform(-np.pi, np.pi, size=(2, L, L)).astype(np.float32)


class TestGaugeOrbit:
    def test_shape_dtype_range(self, theta32):
        orbit = gauge_orbit(theta32, K, np.random.default_rng(0))
        assert orbit.shape == (K, 2, L, L)
        assert orbit.dtype == np.float64
        assert (orbit > -np.pi).all() and (orbit <= np.pi).all()

    def test_seed_reproducible(self, theta32):
        a = gauge_orbit(theta32, K, np.random.default_rng([5, 3]))
        b = gauge_orbit(theta32, K, np.random.default_rng([5, 3]))
        assert np.array_equal(a, b)
        c = gauge_orbit(theta32, K, np.random.default_rng([5, 4]))
        assert not np.array_equal(a, c)

    def test_copies_differ(self, theta32):
        """Every copy differs from the original and from the other copies
        (gauge transforms act freely away from constant alpha)."""
        orbit = gauge_orbit(theta32, K, np.random.default_rng(1))
        theta64 = theta32.astype(np.float64)
        for i in range(K):
            assert not np.allclose(orbit[i], theta64, atol=1e-3)
            for j in range(i + 1, K):
                assert not np.allclose(orbit[i], orbit[j], atol=1e-3)

    def test_float64_promotion_is_exact(self, theta32):
        """Promoting float32 storage before transforming == transforming
        the float64 view: bitwise-identical orbits (the A-3 convention)."""
        a = gauge_orbit(theta32, K, np.random.default_rng(2))
        b = gauge_orbit(theta32.astype(np.float64), K, np.random.default_rng(2))
        assert np.array_equal(a, b)

    def test_invariants_preserved(self, theta32):
        """Plaquette trig, Q, and Wilson loops agree across the orbit to
        float64 transform noise (~1e-12) — the labels-are-reusable fact
        the augmentation wrapper relies on."""
        theta64 = theta32.astype(np.float64)
        cos_p = np.cos(plaquette_angles(theta64))
        q = topological_charge(theta64)
        w22 = wilson_loop(theta64, 2, 2)
        for copy in gauge_orbit(theta32, K, np.random.default_rng(3)):
            np.testing.assert_allclose(np.cos(plaquette_angles(copy)), cos_p, atol=1e-12)
            assert topological_charge(copy) == pytest.approx(q, abs=1e-10)
            assert wilson_loop(copy, 2, 2) == pytest.approx(w22, abs=1e-12)
