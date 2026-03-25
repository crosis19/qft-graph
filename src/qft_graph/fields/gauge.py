"""Stub: Gauge link fields for Phase 2+ (U(1), SU(N))."""

# This module will be implemented in Phase 2 (U(1) gauge theory + fermions).
# The gauge field lives on lattice edges rather than sites, requiring an
# extension to the HeteroGraphBuilder for edge-hosted field nodes.
#
# Key features to implement:
#   - U(1) link variables U_mu(x) = exp(i a A_mu(x)) as complex phases
#   - SU(3) link variables as 3x3 unitary matrices (Phase 3)
#   - Plaquette (Wilson loop) computation as gauge-invariant features
#   - Gauge transformations for equivariance testing
