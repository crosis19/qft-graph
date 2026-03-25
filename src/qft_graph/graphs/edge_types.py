"""Edge type registry for heterogeneous graph construction."""

from __future__ import annotations

# Edge relation triplets (src_type, relation, dst_type) used in PyG HeteroData.
# These define the bipartite structure separating geometry from field content.

# Spacetime-Spacetime: lattice adjacency (nearest neighbors)
ADJACENT = ("spacetime", "adjacent", "spacetime")

# Field-Spacetime: bipartite "inhabits" edge from field node to its host spacetime site
def inhabits_edge(field_type: str) -> tuple[str, str, str]:
    """Create inhabits edge type for a given field type."""
    return (field_type, "inhabits", "spacetime")

# Spacetime-Field: reverse bipartite edge for message passing back to fields
def inhabits_inv_edge(field_type: str) -> tuple[str, str, str]:
    """Create reverse inhabits edge type."""
    return ("spacetime", f"inhabits_inv_{field_type}", field_type)
