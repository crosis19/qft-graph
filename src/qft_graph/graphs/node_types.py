"""Node type registry for heterogeneous graph construction."""

from __future__ import annotations

from enum import Enum


class NodeType(str, Enum):
    """Registered node types in the bipartite heterogeneous graph.

    The fundamental innovation: spacetime geometry and field content
    are represented as distinct node types, enabling the GNN to learn
    the geometry-matter coupling that is fundamental to QFT.
    """

    SPACETIME = "spacetime"
    SCALAR = "scalar"
    GAUGE = "gauge"  # Phase 2+
    FERMION = "fermion"  # Phase 2+
