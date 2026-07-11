"""Heterogeneous bipartite graph construction from lattice + field content."""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from qft_graph.fields.base import Field
from qft_graph.fields.gauge import U1GaugeField
from qft_graph.graphs.edge_types import (
    ADJACENT,
    GAUGE_ENDS_AT,
    GAUGE_STARTS_AT,
    ORIGIN_OF,
    TARGET_OF,
    inhabits_edge,
    inhabits_inv_edge,
)
from qft_graph.lattice.base import Lattice


class HeteroGraphBuilder:
    """Constructs heterogeneous bipartite graphs from lattice geometry + field content.

    The core architectural innovation: spacetime nodes and field nodes are
    distinct types connected by typed edges, forming a bipartite structure
    that mirrors the geometry-matter separation fundamental to QFT.

    Graph structure:
        G = (V_spacetime, V_field1, ..., V_fieldN,
             E_adjacent, E_inhabits_1, ..., E_inhabits_N)

    This builder is generic over field types:
    - Phase 1: ScalarField only (one phi per site)
    - Phase 2: + GaugeField (links on edges) + FermionField (spinors on sites)
    - Phase 3+: Additional field types added with zero changes to this builder

    Args:
        lattice: Lattice instance providing geometry.
        fields: List of Field instances to include in the graph.
        a_in_edges: If True (default, matching ModelConfig.a_in_edges),
            lattice spacing scales the adjacency edge displacement vectors
            and spacetime nodes carry coordinates only. If False, spacing
            is appended as a spacetime node feature instead.
        spacetime_features: "coords" (default) or "constant". The constant
            variant replaces coordinates with a single 1.0 feature so all
            geometry is carried by displacement edge features — restoring
            translation invariance and enabling size transfer (task P1-6).
    """

    def __init__(
        self,
        lattice: Lattice,
        fields: list[Field],
        a_in_edges: bool = True,
        spacetime_features: str = "coords",
    ) -> None:
        if spacetime_features not in ("coords", "constant"):
            raise ValueError(f"Unknown spacetime_features: {spacetime_features!r}")
        self.lattice = lattice
        self.fields = fields
        self._nsites = lattice.num_sites()
        self._a_in_edges = a_in_edges
        self._spacetime_features = spacetime_features

        # Precompute lattice topology (fixed across configurations)
        self._coords = lattice.site_coordinates()
        self._adj_src, self._adj_dst = lattice.neighbor_pairs()
        self._edge_dirs = lattice.edge_directions()

        if a_in_edges:
            self._edge_dirs = self._edge_dirs * lattice.lattice_spacing()

        # Bipartite edges: each field node i connects to spacetime node i
        self._bipartite_idx = torch.arange(self._nsites)

    def build(self, configurations: dict[str, torch.Tensor]) -> HeteroData:
        """Build a single heterogeneous graph from field configurations.

        Args:
            configurations: Mapping from field.node_type_name() to raw
                configuration tensor. E.g., {"scalar": tensor(num_sites,)}.

        Returns:
            PyG HeteroData with typed nodes, edges, and features.
        """
        data = HeteroData()

        # --- Spacetime nodes ---
        if self._spacetime_features == "constant":
            # Coordinate-free: geometry carried entirely by edge displacements
            st_features = torch.ones(self._nsites, 1)
        elif self._a_in_edges:
            # Lattice spacing encoded in edge displacements; nodes get coordinates only
            st_features = self._coords
        else:
            # Features: [coordinates, lattice_spacing]
            spacing = torch.full((self._nsites, 1), self.lattice.lattice_spacing())
            st_features = torch.cat([self._coords, spacing], dim=-1)
        data["spacetime"].x = st_features
        data["spacetime"].num_nodes = self._nsites

        # --- Spacetime adjacency edges ---
        data[ADJACENT].edge_index = torch.stack([self._adj_src, self._adj_dst])
        data[ADJACENT].edge_attr = self._edge_dirs

        # --- Field nodes + bipartite edges ---
        for field_obj in self.fields:
            fname = field_obj.node_type_name()

            if fname not in configurations:
                raise ValueError(
                    f"Missing configuration for field '{fname}'. "
                    f"Provided: {list(configurations.keys())}"
                )

            # Field node features
            features = field_obj.node_features(configurations[fname])
            data[fname].x = features
            data[fname].num_nodes = self._nsites

            # Bipartite edges: field -> spacetime (inhabits)
            inh_edge = inhabits_edge(fname)
            data[inh_edge].edge_index = torch.stack(
                [self._bipartite_idx, self._bipartite_idx]
            )

            # Reverse bipartite: spacetime -> field (for message passing back)
            inv_edge = inhabits_inv_edge(fname)
            data[inv_edge].edge_index = torch.stack(
                [self._bipartite_idx, self._bipartite_idx]
            )

        # --- Graph-level coupling constants (task P1-3) ---
        # Concatenated over fields; PyG batching stacks these to (batch, k)
        globals_list = [
            g for g in (f.global_features() for f in self.fields) if g is not None
        ]
        if globals_list:
            data.globals = torch.cat(globals_list, dim=-1)  # (1, k)

        return data

    def build_dataset(
        self, configurations: dict[str, torch.Tensor], actions: torch.Tensor | None = None
    ) -> list[HeteroData]:
        """Build a list of HeteroData graphs from batched configurations.

        Args:
            configurations: Mapping from field name to (n_configs, ...) tensors.
            actions: Optional (n_configs,) tensor of true action values.

        Returns:
            List of HeteroData objects, one per configuration.
        """
        # Determine batch size from first field
        first_key = next(iter(configurations))
        n_configs = configurations[first_key].shape[0]

        dataset = []
        for i in range(n_configs):
            single_config = {
                fname: configurations[fname][i] for fname in configurations
            }
            data = self.build(single_config)

            if actions is not None:
                data.y = actions[i].unsqueeze(0)

            dataset.append(data)

        return dataset


class U1GaugeGraphBuilder(HeteroGraphBuilder):
    """Three graph variants for quenched compact U(1) link configurations (task A-3).

    Variants (plan §3, task A-3); beta is appended as a scalar input feature
    in all three (Phase I's m² lesson), and all three use constant spacetime
    node features (Phase II decision record, 2026-07-09):

    - "link_nodes" (Variant A, primary): each of the 2*L² links becomes a
      'gauge' node with features [cos θ, sin θ, onehot(μ), β]. A link
      attaches to its two endpoint sites via (gauge, starts_at, spacetime)
      to x and (gauge, ends_at, spacetime) to x+e_μ, with reverses
      (spacetime, origin_of, gauge) and (spacetime, target_of, gauge).
      A plaquette is a length-4 link–site alternating cycle, so plaquette
      information is reachable within 2 message-passing blocks.
    - "edge_features" (Variant B): spacetime nodes only; [cos θ, sin θ] is
      appended to the st–st adjacency edge features alongside the
      displacement. The reversed edge carries the inverse link
      [cos θ, −sin θ]. β is appended to the spacetime node features
      (the only nodes in this variant).
    - "invariant_oracle" (Variant C, control): spacetime node features
      ← [cos θ_P(x), sin θ_P(x), β] — one plaquette per site in 2D. Gauge
      invariance is handed to the model; this is the ceiling that A/B's
      *learned* invariance is measured against (task A-5).

    All trigonometry is computed in float64 and cast to float32 features.
    For Variant C this is load-bearing: the float64 gauge-transform noise in
    cos/sin θ_P (~1e-15) is ~7 orders of magnitude below float32 resolution,
    so the emitted features are bit-identical across gauge copies of a
    config, as the A-3 oracle test demands. Computing from float32 links
    would break this (~5e-7 discrepancies).

    The raw configuration key is "gauge" with tensors of shape (2, L1, L2)
    (link angles θ[μ, x1, x2]; layout as in fields/gauge.py).

    Args:
        lattice: 2D periodic HypercubicLattice (rectangular allowed).
        beta: Wilson gauge coupling, appended as an input feature and
            exposed as the graph-level `globals` (P1-3 pattern).
        variant: "link_nodes", "edge_features", or "invariant_oracle"
            (aliases "A", "B", "C").
    """

    VARIANT_ALIASES = {
        "A": "link_nodes",
        "B": "edge_features",
        "C": "invariant_oracle",
        "link_nodes": "link_nodes",
        "edge_features": "edge_features",
        "invariant_oracle": "invariant_oracle",
    }

    def __init__(self, lattice: Lattice, beta: float, variant: str = "link_nodes") -> None:
        if variant not in self.VARIANT_ALIASES:
            raise ValueError(
                f"Unknown variant: {variant!r}. "
                f"Expected one of {sorted(set(self.VARIANT_ALIASES))}"
            )
        if lattice.dimension() != 2:
            raise ValueError("U1GaugeGraphBuilder requires a 2D lattice")
        boundary = getattr(lattice, "boundary", None)
        if boundary is None or boundary() != "periodic":
            # Frozen convention (plan §3.1): all link/plaquette index
            # arithmetic below assumes wraparound; fail loudly otherwise.
            raise ValueError(
                "U1GaugeGraphBuilder requires periodic boundary conditions, got "
                f"{boundary() if boundary is not None else 'a lattice without a boundary() accessor'}"
            )
        super().__init__(
            lattice,
            [U1GaugeField(lattice.shape)],
            spacetime_features="constant",
        )
        self.beta = float(beta)
        self.variant = self.VARIANT_ALIASES[variant]
        self._dims = lattice.shape

        # Link endpoint site indices, in gauge-node order (mu-major,
        # row-major sites: node = mu * L1*L2 + x1 * L2 + x2, matching
        # U1GaugeField.node_features). Endpoints follow the frozen
        # convention: link (mu, x) runs from x to x + e_mu.
        site = torch.arange(self._nsites).reshape(self._dims)
        self._link_start = torch.cat([site.reshape(-1), site.reshape(-1)])
        self._link_end = torch.cat(
            [torch.roll(site, -1, dims=mu).reshape(-1) for mu in range(2)]
        )
        self._link_idx = torch.arange(2 * self._nsites)

    def _check_theta(self, configurations: dict[str, torch.Tensor]) -> torch.Tensor:
        if "gauge" not in configurations:
            raise ValueError(
                f"Missing configuration for field 'gauge'. "
                f"Provided: {list(configurations.keys())}"
            )
        theta = configurations["gauge"]
        expected = (2, *self._dims)
        if tuple(theta.shape) != expected:
            raise ValueError(f"Expected theta of shape {expected}, got {tuple(theta.shape)}")
        # Trig in float64; features are cast to float32 on the way out
        # (bit-identical gauge invariance of Variant C depends on this).
        return theta.double()

    def build(self, configurations: dict[str, torch.Tensor]) -> HeteroData:
        theta = self._check_theta(configurations)

        data = HeteroData()
        beta_col = torch.full((self._nsites, 1), self.beta)

        # --- Spacetime adjacency (all variants) ---
        data[ADJACENT].edge_index = torch.stack([self._adj_src, self._adj_dst])

        if self.variant == "link_nodes":
            data["spacetime"].x = torch.ones(self._nsites, 1)
            data[ADJACENT].edge_attr = self._edge_dirs

            gauge_feat = self.fields[0].node_features(theta)  # (2*nsites, 4)
            beta_gauge = torch.full((gauge_feat.shape[0], 1), self.beta, dtype=theta.dtype)
            data["gauge"].x = torch.cat([gauge_feat, beta_gauge], dim=-1).float()
            data["gauge"].num_nodes = 2 * self._nsites

            data[GAUGE_STARTS_AT].edge_index = torch.stack([self._link_idx, self._link_start])
            data[GAUGE_ENDS_AT].edge_index = torch.stack([self._link_idx, self._link_end])
            data[ORIGIN_OF].edge_index = torch.stack([self._link_start, self._link_idx])
            data[TARGET_OF].edge_index = torch.stack([self._link_end, self._link_idx])

        elif self.variant == "edge_features":
            data["spacetime"].x = torch.cat([torch.ones(self._nsites, 1), beta_col], dim=-1)

            # Adjacency block order (mu=0,+),(mu=0,-),(mu=1,+),(mu=1,-), each
            # in flat site order. The +mu edge x -> x+e_mu carries theta_mu(x);
            # the -mu edge x -> x-e_mu traverses the link at x-e_mu against
            # its orientation, so it carries the inverse [cos, -sin].
            angles = torch.cat(
                [
                    theta[0].reshape(-1),
                    torch.roll(theta[0], 1, dims=0).reshape(-1),
                    theta[1].reshape(-1),
                    torch.roll(theta[1], 1, dims=1).reshape(-1),
                ]
            )
            sign = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=theta.dtype)
            sign = sign.repeat_interleave(self._nsites)
            link_feat = torch.stack([angles.cos(), sign * angles.sin()], dim=-1)
            data[ADJACENT].edge_attr = torch.cat(
                [self._edge_dirs, link_feat.float()], dim=-1
            )

        else:  # invariant_oracle
            data[ADJACENT].edge_attr = self._edge_dirs

            t1, t2 = theta[0], theta[1]
            theta_p = (
                t1
                + torch.roll(t2, -1, dims=0)  # theta_2(x + e1)
                - torch.roll(t1, -1, dims=1)  # theta_1(x + e2)
                - t2
            ).reshape(-1)
            data["spacetime"].x = torch.cat(
                [
                    torch.stack([theta_p.cos(), theta_p.sin()], dim=-1).float(),
                    beta_col,
                ],
                dim=-1,
            )

        data["spacetime"].num_nodes = self._nsites

        # Graph-level coupling constant (P1-3 pattern; PyG batches to (batch, 1))
        data.globals = torch.tensor([[self.beta]])

        return data
