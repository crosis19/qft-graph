"""HDF5 -> HeteroData datasets for the U(1) ensembles (tasks A-2 -> A-5).

Reads the A-1/A-2 file schema (theta[N, 2, L, L] float32; action, q,
wilson/<RxT> label datasets; beta/L/seed/... attrs) and builds graph
datasets for any of the three A-3 variants.

Targets attached per graph (decision record #3: Wilson targets are RAW
per-config W, uniformly across all (beta, size) cells; -ln<W> is
ensemble-level validation only):
- y:        (1,)          total action
- y_wilson: (1, n_loops)  per-config W in wilson_loops order
- y_q:      (1,)          topological charge

PyG batching stacks these to (B,), (B, n_loops), (B,).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from qft_graph.config import LatticeConfig
from qft_graph.fields.gauge import random_gauge_transform
from qft_graph.graphs.builder import U1GaugeGraphBuilder
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.models.u1_gnn import DEFAULT_WILSON_LOOPS


def load_u1_ensemble(h5_path: str | Path) -> dict:
    """Load one (beta, L) ensemble file into memory.

    Returns:
        Dict with keys: theta (N, 2, L, L) float32 tensor, action (N,),
        q (N,), wilson {name: (N,)}, beta, L, and the remaining file attrs.
    """
    with h5py.File(h5_path, "r") as f:
        out = {
            "theta": torch.from_numpy(f["theta"][:]),
            "action": torch.from_numpy(f["action"][:]).float(),
            "q": torch.from_numpy(f["q"][:]).float(),
            "wilson": {
                name: torch.from_numpy(f[f"wilson/{name}"][:]).float()
                for name in f["wilson"]
            },
            "beta": float(f.attrs["beta"]),
            "L": int(f.attrs["L"]),
            "attrs": {k: f.attrs[k] for k in f.attrs},
        }
    return out


def u1_label_stats(h5_path: str | Path) -> dict:
    """Per-ensemble label means/stds for target standardization + run logs.

    One file is one (beta, L) cell, so per-file Wilson stats are exactly
    the per-(beta, size) statistics the z-scoring decision calls for.
    """
    ens = load_u1_ensemble(h5_path)
    stats = {
        "beta": ens["beta"],
        "L": ens["L"],
        "action": (ens["action"].mean().item(), ens["action"].std().item()),
        "q": (ens["q"].mean().item(), ens["q"].std().item()),
        "wilson": {
            name: (w.mean().item(), w.std().item())
            for name, w in ens["wilson"].items()
        },
    }
    return stats


def build_u1_dataset(
    h5_path: str | Path,
    variant: str,
    wilson_loops: tuple[str, ...] = DEFAULT_WILSON_LOOPS,
    n_configs: int | None = None,
) -> list:
    """Build a list of labeled HeteroData graphs from one ensemble file.

    Args:
        h5_path: Path to a data/u1_configs/*.h5 file.
        variant: Graph variant ("link_nodes"/"edge_features"/
            "invariant_oracle" or aliases "A"/"B"/"C").
        wilson_loops: Loop names to attach as y_wilson columns, in order.
        n_configs: Optional cap; uses the first n (configs are stored
            already decorrelated at the A-1 separation).

    Returns:
        List of HeteroData with y, y_wilson, y_q attached.
    """
    ens = load_u1_ensemble(h5_path)
    L = ens["L"]
    n = ens["theta"].shape[0] if n_configs is None else min(n_configs, ens["theta"].shape[0])

    missing = [name for name in wilson_loops if name not in ens["wilson"]]
    if missing:
        raise ValueError(
            f"Wilson loops {missing} not in {h5_path} "
            f"(available: {sorted(ens['wilson'])})"
        )

    lattice = HypercubicLattice(
        LatticeConfig(dimensions=(L, L), spacing=1.0, boundary="periodic")
    )
    builder = U1GaugeGraphBuilder(lattice, beta=ens["beta"], variant=variant)

    wilson_cols = torch.stack(
        [ens["wilson"][name][:n] for name in wilson_loops], dim=-1
    )  # (n, n_loops)

    dataset = []
    for i in range(n):
        data = builder.build({"gauge": ens["theta"][i]})
        data.y = ens["action"][i].unsqueeze(0)
        data.y_wilson = wilson_cols[i].unsqueeze(0)  # (1, n_loops)
        data.y_q = ens["q"][i].unsqueeze(0)
        # Column-order provenance: consumers (standardizer, loss code)
        # validate against this instead of trusting a repeated tuple.
        data.wilson_names = list(wilson_loops)
        # Position in the ensemble file — lets consumers that need the raw
        # theta back (gauge augmentation, task A-5) align by index instead
        # of assuming a split scheme.
        data.config_idx = i
        dataset.append(data)
    return dataset


class GaugeAugmentedU1Dataset(torch.utils.data.Dataset):
    """Train-split wrapper: a fresh random gauge copy per access (task A-5).

    On-the-fly augmentation for the A-5 experiment: every __getitem__
    draws a new random gauge transform of the underlying configuration
    (in float64, from the float32 storage — the A-3/A-5 numerics
    convention) and rebuilds the graph. Targets are copied from the
    wrapped base graph: action, Wilson loops, and Q are exactly
    gauge-invariant, so the already-standardized labels transfer
    unchanged.

    With augment=False the base graphs pass through untouched, so both
    arms of the augmentation experiment share one DataLoader code path.

    Reproducibility: transforms are drawn sequentially from one generator
    seeded at construction; together with the seeded DataLoader shuffle
    (set_seed in scripts/train_u1.py) the augmentation stream is
    deterministic per run.

    Args:
        base: Graphs from build_u1_dataset (standardized or not), e.g.
            the train split.
        thetas: (n_file_configs, 2, L, L) raw link angles for the WHOLE
            ensemble file; each base graph picks its own row via the
            config_idx stamped by build_u1_dataset.
        builder: U1GaugeGraphBuilder matching the base graphs' variant.
        seed: Seed for the augmentation stream.
        augment: False = pass base graphs through (control arm).
    """

    def __init__(
        self,
        base: list,
        thetas: torch.Tensor | np.ndarray,
        builder: U1GaugeGraphBuilder,
        seed: int,
        augment: bool = True,
    ) -> None:
        idx = []
        for pos, d in enumerate(base):
            ci = getattr(d, "config_idx", None)
            if ci is None:
                raise ValueError(
                    f"base graph at position {pos} lacks config_idx provenance; "
                    "rebuild with the current build_u1_dataset"
                )
            idx.append(int(ci))
        thetas64 = np.asarray(thetas, dtype=np.float64)
        if thetas64.ndim != 4 or (idx and max(idx) >= thetas64.shape[0]):
            raise ValueError(
                f"thetas shape {thetas64.shape} cannot cover config_idx up to "
                f"{max(idx) if idx else 'n/a'}"
            )
        self._base = base
        self._thetas = thetas64[idx]  # aligned copy, one row per base graph
        self._builder = builder
        self._augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, i: int):
        base = self._base[i]
        if not self._augment:
            return base
        theta_g = random_gauge_transform(self._thetas[i], self._rng)
        data = self._builder.build({"gauge": torch.from_numpy(theta_g)})
        data.y = base.y
        data.y_wilson = base.y_wilson
        data.y_q = base.y_q
        data.wilson_names = base.wilson_names
        data.config_idx = base.config_idx
        return data


def gauge_augmented_train_split(
    h5_path: str | Path,
    variant: str,
    train: list,
    seed: int,
    augment: bool = True,
) -> GaugeAugmentedU1Dataset:
    """Wrap an (already standardized) train split with gauge augmentation.

    Reloads the raw link angles and (beta, L) from the ensemble file the
    split was built from, so callers cannot desynchronize thetas from
    graphs — alignment runs through each graph's config_idx.
    """
    ens = load_u1_ensemble(h5_path)
    lattice = HypercubicLattice(
        LatticeConfig(dimensions=(ens["L"], ens["L"]), spacing=1.0, boundary="periodic")
    )
    builder = U1GaugeGraphBuilder(lattice, beta=ens["beta"], variant=variant)
    return GaugeAugmentedU1Dataset(train, ens["theta"], builder, seed=seed, augment=augment)


def standardize_scalar_targets(
    dataset: list, attr: str, stats: tuple[float, float] | None = None
) -> tuple[float, float]:
    """Z-score a (1,)-shaped scalar target (y or y_q) in place; returns stats.

    Protocol v2 (decision record #5): ALL targets are standardized with
    train-split stats for loss-scale balance — Q's natural integer scale
    grows with volume and destabilized the joint loss at L=32 and at
    beta=4 with deep stacks (Q-head collapse dragging the shared trunk).
    Invert via the returned stats before scale-dependent metrics
    (q_rounded_accuracy, relative_error); Pearson r is unaffected.
    """
    y = torch.cat([getattr(d, attr) for d in dataset])
    if stats is None:
        if y.shape[0] < 2:
            raise ValueError(
                f"Cannot standardize {attr} from {y.shape[0]} config(s); "
                "need >=2, or pass train-split stats"
            )
        stats = (y.mean().item(), y.std().item())
    mean, std = stats
    if not (np.isfinite(std) and std > 0):
        raise ValueError(f"Degenerate {attr} std: {std}")
    for d in dataset:
        setattr(d, attr, (getattr(d, attr) - mean) / std)
    return stats


def standardize_wilson_targets(
    dataset: list, stats: dict[str, tuple[float, float]] | None = None,
) -> dict[str, tuple[float, float]]:
    """Z-score y_wilson in place per loop size; returns the stats used.

    Decision record #3 allows per-(beta, size) standardization for
    training stability. Apply per ensemble (a dataset built from one file
    is one (beta, L) cell); pass `stats` to reuse train-split statistics
    on a val/test split. The transform is invertible from the returned
    stats and leaves Pearson r unchanged.

    Column order is taken from the wilson_names the dataset builder
    stamped on each graph — no independently repeated loop tuple to
    fall out of sync.
    """
    wilson_loops = tuple(dataset[0].wilson_names)
    for d in dataset:
        if tuple(d.wilson_names) != wilson_loops:
            raise ValueError(
                f"Inconsistent wilson_names in dataset: "
                f"{d.wilson_names} vs {list(wilson_loops)}"
            )
    y = torch.cat([d.y_wilson for d in dataset], dim=0)  # (n, n_loops)
    if stats is None and y.shape[0] < 2:
        raise ValueError(
            f"Degenerate Wilson std: cannot standardize {y.shape[0]} config(s); "
            "need >=2, or pass train-split stats"
        )
    if stats is None:
        stats = {
            name: (y[:, i].mean().item(), y[:, i].std().item())
            for i, name in enumerate(wilson_loops)
        }
    elif set(stats) != set(wilson_loops):
        raise ValueError(
            f"stats keys {sorted(stats)} do not match dataset loops "
            f"{sorted(wilson_loops)}"
        )
    mean = torch.tensor([stats[name][0] for name in wilson_loops])
    std = torch.tensor([stats[name][1] for name in wilson_loops])
    if not (torch.isfinite(std).all() and (std > 0).all()):
        # n=1 datasets give std=NaN (correction=1); constant columns give 0.
        # Either way the z-score would silently write NaN targets.
        raise ValueError(
            f"Degenerate Wilson std per loop: "
            f"{dict(zip(wilson_loops, std.tolist()))} — need >=2 configs "
            f"and non-constant columns to standardize"
        )
    for d in dataset:
        d.y_wilson = (d.y_wilson - mean) / std
    return stats
