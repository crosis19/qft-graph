"""Tests for the HDF5 -> HeteroData pipeline (task A-4)."""

import h5py
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from torch_geometric.loader import DataLoader

from qft_graph.config import ModelConfig
from qft_graph.graphs.edge_types import ADJACENT
from qft_graph.graphs.u1_dataset import (
    build_u1_dataset,
    gauge_augmented_train_split,
    load_u1_ensemble,
    standardize_scalar_targets,
    standardize_wilson_targets,
    u1_label_stats,
)
from qft_graph.models.u1_gnn import U1GaugeGNN

L = 4
N = 8
BETA = 1.5
LOOPS = ("1x1", "2x2", "2x4", "3x3", "4x4")


@pytest.fixture
def h5file(tmp_path):
    """Synthetic ensemble file following the A-1/A-2 schema exactly."""
    rng = np.random.default_rng(11)
    path = tmp_path / f"u1_L{L}_beta{BETA}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "theta",
            data=rng.uniform(-np.pi, np.pi, size=(N, 2, L, L)).astype(np.float32),
        )
        f.create_dataset("action", data=rng.normal(10.0, 1.0, size=N))
        f.create_dataset("q", data=rng.integers(-2, 3, size=N).astype(np.float64))
        for name in LOOPS:
            f.create_dataset(f"wilson/{name}", data=rng.normal(0.3, 0.1, size=N))
        f.attrs["beta"] = BETA
        f.attrs["L"] = L
        f.attrs["seed"] = 123
        f.attrs["separation"] = 5
        f.attrs["thermalization"] = 1000
    return path


class TestLoad:
    def test_load_roundtrip(self, h5file):
        ens = load_u1_ensemble(h5file)
        assert ens["theta"].shape == (N, 2, L, L)
        assert ens["theta"].dtype == torch.float32
        assert ens["beta"] == BETA and ens["L"] == L
        assert set(ens["wilson"]) == set(LOOPS)

    def test_label_stats(self, h5file):
        stats = u1_label_stats(h5file)
        ens = load_u1_ensemble(h5file)
        assert stats["wilson"]["2x2"][0] == pytest.approx(
            ens["wilson"]["2x2"].mean().item()
        )
        assert stats["beta"] == BETA


class TestBuildDataset:
    @pytest.mark.parametrize("variant", ["link_nodes", "edge_features", "invariant_oracle"])
    def test_targets_attached(self, h5file, variant):
        dataset = build_u1_dataset(h5file, variant)
        ens = load_u1_ensemble(h5file)
        assert len(dataset) == N
        d = dataset[3]
        assert d.y.shape == (1,)
        assert d.y_wilson.shape == (1, len(LOOPS))
        assert d.y_q.shape == (1,)
        assert d.y.item() == pytest.approx(ens["action"][3].item())
        assert d.y_q.item() == pytest.approx(ens["q"][3].item())
        for i, name in enumerate(LOOPS):
            assert d.y_wilson[0, i].item() == pytest.approx(
                ens["wilson"][name][3].item()
            )

    def test_batching_shapes(self, h5file):
        dataset = build_u1_dataset(h5file, "link_nodes")
        batch = Batch.from_data_list(dataset[:4])
        assert batch.y.shape == (4,)
        assert batch.y_wilson.shape == (4, len(LOOPS))
        assert batch.y_q.shape == (4,)
        assert batch.globals.shape == (4, 1)

    def test_n_configs_cap(self, h5file):
        assert len(build_u1_dataset(h5file, "A", n_configs=3)) == 3

    def test_missing_loop_raises(self, h5file):
        with pytest.raises(ValueError, match="not in"):
            build_u1_dataset(h5file, "A", wilson_loops=("9x9",))

    def test_end_to_end_forward(self, h5file):
        """Pipeline smoke test: file -> graphs -> batch -> model heads."""
        dataset = build_u1_dataset(h5file, "link_nodes", n_configs=4)
        config = ModelConfig(hidden_dim=16, n_mp_blocks=2, encoder_layers=1)
        model = U1GaugeGNN(config, variant="link_nodes", wilson_loops=LOOPS).eval()
        batch = Batch.from_data_list(dataset)
        out = model(batch)
        assert out["energy"].shape == batch.y.shape
        assert out["q"].shape == batch.y_q.shape
        for i, name in enumerate(LOOPS):
            assert out[f"wilson_{name}"].shape == batch.y_wilson[:, i].shape


class TestStandardize:
    def test_zscore_in_place(self, h5file):
        dataset = build_u1_dataset(h5file, "C")
        stats = standardize_wilson_targets(dataset)
        y = torch.cat([d.y_wilson for d in dataset], dim=0)
        assert y.mean(dim=0).abs().max() < 1e-5
        assert (y.std(dim=0) - 1).abs().max() < 1e-5
        # invertible from the returned stats
        m, s = stats[LOOPS[0]]
        ens = load_u1_ensemble(h5file)
        restored = y[:, 0] * s + m
        torch.testing.assert_close(
            restored, ens["wilson"][LOOPS[0]], atol=1e-5, rtol=1e-5
        )

    def test_reuse_train_stats(self, h5file):
        train = build_u1_dataset(h5file, "C", n_configs=6)
        val = build_u1_dataset(h5file, "C")
        stats = standardize_wilson_targets(train)
        stats_val = standardize_wilson_targets(val, stats=stats)
        assert stats_val == stats  # applied, not recomputed
        # same raw value -> same standardized value in both splits
        torch.testing.assert_close(train[0].y_wilson, val[0].y_wilson)

    def test_wilson_names_stamped_and_column_order_safe(self, h5file):
        """Column order is provenance on the graphs; the standardizer takes
        it from there, so a reordered build cannot be silently mis-keyed."""
        reordered = tuple(reversed(LOOPS))
        dataset = build_u1_dataset(h5file, "C", wilson_loops=reordered)
        assert dataset[0].wilson_names == list(reordered)
        stats = standardize_wilson_targets(dataset)
        ens = load_u1_ensemble(h5file)
        # stats are keyed by the ACTUAL column names, not a default tuple
        assert stats[reordered[0]][0] == pytest.approx(
            ens["wilson"][reordered[0]].mean().item()
        )

    def test_mismatched_stats_keys_raise(self, h5file):
        dataset = build_u1_dataset(h5file, "C", wilson_loops=("1x1", "2x2"))
        bad_stats = {"1x1": (0.0, 1.0), "3x3": (0.0, 1.0)}
        with pytest.raises(ValueError, match="do not match"):
            standardize_wilson_targets(dataset, stats=bad_stats)

    def test_degenerate_std_raises(self, h5file):
        """n=1 gives std=NaN, constant columns give std=0 — both would
        silently write NaN targets without the guard."""
        dataset = build_u1_dataset(h5file, "C", n_configs=1)
        with pytest.raises(ValueError, match="Degenerate|need >=2"):
            standardize_wilson_targets(dataset)


class TestStandardizeScalar:
    """Protocol v2 (decision 5): action AND Q z-scored like Wilson targets."""

    @pytest.mark.parametrize("attr", ["y", "y_q"])
    def test_zscore_and_invert(self, h5file, attr):
        dataset = build_u1_dataset(h5file, "C")
        raw = torch.cat([getattr(d, attr) for d in dataset]).clone()
        stats = standardize_scalar_targets(dataset, attr)
        z = torch.cat([getattr(d, attr) for d in dataset])
        assert abs(z.mean().item()) < 1e-5
        assert abs(z.std().item() - 1) < 1e-5
        mean, std = stats
        torch.testing.assert_close(z * std + mean, raw, atol=1e-5, rtol=1e-5)

    def test_q_integers_survive_roundtrip(self, h5file):
        """De-standardized Q must round back to the exact original integers
        (the q_rounded_accuracy metric depends on this)."""
        dataset = build_u1_dataset(h5file, "C")
        raw = torch.cat([d.y_q for d in dataset]).clone()
        mean, std = standardize_scalar_targets(dataset, "y_q")
        z = torch.cat([d.y_q for d in dataset])
        assert torch.equal(torch.round(z * std + mean), torch.round(raw))

    def test_reuse_train_stats(self, h5file):
        train = build_u1_dataset(h5file, "C", n_configs=6)
        val = build_u1_dataset(h5file, "C")
        stats = standardize_scalar_targets(train, "y_q")
        stats_val = standardize_scalar_targets(val, "y_q", stats=stats)
        assert stats_val == stats
        torch.testing.assert_close(train[0].y_q, val[0].y_q)

    def test_single_config_raises(self, h5file):
        dataset = build_u1_dataset(h5file, "C", n_configs=1)
        with pytest.raises(ValueError, match="need >=2"):
            standardize_scalar_targets(dataset, "y_q")


class TestGaugeAugmentation:
    """On-the-fly gauge augmentation wrapper (task A-5)."""

    def test_config_idx_stamped(self, h5file):
        dataset = build_u1_dataset(h5file, "A")
        assert [d.config_idx for d in dataset] == list(range(N))

    def test_no_augment_passthrough(self, h5file):
        train = build_u1_dataset(h5file, "A", n_configs=4)
        ds = gauge_augmented_train_split(h5file, "A", train, seed=0, augment=False)
        assert len(ds) == 4
        for i in range(4):
            assert ds[i] is train[i]

    def test_labels_copied_features_change(self, h5file):
        """Augmented graphs keep the (gauge-invariant) targets but carry
        transformed link features; repeated access re-transforms."""
        train = build_u1_dataset(h5file, "A", n_configs=4)
        standardize_wilson_targets(train)  # wrap AFTER standardizing, as train_u1 does
        ds = gauge_augmented_train_split(h5file, "A", train, seed=0)
        first = ds[2]
        assert torch.equal(first.y, train[2].y)
        assert torch.equal(first.y_wilson, train[2].y_wilson)
        assert torch.equal(first.y_q, train[2].y_q)
        assert first.wilson_names == train[2].wilson_names
        assert first.config_idx == 2
        assert not torch.equal(first["gauge"].x, train[2]["gauge"].x)
        second = ds[2]  # fresh transform every access ("on-the-fly")
        assert not torch.equal(second["gauge"].x, first["gauge"].x)

    def test_variant_b_edge_features_change(self, h5file):
        train = build_u1_dataset(h5file, "B", n_configs=4)
        ds = gauge_augmented_train_split(h5file, "B", train, seed=0)
        assert not torch.equal(
            ds[1][ADJACENT].edge_attr, train[1][ADJACENT].edge_attr
        )

    def test_variant_c_features_bit_identical(self, h5file):
        """The invariant oracle sees THE SAME inputs under augmentation —
        float64 transform noise is below float32 feature resolution."""
        train = build_u1_dataset(h5file, "C", n_configs=4)
        ds = gauge_augmented_train_split(h5file, "C", train, seed=0)
        for i in range(4):
            assert torch.equal(ds[i]["spacetime"].x, train[i]["spacetime"].x)

    def test_deterministic_given_seed(self, h5file):
        train = build_u1_dataset(h5file, "A", n_configs=4)
        ds1 = gauge_augmented_train_split(h5file, "A", train, seed=7)
        ds2 = gauge_augmented_train_split(h5file, "A", train, seed=7)
        for i in (0, 3, 1):  # same access order
            assert torch.equal(ds1[i]["gauge"].x, ds2[i]["gauge"].x)

    def test_alignment_via_config_idx(self, h5file):
        """A non-leading subset (e.g. after --n_train slicing or any future
        reordering) still picks the right thetas: variant C features of the
        augmented graph match the base graph built from the same config."""
        dataset = build_u1_dataset(h5file, "C")
        subset = [dataset[5], dataset[2]]  # out of order, non-leading
        ds = gauge_augmented_train_split(h5file, "C", subset, seed=0)
        assert torch.equal(ds[0]["spacetime"].x, dataset[5]["spacetime"].x)
        assert torch.equal(ds[1]["spacetime"].x, dataset[2]["spacetime"].x)

    def test_missing_config_idx_raises(self, h5file):
        train = build_u1_dataset(h5file, "A", n_configs=2)
        del train[1].config_idx
        with pytest.raises(ValueError, match="config_idx"):
            gauge_augmented_train_split(h5file, "A", train, seed=0)

    def test_dataloader_batching(self, h5file):
        """PyG DataLoader over the wrapper: shapes match the plain path and
        a model forward consumes the batch."""
        train = build_u1_dataset(h5file, "A", n_configs=6)
        ds = gauge_augmented_train_split(h5file, "A", train, seed=0)
        batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False)))
        assert batch.y.shape == (4,)
        assert batch.y_wilson.shape == (4, len(LOOPS))
        assert batch.y_q.shape == (4,)
        assert batch.globals.shape == (4, 1)
        config = ModelConfig(hidden_dim=16, n_mp_blocks=2, encoder_layers=1)
        model = U1GaugeGNN(config, variant="link_nodes", wilson_loops=LOOPS).eval()
        out = model(batch)
        assert out["energy"].shape == (4,)
