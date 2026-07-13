"""A-5: eps_gauge measurement on saved U(1) checkpoints (plan §3, task A-5).

For every run record matching --runs, load its checkpoint, rebuild the
test split of its ensemble (mirroring scripts/train_u1.py's deterministic
leading-block split), and for each test config generate K random gauge
copies. Per output head:

    eps_gauge = mean_configs[ std_over_gauge_copies(y_hat) ]
                / std_over_configs(y_hat)

(training.metrics.gauge_invariance_error). Predictions stay on the
model's standardized output scale — eps_gauge is affine-invariant.

Numerics conventions (both load-bearing, both pinned by tests):
- Gauge copies are generated in FLOAT64 from the float32-stored links,
  per-config seeded via default_rng([gauge_seed, file_config_index]), so
  every model evaluated on the same ensemble file sees identical copies
  (paired comparison) and Variant C's inputs are bit-identical across
  the orbit (A-3 oracle).
- Every graph is forwarded UNBATCHED (batch size 1). Batched evaluation
  is position-in-batch dependent at float32 lsb (~1e-8): bit-identical
  graphs at different positions of one Batch return slightly different
  outputs, which would put a spurious ~1e-8 floor under Variant C's
  orbit std. Per-graph forwards are deterministic, so the C sanity
  check ("~0 by construction") comes out exactly 0.

Writes results/a5eps_<source_run_id>.json per input run (ground rule 5);
existing outputs are skipped unless --force. Aggregate with
scripts/make_a5_table.py.

Usage:
  python scripts/measure_gauge_invariance.py            # A-4 comparison set
  python scripts/measure_gauge_invariance.py --runs "a5aug_*" "a5base_*"
  python scripts/measure_gauge_invariance.py --max_configs 20   # smoke test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qft_graph.config import LatticeConfig
from qft_graph.fields.gauge import gauge_orbit
from qft_graph.graphs.builder import U1GaugeGraphBuilder
from qft_graph.graphs.u1_dataset import load_u1_ensemble
from qft_graph.lattice.hypercubic import HypercubicLattice
from qft_graph.models.u1_gnn import U1GaugeGNN
from qft_graph.training.metrics import gauge_invariance_error
from qft_graph.utils.logging import setup_logging
from qft_graph.utils.run_logging import log_run

# The A-4 protocol-v2 comparison set: A/B null-robustness runs, Variant C
# at the same three (L, beta) cells (5 seeds), the parameter-matched arm,
# and the Variant A escape hatches. u1pilot_* (protocol v1) is excluded.
DEFAULT_PATTERNS = (
    "a4null_*",
    "a4C_invariant_oracle_u1_L8_beta1_H64_*",
    "a4C_invariant_oracle_u1_L8_beta2_H64_*",
    "a4C_invariant_oracle_u1_L16_beta2_H64_*",
    "a4match_*",
    "a4hatch*",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A-5 eps_gauge measurement")
    p.add_argument("--runs", type=str, nargs="+", default=list(DEFAULT_PATTERNS),
                   help="Glob patterns over results/<pattern>.json run records")
    p.add_argument("--k_copies", type=int, default=32,
                   help="Gauge copies per config K (plan A-5 protocol: 32)")
    p.add_argument("--gauge_seed", type=int, default=20260713,
                   help="Master seed; per-config rng is [gauge_seed, config_idx]")
    p.add_argument("--max_configs", type=int, default=None,
                   help="Cap on test configs per run (smoke tests); default all")
    p.add_argument("--device", type=str, default="cpu",
                   help="cpu is the intended target; single-graph forwards "
                        "gain little from GPU")
    p.add_argument("--force", action="store_true",
                   help="Recompute runs whose a5eps output already exists")
    return p.parse_args()


def find_run_records(patterns: list[str]) -> list[Path]:
    seen: dict[str, Path] = {}
    for pat in patterns:
        for path in sorted(Path("results").glob(f"{pat}.json")):
            if path.name.startswith("a5eps_"):
                continue  # never re-measure our own outputs
            d = json.loads(path.read_text())
            if "run_id" not in d or "checkpoint" not in d.get("extra", {}):
                continue
            seen[d["run_id"]] = path
    return [seen[k] for k in sorted(seen)]


def test_split_indices(rec: dict, n_available: int) -> range:
    """File positions of the test split, mirroring train_u1.split_dataset
    (deterministic leading-block 80/10/10 split, no shuffle)."""
    cfg = rec["config"]
    n = n_available if cfg.get("n_configs") is None else min(cfg["n_configs"], n_available)
    n_train = int(cfg.get("train_frac", 0.8) * n)
    n_val = int(cfg.get("val_frac", 0.1) * n)
    return range(n_train + n_val, n)


@torch.no_grad()
def measure_run(rec: dict, args, logger) -> dict | None:
    cfg = rec["config"]
    ckpt_path = Path(rec["extra"]["checkpoint"])
    if not ckpt_path.exists():
        logger.warning("checkpoint missing, skipping: %s", ckpt_path)
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    model = U1GaugeGNN(
        ckpt["model_config"],
        variant=ckpt["variant"],
        wilson_loops=tuple(ckpt["wilson_loops"]),
        predict_q=True,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(args.device).eval()
    keys = ("energy", *(f"wilson_{n}" for n in ckpt["wilson_loops"]), "q")

    ens = load_u1_ensemble(cfg["data"])
    L = ens["L"]
    lattice = HypercubicLattice(
        LatticeConfig(dimensions=(L, L), spacing=1.0, boundary="periodic")
    )
    builder = U1GaugeGraphBuilder(lattice, beta=ens["beta"], variant=ckpt["variant"])

    idx = list(test_split_indices(rec, ens["theta"].shape[0]))
    if args.max_configs is not None:
        idx = idx[: args.max_configs]
    if len(idx) < 2:
        raise ValueError(f"{rec['run_id']}: test split has {len(idx)} configs")

    orbit_preds = {k: [] for k in keys}
    config_preds = {k: [] for k in keys}
    t0 = time.time()
    for count, i in enumerate(idx):
        theta32 = ens["theta"][i].numpy()
        copies = gauge_orbit(
            theta32, args.k_copies, np.random.default_rng([args.gauge_seed, i])
        )
        graphs = [builder.build({"gauge": torch.from_numpy(theta32)})]
        graphs += [builder.build({"gauge": torch.from_numpy(c)}) for c in copies]
        outs = [model(g.to(args.device)) for g in graphs]  # batch size 1 (module docstring)
        for k in keys:
            vals = torch.cat([o[k].cpu() for o in outs])
            config_preds[k].append(vals[0])
            orbit_preds[k].append(vals[1:])
        if (count + 1) % 100 == 0:
            logger.info("  %d/%d configs (%.1f s)", count + 1, len(idx), time.time() - t0)

    metrics = {
        k: gauge_invariance_error(
            torch.stack(orbit_preds[k]), torch.stack(config_preds[k])
        )
        for k in keys
    }
    metrics["wall_time_s"] = time.time() - t0
    return metrics


def main() -> None:
    args = parse_args()
    logger = setup_logging()
    records = find_run_records(args.runs)
    if not records:
        raise SystemExit(f"No run records match {args.runs}")
    logger.info("%d run records to measure", len(records))

    done = 0
    for path in records:
        rec = json.loads(path.read_text())
        run_id = rec["run_id"]
        if rec["config"].get("protocol_version") != 2:
            logger.warning("skipping non-v2 run: %s", run_id)
            continue
        out_path = Path("results") / f"a5eps_{run_id}.json"
        if out_path.exists() and not args.force:
            logger.info("exists, skipping (use --force): %s", out_path.name)
            continue

        logger.info("measuring %s", run_id)
        metrics = measure_run(rec, args, logger)
        if metrics is None:
            continue
        log_run(
            f"a5eps_{run_id}",
            config={
                "source_run_id": run_id,
                "source_config_hash": rec["config_hash"],
                "checkpoint": rec["extra"]["checkpoint"],
                "data": rec["config"]["data"],
                "variant": rec["config"]["variant"],
                "k_copies": args.k_copies,
                "gauge_seed": args.gauge_seed,
                "max_configs": args.max_configs,
                "protocol_version": 2,
            },
            metrics=metrics,
            extra={"source_test_metrics": rec["metrics"]["test"]},
        )
        eps_summary = {
            k: round(v["eps_gauge"], 4)
            for k, v in metrics.items()
            if isinstance(v, dict)
        }
        logger.info("  -> %s | eps: %s", out_path.name, eps_summary)
        done += 1
    logger.info("done: %d measured, %d total matched", done, len(records))


if __name__ == "__main__":
    main()
