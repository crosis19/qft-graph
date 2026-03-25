"""Model, optimizer, and RNG state checkpointing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Any,
    metrics: dict | None = None,
) -> None:
    """Save training checkpoint including RNG states."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "metrics": metrics or {},
            "rng_state_torch": torch.random.get_rng_state(),
            "rng_state_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> dict:
    """Load training checkpoint and restore RNG states.

    Returns:
        Dict with 'epoch', 'config', 'metrics' keys.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    torch.random.set_rng_state(ckpt["rng_state_torch"])
    if ckpt["rng_state_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["rng_state_cuda"])
    return {"epoch": ckpt["epoch"], "config": ckpt["config"], "metrics": ckpt["metrics"]}
