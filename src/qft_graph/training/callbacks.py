"""Training callbacks for logging, early stopping, and LR scheduling."""

from __future__ import annotations

import logging

logger = logging.getLogger("qft_graph.training")


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait after last improvement.
        min_delta: Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def should_stop(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            logger.info(
                "Early stopping triggered after %d epochs without improvement",
                self.patience,
            )
            return True
        return False
