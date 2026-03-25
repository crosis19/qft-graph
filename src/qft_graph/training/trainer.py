"""Training loop with checkpointing, logging, and metrics."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch_geometric.loader import DataLoader

from qft_graph.config import TrainingConfig
from qft_graph.training.losses import EnergyMatchingLoss, KLDivergenceLoss, RelativeEnergyLoss
from qft_graph.training.metrics import energy_correlation, energy_std_ratio, relative_error
from qft_graph.utils.checkpointing import save_checkpoint

logger = logging.getLogger("qft_graph.training")


class Trainer:
    """Training loop for the heterogeneous GNN energy model.

    Manages:
    - DataLoader over HeteroData graphs
    - Optimizer and LR scheduler
    - Periodic checkpoint saving
    - Metric computation and logging
    - TensorBoard logging (optional)

    Args:
        model: HeteroGNN model.
        train_dataset: List of HeteroData training graphs.
        val_dataset: List of HeteroData validation graphs.
        config: TrainingConfig with training hyperparameters.
        experiment_dir: Directory for checkpoints and logs.
        device: Device to train on.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: list,
        val_dataset: list,
        config: TrainingConfig,
        experiment_dir: str | Path = "experiments/runs/default",
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False
        )

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        # Scheduler
        if config.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config.epochs
            )
        elif config.scheduler == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, patience=10, factor=0.5
            )
        else:
            self.scheduler = None

        # Loss function
        if config.loss == "energy_matching":
            self.criterion = EnergyMatchingLoss()
        elif config.loss == "kl_divergence":
            self.criterion = KLDivergenceLoss()
        elif config.loss == "relative_energy":
            self.criterion = RelativeEnergyLoss()
        else:
            raise ValueError(f"Unknown loss: {config.loss}")

        # TensorBoard
        self._writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(str(self.experiment_dir / "tb_logs"))
        except ImportError:
            logger.warning("TensorBoard not available, skipping logging")

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_corr": [],
        }

    def train(self) -> dict[str, list[float]]:
        """Run the full training loop.

        Returns:
            History dict with per-epoch metrics.
        """
        logger.info("Starting training for %d epochs", self.config.epochs)
        start_time = time.time()

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_epoch()
            val_metrics = self._validate()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_corr"].append(val_metrics["correlation"])

            # LR scheduling
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics["loss"])
            elif self.scheduler is not None:
                self.scheduler.step()

            # Logging
            lr = self.optimizer.param_groups[0]["lr"]
            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %d/%d | Train Loss: %.6f | Val Loss: %.6f | "
                    "Val Corr: %.4f | LR: %.2e",
                    epoch,
                    self.config.epochs,
                    train_loss,
                    val_metrics["loss"],
                    val_metrics["correlation"],
                    lr,
                )

            # TensorBoard
            if self._writer is not None:
                self._writer.add_scalar("loss/train", train_loss, epoch)
                self._writer.add_scalar("loss/val", val_metrics["loss"], epoch)
                self._writer.add_scalar("metrics/correlation", val_metrics["correlation"], epoch)
                self._writer.add_scalar("metrics/rel_error", val_metrics["rel_error"], epoch)
                self._writer.add_scalar("lr", lr, epoch)

            # Checkpointing
            if epoch % self.config.checkpoint_every == 0:
                path = self.experiment_dir / f"checkpoint_epoch_{epoch}.pt"
                save_checkpoint(
                    path, self.model, self.optimizer, epoch,
                    config=None, metrics=val_metrics,
                )

        elapsed = time.time() - start_time
        logger.info("Training complete in %.1f seconds", elapsed)

        # Save final checkpoint
        save_checkpoint(
            self.experiment_dir / "checkpoint_final.pt",
            self.model, self.optimizer, self.config.epochs,
            config=None, metrics=val_metrics,
        )

        if self._writer is not None:
            self._writer.close()

        return self.history

    def _train_epoch(self) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            output = self.model(batch)
            predicted = output["energy"]
            target = batch.y.to(self.device)

            loss = self.criterion(predicted, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _validate(self) -> dict[str, float]:
        """Run validation and compute metrics."""
        self.model.eval()
        all_pred = []
        all_true = []
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                output = self.model(batch)
                predicted = output["energy"]
                target = batch.y.to(self.device)

                loss = self.criterion(predicted, target)
                total_loss += loss.item()
                n_batches += 1

                all_pred.append(predicted.cpu())
                all_true.append(target.cpu())

        all_pred = torch.cat(all_pred)
        all_true = torch.cat(all_true)

        return {
            "loss": total_loss / max(n_batches, 1),
            "correlation": energy_correlation(all_pred, all_true),
            "rel_error": relative_error(all_pred, all_true),
            "std_ratio": energy_std_ratio(all_pred, all_true),
        }
