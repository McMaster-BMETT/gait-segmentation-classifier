"""Training loop with early stopping and augmentation."""

import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .metrics import evaluate_model


class GaitTrainer:
    """Handles training loop, validation, early stopping, and augmentation."""

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, criterion: nn.Module,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 epochs: int = 100, patience: int = 15,
                 device: str = "cpu",
                 augmentation: dict | None = None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.augmentation = augmentation or {}

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)

        self.best_val_loss = float("inf")
        self.best_state = None
        self.patience_counter = 0
        self.history: list[dict] = []

    def _augment_batch(self, data: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to a batch. Operates in-place for efficiency."""
        if not self.augmentation:
            return data

        # Gaussian noise
        noise_std = self.augmentation.get("gaussian_noise_std", 0)
        if noise_std > 0:
            data = data + torch.randn_like(data) * noise_std

        # Time warping (simple: random speed change per sample)
        warp_range = self.augmentation.get("time_warp_range", 0)
        if warp_range > 0:
            B, C, T = data.shape
            for i in range(B):
                factor = 1.0 + (torch.rand(1).item() * 2 - 1) * warp_range
                new_T = max(10, int(T * factor))
                warped = torch.nn.functional.interpolate(
                    data[i:i+1], size=new_T, mode="linear", align_corners=False
                )
                # Resize back to original T
                data[i:i+1] = torch.nn.functional.interpolate(
                    warped, size=T, mode="linear", align_corners=False
                )

        # EMG amplitude scaling (channels 6 and 7)
        scale_range = self.augmentation.get("emg_amplitude_scale_range")
        if scale_range:
            lo, hi = scale_range
            scale = torch.empty(data.shape[0], 1, 1).uniform_(lo, hi).to(data.device)
            data[:, 6:8, :] = data[:, 6:8, :] * scale

        return data

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for data, labels, lengths in self.train_loader:
            data = data.to(self.device)
            labels = labels.to(self.device)
            data = self._augment_batch(data)

            self.optimizer.zero_grad()
            logits = self.model(data)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for data, labels, lengths in self.val_loader:
            data = data.to(self.device)
            labels = labels.to(self.device)
            logits = self.model(data)
            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self) -> dict:
        """Run full training loop with early stopping.

        Returns:
            dict with training history and best validation metrics
        """
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step()

            record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                      "lr": self.scheduler.get_last_lr()[0]}
            self.history.append(record)

            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                      f"val_loss={val_loss:.4f} | lr={record['lr']:.2e}")

            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return {
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": len(self.history),
        }
