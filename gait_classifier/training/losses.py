"""Loss functions for gait phase segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(labels: list[torch.Tensor], num_classes: int = 6) -> torch.Tensor:
    """Compute inverse-frequency class weights from label tensors.

    Args:
        labels: list of (T,) label tensors
        num_classes: number of gait phases

    Returns:
        (num_classes,) weight tensor
    """
    counts = torch.zeros(num_classes)
    for l in labels:
        for c in range(num_classes):
            counts[c] += (l == c).sum()
    # Inverse frequency, normalized
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    return weights


class WeightedCELoss(nn.Module):
    """Weighted cross-entropy loss with label smoothing, ignoring padded positions."""

    def __init__(self, weight: torch.Tensor | None = None, label_smoothing: float = 0.05,
                 ignore_index: int = -1):
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, T) model output
            targets: (B, T) ground truth labels
        """
        B, C, T = logits.shape
        # Reshape to (B*T, C) and (B*T,)
        logits_flat = logits.permute(0, 2, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)
        return F.cross_entropy(logits_flat, targets_flat, weight=self.weight,
                               ignore_index=self.ignore_index,
                               label_smoothing=self.label_smoothing)
