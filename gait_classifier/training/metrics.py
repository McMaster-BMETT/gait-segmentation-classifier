"""Evaluation metrics for gait phase segmentation."""

import numpy as np
import torch
from ..constants import NUM_PHASES, PHASE_NAMES


def compute_confusion_matrix(preds: np.ndarray, labels: np.ndarray,
                             num_classes: int = NUM_PHASES,
                             ignore_index: int = -1) -> np.ndarray:
    """Compute confusion matrix from flat predictions and labels.

    Returns:
        (num_classes, num_classes) array where [true, pred] = count
    """
    mask = labels != ignore_index
    preds, labels = preds[mask], labels[mask]
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def per_phase_f1(confusion_matrix: np.ndarray) -> dict[str, float]:
    """Compute per-phase precision, recall, F1 from confusion matrix."""
    results = {}
    for i in range(confusion_matrix.shape[0]):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[PHASE_NAMES[i]] = {"precision": precision, "recall": recall, "f1": f1}
    return results


def overall_accuracy(preds: np.ndarray, labels: np.ndarray,
                     ignore_index: int = -1) -> float:
    """Compute overall accuracy ignoring padded positions."""
    mask = labels != ignore_index
    return (preds[mask] == labels[mask]).mean()


@torch.no_grad()
def evaluate_model(model, dataloader, device: str = "cpu") -> dict:
    """Run full evaluation on a dataloader.

    Returns dict with accuracy, per-phase F1, and confusion matrix.
    """
    model.eval()
    all_preds, all_labels = [], []

    for data, labels, lengths in dataloader:
        data = data.to(device)
        logits = model(data)  # (B, C, T)
        pred = logits.argmax(dim=1).cpu().numpy()  # (B, T)
        all_preds.append(pred.flatten())
        all_labels.append(labels.numpy().flatten())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    cm = compute_confusion_matrix(preds, labels)
    return {
        "accuracy": overall_accuracy(preds, labels),
        "per_phase_f1": per_phase_f1(cm),
        "confusion_matrix": cm,
    }
