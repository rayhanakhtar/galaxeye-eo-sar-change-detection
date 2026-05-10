"""Metrics helpers for binary change detection."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def logits_to_prediction(logits: Tensor, threshold: float = 0.5) -> Tensor:
    """Convert model logits to a binary mask."""
    if logits.shape[1] == 1:
        probs = torch.sigmoid(logits)
        return (probs >= threshold).long().squeeze(1)

    if logits.shape[1] == 2:
        probs = torch.softmax(logits, dim=1)[:, 1]
        return (probs >= threshold).long()

    raise ValueError(f"Unsupported output channel count: {logits.shape[1]}")


def compute_confusion_matrix(
    pred: Tensor, target: Tensor, num_classes: int = 2
) -> Tensor:
    """Compute confusion matrix over all pixels in a batch."""
    pred = pred.view(-1).long()
    target = target.view(-1).long()
    valid = (target >= 0) & (target < num_classes)
    indices = target[valid] * num_classes + pred[valid]
    counts = torch.bincount(indices, minlength=num_classes * num_classes)
    return counts.view(num_classes, num_classes)


def metrics_from_confusion_matrix(
    conf_matrix: Tensor,
) -> dict[str, float | list[list[int]]]:
    """Compute IoU, precision, recall, and F1 for the change class."""
    conf = conf_matrix.detach().cpu().to(torch.float64)
    fp = conf[0, 1].item()
    fn = conf[1, 0].item()
    tp = conf[1, 1].item()

    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall + 1e-8)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": conf.to(torch.int64).tolist(),
    }


def average_loss(total_loss: float, num_batches: int) -> float:
    """Safely average a summed batch loss."""
    return total_loss / max(num_batches, 1)


def metrics_to_numpy(
    metrics: dict[str, float | list[list[int]]],
) -> dict[str, float | np.ndarray]:
    """Convert list confusion matrix to numpy for saving."""
    output: dict[str, float | np.ndarray] = {}
    for key, value in metrics.items():
        if key == "confusion_matrix":
            output[key] = np.asarray(value)
        else:
            output[key] = value
    return output
