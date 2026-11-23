from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def hd95(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return float("nan")
    pred_border = prediction ^ binary_erosion(prediction)
    target_border = target ^ binary_erosion(target)
    distance_to_pred = distance_transform_edt(~pred_border)
    distance_to_target = distance_transform_edt(~target_border)
    distances = np.concatenate([distance_to_target[pred_border], distance_to_pred[target_border]])
    return float(np.percentile(distances, 95)) if distances.size else 0.0


def binary_metrics(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> dict[str, float]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(prediction, target).sum()
    tn = np.logical_and(~prediction, ~target).sum()
    fp = np.logical_and(prediction, ~target).sum()
    fn = np.logical_and(~prediction, target).sum()
    return {
        "dice": float((2 * tp + eps) / (2 * tp + fp + fn + eps)),
        "iou": float((tp + eps) / (tp + fp + fn + eps)),
        "precision": float((tp + eps) / (tp + fp + eps)),
        "recall_sensitivity": float((tp + eps) / (tp + fn + eps)),
        "specificity": float((tn + eps) / (tn + fp + eps)),
        "accuracy": float((tp + tn + eps) / (tp + tn + fp + fn + eps)),
        "hd95": hd95(prediction, target),
    }


def dice_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> float:
    prediction = (logits.sigmoid() > 0.5).float()
    intersection = (prediction * target).sum(dim=(1, 2, 3))
    denominator = prediction.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return float(((2 * intersection + eps) / (denominator + eps)).mean())
