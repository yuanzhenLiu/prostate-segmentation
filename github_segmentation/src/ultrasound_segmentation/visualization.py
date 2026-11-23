from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def blend_mask(image: np.ndarray, mask: np.ndarray, color: list[int], alpha: float) -> np.ndarray:
    """Blend color into ROI pixels while leaving background pixels unchanged."""
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    output = image.astype(np.float32).copy()
    roi = mask.astype(bool)
    output[roi] = output[roi] * (1.0 - alpha) + np.asarray(color, dtype=np.float32) * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def save_comparison(path: str | Path, image: np.ndarray, target: np.ndarray, prediction: np.ndarray, title: str) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title("Image")
    axes[1].imshow(blend_mask(image, target, [0, 220, 120], 0.45))
    axes[1].set_title("Ground truth")
    axes[2].imshow(blend_mask(image, prediction, [255, 0, 0], 0.45))
    axes[2].set_title("Prediction")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(title)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_mask_and_overlay(
    output_dir: str | Path,
    relative_path: Path,
    original: Image.Image,
    probability: np.ndarray,
    threshold: float,
    color: list[int],
    alpha: float,
) -> tuple[Path, Path, Path]:
    destination = Path(output_dir) / relative_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    stem = relative_path.stem
    mask = probability > threshold
    original_array = np.asarray(original.convert("RGB"))
    mask_path = destination / f"{stem}_mask.png"
    overlay_path = destination / f"{stem}_overlay.png"
    probability_path = destination / f"{stem}_probability.npy"
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
    Image.fromarray(blend_mask(original_array, mask, color, alpha)).save(overlay_path)
    np.save(probability_path, probability.astype(np.float32))
    return mask_path, overlay_path, probability_path
