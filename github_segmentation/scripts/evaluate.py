from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ultrasound_segmentation.checkpoint import load_trained_model  # noqa: E402
from ultrasound_segmentation.config import load_config  # noqa: E402
from ultrasound_segmentation.data import SegmentationDataset, denormalize_tensor  # noqa: E402
from ultrasound_segmentation.metrics import binary_metrics  # noqa: E402
from ultrasound_segmentation.visualization import blend_mask, save_comparison  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate every test sample and save metrics and visualizations.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "unet_v2.json"))
    parser.add_argument("--split-table", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-visuals", type=int, default=30)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def choose_device(choice: str) -> torch.device:
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(choice)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(args.device)
    model, checkpoint = load_trained_model(args.checkpoint, config, device)
    dataset = SegmentationDataset(args.split_table, "test", config, augment=False)
    loader = DataLoader(dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    output = Path(args.output_dir).expanduser().resolve()
    overlays, comparisons = output / "prediction_overlays", output / "visualizations"
    overlays.mkdir(parents=True, exist_ok=True)
    comparisons.mkdir(parents=True, exist_ok=True)
    data_config, infer = config["data"], config["inference"]
    threshold = float(infer["threshold"])
    rows, visual_count = [], 0
    with torch.inference_mode():
        for images, masks, metadata in tqdm(loader, desc="evaluate", dynamic_ncols=True):
            probabilities = model(images.to(device)).sigmoid().cpu().numpy()[:, 0]
            targets = masks.numpy()[:, 0].astype(np.uint8)
            for index in range(images.size(0)):
                prediction = (probabilities[index] > threshold).astype(np.uint8)
                stem = metadata["stem"][index]
                row = {
                    "stem": stem,
                    "patient_id": metadata["patient_id"][index],
                    "frame_id": metadata["frame_id"][index],
                    **binary_metrics(prediction, targets[index]),
                }
                rows.append(row)
                image = denormalize_tensor(images[index], data_config["normalization_mean"], data_config["normalization_std"])
                overlay = blend_mask(image, prediction, infer["overlay_color_rgb"], float(infer["overlay_alpha"]))
                Image.fromarray(overlay).save(overlays / f"{stem}_overlay.png")
                if visual_count < args.num_visuals:
                    save_comparison(comparisons / f"{stem}_compare.png", image, targets[index], prediction, f"{metadata['patient_id'][index]}_{metadata['frame_id'][index]}")
                    visual_count += 1
    per_image = pd.DataFrame(rows)
    metric_columns = ["dice", "iou", "precision", "recall_sensitivity", "specificity", "accuracy", "hd95"]
    summary = per_image[metric_columns].agg(["mean", "std"]).T.reset_index(names="metric")
    per_image.to_csv(output / "metrics_per_image.csv", index=False)
    summary.to_csv(output / "metrics_summary.csv", index=False)
    print(f"Evaluated {len(per_image)} images on {device}.")
    print(f"Checkpoint best validation Dice: {checkpoint.get('best_dice', 'unknown')}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
