from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ultrasound_segmentation.checkpoint import load_trained_model  # noqa: E402
from ultrasound_segmentation.config import load_config  # noqa: E402
from ultrasound_segmentation.data import SUPPORTED_EXTENSIONS, normalize_image, pil_resampling  # noqa: E402
from ultrasound_segmentation.visualization import save_mask_and_overlay  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one image or every supported image in a directory.")
    parser.add_argument("--input", required=True, help="Image file or directory.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs" / "unet_v2.json"))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    return parser.parse_args()


def choose_device(choice: str) -> torch.device:
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Use --device cpu or install a compatible PyTorch build.")
    return torch.device(choice)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = choose_device(args.device)
    model, checkpoint = load_trained_model(args.checkpoint, config, device)
    source = Path(args.input).expanduser().resolve()
    if source.is_file():
        files = [source]
        source_root = source.parent
    elif source.is_dir():
        files = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
        source_root = source
    else:
        raise FileNotFoundError(source)
    if not files:
        raise FileNotFoundError(f"No supported images found in {source}")
    data_config, inference = config["data"], config["inference"]
    size = int(checkpoint.get("image_size", data_config["image_size"]))
    threshold = float(inference["threshold"] if args.threshold is None else args.threshold)
    alpha = float(inference["overlay_alpha"] if args.alpha is None else args.alpha)
    if not 0.0 < threshold < 1.0 or not 0.0 <= alpha <= 1.0:
        raise ValueError("Threshold must be in (0, 1), and alpha in [0, 1].")
    with torch.inference_mode():
        for path in files:
            original = Image.open(path).convert("RGB")
            tensor = normalize_image(original, size, data_config["normalization_mean"], data_config["normalization_std"])
            probability = model(tensor.unsqueeze(0).to(device)).sigmoid()[0, 0].cpu().numpy()
            probability = np.asarray(Image.fromarray(probability).resize(original.size, pil_resampling("BILINEAR")), dtype=np.float32)
            relative = path.relative_to(source_root)
            _, overlay_path, _ = save_mask_and_overlay(args.output_dir, relative, original, probability, threshold, inference["overlay_color_rgb"], alpha)
            print(f"Saved: {overlay_path}")
    print(f"Predicted {len(files)} image(s) on {device}.")


if __name__ == "__main__":
    main()
