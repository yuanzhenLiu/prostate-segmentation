from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a synthetic, non-patient ultrasound-like example image.")
    parser.add_argument("--output", default="examples/input/synthetic_ultrasound.png")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(2026)
    y, x = np.mgrid[0:args.height, 0:args.width]
    radial = np.sqrt(((x - args.width / 2) / args.width) ** 2 + ((y - args.height * 0.15) / args.height) ** 2)
    base = 35 + 75 * np.exp(-2.5 * radial) + rng.normal(0, 15, (args.height, args.width))
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.ellipse((args.width * 0.34, args.height * 0.42, args.width * 0.67, args.height * 0.72), fill=(115, 115, 115))
    image = image.filter(ImageFilter.GaussianBlur(2.0))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(f"Saved synthetic image: {output.resolve()}")


if __name__ == "__main__":
    main()
