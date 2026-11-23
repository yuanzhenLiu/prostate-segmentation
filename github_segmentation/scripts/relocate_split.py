from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ultrasound_segmentation.data import load_split_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update split-table paths while preserving patient assignments.")
    parser.add_argument("--input", required=True, help="Existing private .xlsx or .csv split table.")
    parser.add_argument("--data-root", required=True, help="New root containing image1/mask1/image2/mask2.")
    parser.add_argument("--output", required=True, help="Relocated .xlsx or .csv table.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_split_table(args.input)
    root = Path(args.data_root).expanduser().resolve()
    source_dirs = {
        "image1_mask1": ("image1", "mask1"),
        "image2_mask2": ("image2", "mask2"),
    }
    image_paths, mask_paths = [], []
    for row in frame.itertuples(index=False):
        if row.source not in source_dirs:
            raise ValueError(f"Unknown source in split table: {row.source}")
        image_dir, mask_dir = source_dirs[row.source]
        image = root / image_dir / row.filename
        old_mask_suffix = Path(row.mask_path).suffix
        mask = root / mask_dir / f"{row.stem}{old_mask_suffix}"
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"Missing relocated pair: {image} | {mask}")
        image_paths.append(str(image.resolve()))
        mask_paths.append(str(mask.resolve()))
    frame["image_path"], frame["mask_path"] = image_paths, mask_paths
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        frame.to_csv(output, index=False)
    elif output.suffix.lower() == ".xlsx":
        with pd.ExcelWriter(output) as writer:
            frame.to_excel(writer, sheet_name="paired_split", index=False)
    else:
        raise ValueError("--output must end in .xlsx or .csv")
    print(f"Verified every pair and saved: {output}")


if __name__ == "__main__":
    main()
