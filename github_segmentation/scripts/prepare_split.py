from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ultrasound_segmentation.data import SUPPORTED_EXTENSIONS  # noqa: E402

PATIENT_FRAME = re.compile(r"^(P-+[^_]+)_([^_]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a leakage-safe patient-level 7:1:2 split table.")
    parser.add_argument("--data-root", required=True, help="Directory containing image1/mask1/image2/mask2.")
    parser.add_argument("--output", required=True, help="Output .xlsx or .csv split table.")
    parser.add_argument("--seed", type=int, default=2026, help="Patient shuffling seed.")
    return parser.parse_args()


def collect(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path.resolve()
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }


def patient_and_frame(stem: str) -> tuple[str, str]:
    match = PATIENT_FRAME.match(stem)
    if not match:
        raise ValueError(f"Filename does not match P-XXX_YYY: {stem}")
    patient = re.sub(r"^P-+", "P-", match.group(1))
    return patient, match.group(2)


def assign_patients(patient_ids: list[str], seed: int) -> dict[str, str]:
    ids = sorted(patient_ids)
    random.Random(seed).shuffle(ids)
    train_end = round(len(ids) * 0.7)
    validation_end = train_end + round(len(ids) * 0.1)
    return {
        patient: "train" if index < train_end else "val" if index < validation_end else "test"
        for index, patient in enumerate(ids)
    }


def main() -> None:
    args = parse_args()
    root = Path(args.data_root).expanduser().resolve()
    pairs = [
        ("image1_mask1", root / "image1", root / "mask1"),
        ("image2_mask2", root / "image2", root / "mask2"),
    ]
    rows, unmatched = [], []
    for source, image_dir, mask_dir in pairs:
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing pair directories: {image_dir} | {mask_dir}")
        images, masks = collect(image_dir), collect(mask_dir)
        for stem, image_path in sorted(images.items()):
            mask_path = masks.get(stem)
            if mask_path is None:
                unmatched.append({"source": source, "stem": stem, "problem": "missing_mask"})
                continue
            patient_id, frame_id = patient_and_frame(stem)
            rows.append({
                "source": source,
                "filename": image_path.name,
                "stem": stem,
                "patient_id": patient_id,
                "frame_id": frame_id,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            })
        for stem in sorted(set(masks).difference(images)):
            unmatched.append({"source": source, "stem": stem, "problem": "missing_image"})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No image/mask pairs found under {root}")
    assignment = assign_patients(frame["patient_id"].unique().tolist(), args.seed)
    frame["split"] = frame["patient_id"].map(assignment)
    if frame.groupby("patient_id")["split"].nunique().max() != 1:
        raise RuntimeError("Patient leakage detected while constructing the split.")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        frame.to_csv(output, index=False)
        pd.DataFrame(unmatched).to_csv(output.with_name(output.stem + "_unmatched.csv"), index=False)
    elif output.suffix.lower() == ".xlsx":
        with pd.ExcelWriter(output) as writer:
            frame.to_excel(writer, sheet_name="paired_split", index=False)
            pd.DataFrame(unmatched).to_excel(writer, sheet_name="unmatched", index=False)
    else:
        raise ValueError("--output must end in .xlsx or .csv")
    image_counts = frame["split"].value_counts().to_dict()
    patient_counts = frame.groupby("split")["patient_id"].nunique().to_dict()
    print(f"Saved private split table: {output}")
    print(f"Images: {image_counts}")
    print(f"Patients: {patient_counts}")
    print(f"Unmatched files: {len(unmatched)}")


if __name__ == "__main__":
    main()
