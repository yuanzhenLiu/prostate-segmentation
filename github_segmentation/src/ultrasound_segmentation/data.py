from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_split_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, sheet_name="paired_split")
    else:
        raise ValueError("Split table must be CSV or Excel.")
    required = {"source", "stem", "patient_id", "frame_id", "image_path", "mask_path", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Split table is missing columns: {sorted(missing)}")
    return frame


def frame_to_int(value) -> int:
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else 0


def sample_video_frames(frame: pd.DataFrame, stride: int, maximum: int) -> pd.DataFrame:
    """Sample only the training rows, independently within each source/patient group."""
    if stride <= 1 and maximum <= 0:
        return frame.reset_index(drop=True)
    groups = []
    for _, group in frame.groupby(["source", "patient_id"], sort=False):
        group = group.copy()
        group["_frame_number"] = group["frame_id"].map(frame_to_int)
        group = group.sort_values(["_frame_number", "stem"]).reset_index(drop=True)
        if stride > 1:
            group = group.iloc[::stride]
        if maximum > 0:
            group = group.iloc[:maximum]
        groups.append(group.drop(columns="_frame_number"))
    return pd.concat(groups, ignore_index=True) if groups else frame.iloc[0:0].copy()


def pil_resampling(name: str):
    resampling = getattr(Image, "Resampling", Image)
    return getattr(resampling, name)


def normalize_image(image: Image.Image, size: int, mean: list[float], std: list[float]) -> torch.Tensor:
    image = image.convert("RGB").resize((size, size), pil_resampling("BILINEAR"))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(array.transpose(2, 0, 1)).float()


def denormalize_tensor(image: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    array = image.detach().cpu().numpy().transpose(1, 2, 0)
    array = array * np.asarray(std) + np.asarray(mean)
    return np.clip(array, 0.0, 1.0)


class SegmentationDataset(Dataset):
    def __init__(
        self,
        split_table: str | Path,
        split: str,
        config: dict,
        augment: bool = False,
    ):
        frame = load_split_table(split_table)
        self.frame = frame[frame["split"] == split].reset_index(drop=True)
        data = config["data"]
        if split == "train":
            self.frame = sample_video_frames(
                self.frame,
                int(data["frame_stride"]),
                int(data["max_frames_per_group"]),
            )
        self.split = split
        self.size = int(data["image_size"])
        self.mean = data["normalization_mean"]
        self.std = data["normalization_std"]
        self.augment = augment
        self.augmentation = config["augmentation"]

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Missing pair: {image_path} | {mask_path}")
        image = Image.open(image_path).convert("RGB").resize((self.size, self.size), pil_resampling("BILINEAR"))
        mask = Image.open(mask_path).convert("L").resize((self.size, self.size), pil_resampling("NEAREST"))

        if self.augment:
            aug = self.augmentation
            if np.random.rand() < float(aug["horizontal_flip_probability"]):
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if np.random.rand() < float(aug["rotation_probability"]):
                degrees = float(aug["rotation_degrees"])
                angle = float(np.random.uniform(-degrees, degrees))
                image = image.rotate(angle, resample=pil_resampling("BILINEAR"), fillcolor=(0, 0, 0))
                mask = mask.rotate(angle, resample=pil_resampling("NEAREST"), fillcolor=0)
            if np.random.rand() < 0.5:
                value = float(aug["brightness"])
                image = ImageEnhance.Brightness(image).enhance(float(np.random.uniform(1 - value, 1 + value)))
            if np.random.rand() < 0.5:
                value = float(aug["contrast"])
                image = ImageEnhance.Contrast(image).enhance(float(np.random.uniform(1 - value, 1 + value)))
            if np.random.rand() < float(aug["blur_probability"]):
                image = image.filter(ImageFilter.GaussianBlur(radius=float(np.random.uniform(0.3, 0.8))))

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        if self.augment and np.random.rand() < 0.5:
            noise_std = float(self.augmentation["noise_std"])
            noise = np.random.normal(0.0, noise_std, image_array.shape).astype(np.float32)
            image_array = np.clip(image_array + noise, 0.0, 1.0)
        image_array = (image_array - np.asarray(self.mean, dtype=np.float32)) / np.asarray(self.std, dtype=np.float32)
        mask_array = (np.asarray(mask) > 0).astype(np.float32)
        metadata = {
            "source": str(row["source"]),
            "stem": str(row["stem"]),
            "patient_id": str(row["patient_id"]),
            "frame_id": str(row["frame_id"]),
        }
        return (
            torch.from_numpy(image_array.transpose(2, 0, 1)).float(),
            torch.from_numpy(mask_array[None]).float(),
            metadata,
        )
