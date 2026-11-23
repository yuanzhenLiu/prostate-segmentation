from __future__ import annotations

from pathlib import Path

import torch

from .model import build_model


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_trained_model(path: str | Path, config: dict, device: torch.device):
    checkpoint = load_checkpoint(path, device)
    state_dict = checkpoint.get("model", checkpoint)
    model = build_model(config["model"])
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    return model, checkpoint
