from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ultrasound_segmentation.config import load_config  # noqa: E402
from ultrasound_segmentation.data import SegmentationDataset  # noqa: E402
from ultrasound_segmentation.losses import BCEDiceLoss  # noqa: E402
from ultrasound_segmentation.metrics import dice_from_logits  # noqa: E402
from ultrasound_segmentation.model import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the final Classic U-Net v2 model on a GPU.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "unet_v2.json"))
    parser.add_argument("--split-table", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=None, help="Optional reproducibility seed; original run used no global seed.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA, but torch.cuda.is_available() is False. Install a GPU-compatible PyTorch build.")
    try:
        (torch.ones(1, device="cuda") + 1).cpu()
    except RuntimeError as error:
        raise RuntimeError("CUDA is visible but could not execute a kernel. Install a PyTorch build that supports this GPU.") from error
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | CUDA {torch.version.cuda}")
    return torch.device("cuda")


def set_seed(seed: int | None) -> None:
    if seed is None:
        print("Global training seed: unset (matches the original v2 experiment).")
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Global training seed: {seed}")


def run_epoch(model, loader, criterion, optimizer, scaler, device, description: str) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_dice = 0.0
    processed = 0
    progress = tqdm(loader, desc=description, leave=False, dynamic_ncols=True, file=sys.stdout)
    with torch.set_grad_enabled(training):
        for images, masks, _ in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(images)
                loss = criterion(logits, masks)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            batch = images.size(0)
            processed += batch
            total_loss += float(loss.detach()) * batch
            total_dice += dice_from_logits(logits.detach(), masks) * batch
            progress.set_postfix(loss=f"{total_loss / processed:.4f}", dice=f"{total_dice / processed:.4f}")
    return {"loss": total_loss / processed, "dice": total_dice / processed}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = config["training"]
    batch_size = args.batch_size or int(training["batch_size"])
    workers = int(training["num_workers"] if args.num_workers is None else args.num_workers)
    seed = training.get("seed") if args.seed is None else args.seed
    device = require_cuda()
    set_seed(seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    train_data = SegmentationDataset(args.split_table, "train", config, augment=True)
    validation_data = SegmentationDataset(args.split_table, "val", config, augment=False)
    if not train_data or not validation_data:
        raise RuntimeError("Train and validation sets must both contain samples.")
    print(f"Training samples after frame sampling: {len(train_data)}")
    print(f"Validation samples without frame sampling: {len(validation_data)}")
    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True)
    validation_loader = DataLoader(validation_data, batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    model = build_model(config["model"]).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    best_dice, best_epoch, without_improvement = -1.0, 0, 0
    history = []
    for epoch in range(1, int(training["epochs"]) + 1):
        train_log = run_epoch(model, train_loader, criterion, optimizer, scaler, device, f"train {epoch}/{training['epochs']}")
        val_log = run_epoch(model, validation_loader, criterion, None, scaler, device, f"val   {epoch}/{training['epochs']}")
        history.append({"epoch": epoch, "train": train_log, "val": val_log})
        print(f"epoch {epoch:03d} | train loss {train_log['loss']:.4f} dice {train_log['dice']:.4f} | val loss {val_log['loss']:.4f} dice {val_log['dice']:.4f}")
        if val_log["dice"] > best_dice + float(training["min_delta"]):
            best_dice, best_epoch, without_improvement = val_log["dice"], epoch, 0
            torch.save({
                "model": model.state_dict(),
                "model_name": "unet",
                "image_size": int(config["data"]["image_size"]),
                "best_dice": best_dice,
                "config": config,
            }, checkpoint_dir / "unet_best.pt")
            print(f"Saved best checkpoint at epoch {epoch}: val Dice {best_dice:.4f}")
        else:
            without_improvement += 1
            print(f"Early-stop counter: {without_improvement}/{training['patience']}")
        if without_improvement >= int(training["patience"]):
            print(f"Early stopping. Best epoch: {best_epoch}; best val Dice: {best_dice:.4f}")
            break
    (output_dir / "history_unet.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
