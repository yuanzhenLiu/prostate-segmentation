from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot U-Net train/validation loss and Dice curves.")
    parser.add_argument("--history", required=True, help="history_unet.json")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def plot(frame: pd.DataFrame, train_column: str, val_column: str, label: str, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    color = "#1f77b4"
    axis.plot(frame["epoch"], frame[train_column], color=color, linestyle="-", linewidth=2, label="train")
    axis.plot(frame["epoch"], frame[val_column], color=color, linestyle="--", linewidth=2, label="validation")
    axis.set(xlabel="Epoch", ylabel=label, title=f"Classic U-Net {label}")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    history = json.loads(Path(args.history).read_text(encoding="utf-8"))
    rows = [{
        "epoch": item["epoch"],
        "train_loss": item["train"]["loss"],
        "train_dice": item["train"]["dice"],
        "val_loss": item["val"]["loss"],
        "val_dice": item["val"]["dice"],
    } for item in history]
    frame = pd.DataFrame(rows)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "training_history.csv", index=False)
    frame.loc[[frame["val_dice"].idxmax()]].to_csv(output / "best_validation_score.csv", index=False)
    plot(frame, "train_loss", "val_loss", "Loss", output / "loss_curves.png")
    plot(frame, "train_dice", "val_dice", "Dice", output / "dice_curves.png")
    print(f"Saved curves and tables to {output}")


if __name__ == "__main__":
    main()
