from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class ClassicUNet(nn.Module):
    """The exact Classic U-Net architecture used by the final v2 experiment."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 24, dropout: float = 0.15):
        super().__init__()
        base = base_channels
        self.down1 = DoubleConv(in_channels, base, dropout)
        self.down2 = DoubleConv(base, base * 2, dropout)
        self.down3 = DoubleConv(base * 2, base * 4, dropout)
        self.down4 = DoubleConv(base * 4, base * 8, dropout)
        self.pool = nn.MaxPool2d(2)
        self.mid = DoubleConv(base * 8, base * 16, dropout)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.conv4 = DoubleConv(base * 16, base * 8, dropout)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.conv3 = DoubleConv(base * 8, base * 4, dropout)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.conv2 = DoubleConv(base * 4, base * 2, dropout)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.conv1 = DoubleConv(base * 2, base, dropout)
        self.out = nn.Conv2d(base, out_channels, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        down1 = self.down1(inputs)
        down2 = self.down2(self.pool(down1))
        down3 = self.down3(self.pool(down2))
        down4 = self.down4(self.pool(down3))
        outputs = self.mid(self.pool(down4))
        outputs = self.conv4(torch.cat([self.up4(outputs), down4], dim=1))
        outputs = self.conv3(torch.cat([self.up3(outputs), down3], dim=1))
        outputs = self.conv2(torch.cat([self.up2(outputs), down2], dim=1))
        outputs = self.conv1(torch.cat([self.up1(outputs), down1], dim=1))
        return self.out(outputs)


def build_model(model_config: dict) -> ClassicUNet:
    if model_config.get("name", "unet").lower() != "unet":
        raise ValueError("This repository contains only the final Classic U-Net v2 model.")
    if model_config.get("pretrained", False):
        raise ValueError("Classic U-Net v2 was trained from scratch and has no pretrained backbone.")
    return ClassicUNet(
        in_channels=int(model_config.get("in_channels", 3)),
        out_channels=int(model_config.get("out_channels", 1)),
        base_channels=int(model_config.get("base_channels", 24)),
        dropout=float(model_config.get("dropout", 0.15)),
    )
