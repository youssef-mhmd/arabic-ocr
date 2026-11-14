"""The text-line detector: a small CNN that paints a "line core" map over a page.

Input:  grayscale page, uint8 or float in [0, 255], shape (B, 1, H, W); H and W must
        be multiples of 32.
Output: logits (B, 1, H, W): for every pixel, how likely it is to be in the core of
        a text line (apply a sigmoid to get a 0-1 probability). See
        arabic_ocr/detect/boxes.py for how cores become line boxes.

The three parts ("DBNet-style", like most modern text detectors):
1. **Backbone**: residual CNN blocks that shrink the page step by step (to 1/4,
   1/8, 1/16 and 1/32 of its size). Early levels see fine detail (dots, strokes);
   deep levels see wide context (a whole line, a column).
2. **Feature pyramid (FPN)**: mixes the levels back together, so each pixel's
   decision uses both fine detail and context, then brings them to 1/4 size.
3. **Head**: two upsampling steps back to full size and one score per pixel.

Trained from scratch (no ImageNet weights), so its license is only our data's.
Size: about 3.6 million weights, ~14 MB as float32 (PRD target: <= 25 MB).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def conv_bn(c_in: int, c_out: int, k: int = 3, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, k, stride, padding=k // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with a shortcut ("residual" block, as in ResNet).
    The shortcut lets information and gradients skip ahead, so deeper networks
    still train well."""

    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = conv_bn(c_in, c_out, 3, stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False), nn.BatchNorm2d(c_out)
        )
        self.short = (
            nn.Identity()
            if stride == 1 and c_in == c_out
            else nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.conv2(self.conv1(x)) + self.short(x))


def stage(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(BasicBlock(c_in, c_out, 2), BasicBlock(c_out, c_out))


class Detector(nn.Module):
    def __init__(self, widths=(48, 96, 160, 256), fpn: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(conv_bn(1, 24, 3, 2), conv_bn(24, 32))  # 1/2
        self.c2 = stage(32, widths[0])  # 1/4
        self.c3 = stage(widths[0], widths[1])  # 1/8
        self.c4 = stage(widths[1], widths[2])  # 1/16
        self.c5 = stage(widths[2], widths[3])  # 1/32
        self.lateral = nn.ModuleList(nn.Conv2d(c, fpn, 1) for c in widths)
        self.smooth = nn.ModuleList(conv_bn(fpn, fpn // 2) for _ in widths)
        self.head = nn.Sequential(
            conv_bn(2 * fpn, fpn),  # 4 levels x fpn/2 channels = 2 * fpn
            nn.ConvTranspose2d(fpn, fpn // 2, 2, 2),  # 1/4 -> 1/2
            nn.BatchNorm2d(fpn // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(fpn // 2, 1, 2, 2),  # 1/2 -> full size
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = (images.float() / 255.0 - 0.5) / 0.5
        x = self.stem(x)
        feats = []
        for layer in (self.c2, self.c3, self.c4, self.c5):
            x = layer(x)
            feats.append(x)
        # Top-down: each level gets the (upsampled) context of the level below it.
        lat = [conv(f) for conv, f in zip(self.lateral, feats, strict=True)]
        for i in range(len(lat) - 2, -1, -1):
            lat[i] = lat[i] + F.interpolate(lat[i + 1], size=lat[i].shape[2:], mode="nearest")
        size = lat[0].shape[2:]
        merged = [
            F.interpolate(s(p), size=size, mode="nearest") if i else s(p)
            for i, (s, p) in enumerate(zip(self.smooth, lat, strict=True))
        ]
        return self.head(torch.cat(merged, dim=1))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
