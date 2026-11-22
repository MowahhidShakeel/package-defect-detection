"""
The inspector: an encoder-decoder network that outputs a defect map.

    crop (3 x H x W)  ->  encoder  ->  decoder  ->  map (1 x H x W)

Each pixel of the map is the probability that the print is defective there.
The carton's score is taken from the strongest part of the map, so one small
defect is not averaged away by the clean carton around it.

The encoder is the part under study and is chosen by name: any `timm` model
that can return feature maps at several scales ("resnet34", "mobilevit_xxs",
...). The decoder, the scoring and everything around them stay the same.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DECODER_WIDTHS = (256, 128, 64, 64)
TOP_K = 40                       # pixels of the map that make up the carton's score


def to_tensor(crop_bgr: np.ndarray) -> torch.Tensor:
    """uint8 BGR crop (H x W x 3) -> normalised float tensor (3 x H x W)."""
    rgb = torch.from_numpy(np.ascontiguousarray(crop_bgr[:, :, ::-1])).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (rgb - mean) / std


class _Block(nn.Sequential):
    def __init__(self, cin: int, cout: int):
        super().__init__(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )


class DefectMapNet(nn.Module):
    def __init__(self, encoder: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.encoder_name = encoder
        self.encoder = timm.create_model(encoder, features_only=True, pretrained=pretrained)
        channels: List[int] = self.encoder.feature_info.channels()

        # Decoder: start from the coarsest features and add finer ones step by step
        blocks, cin = [], channels[-1]
        for i, skip in enumerate(reversed(channels[:-1])):
            cout = DECODER_WIDTHS[min(i, len(DECODER_WIDTHS) - 1)]
            blocks.append(_Block(cin + skip, cout))
            cin = cout
        self.decoder = nn.ModuleList(blocks)
        self.head = nn.Conv2d(cin, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits, 1 x H x W per image, at the input resolution."""
        feats = self.encoder(x)
        y = feats[-1]
        for block, skip in zip(self.decoder, reversed(feats[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = block(torch.cat([y, skip], dim=1))
        return F.interpolate(self.head(y), size=x.shape[-2:], mode="bilinear", align_corners=False)


def carton_score(logits: torch.Tensor, top_k: int = TOP_K) -> torch.Tensor:
    """One score per image (0..1): the mean of the `top_k` highest values of the
    lightly smoothed probability map."""
    prob = F.avg_pool2d(torch.sigmoid(logits), kernel_size=5, stride=1, padding=2)
    return prob.flatten(1).topk(top_k, dim=1).values.mean(dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(path, model: DefectMapNet, **extra) -> None:
    torch.save({"encoder": model.encoder_name, "state": model.state_dict(), **extra}, str(path))


def load_checkpoint(path, device: Optional[str] = None):
    """Returns (model in eval mode, the dict of extra values stored with it)."""
    ckpt = torch.load(str(path), map_location=device or "cpu", weights_only=False)
    model = DefectMapNet(ckpt["encoder"], pretrained=False)
    model.load_state_dict(ckpt["state"])
    model.eval()
    if device:
        model.to(device)
    return model, {k: v for k, v in ckpt.items() if k != "state"}
