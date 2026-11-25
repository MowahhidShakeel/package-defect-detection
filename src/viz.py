"""Small drawing helpers shared by the review sheets and the evaluation pictures."""
from __future__ import annotations

import cv2
import numpy as np


def caption_bar(img: np.ndarray, text: str, height: int = 26) -> np.ndarray:
    """Puts the text in a dark bar above the picture, white on dark, so it is
    readable whatever the picture shows. The text shrinks to fit the width."""
    scale = 0.5
    while scale > 0.3 and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > img.shape[1] - 12:
        scale -= 0.02
    bar = np.full((height, img.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(bar, text, (6, height - 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def heat_overlay(crop: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """The defect map laid over the crop. `prob` is 0..255 (uint8) or 0..1
    (float). Where the map is near 0 the crop is left as it is, so a good
    carton looks like a plain photo and only suspect areas are coloured."""
    p = prob.astype(np.float32) / 255.0 if prob.dtype == np.uint8 else np.clip(prob.astype(np.float32), 0.0, 1.0)
    heat = cv2.applyColorMap((p * 255).astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)
    alpha = (0.75 * np.sqrt(p))[:, :, None]
    return (crop.astype(np.float32) * (1.0 - alpha) + heat * alpha).astype(np.uint8)


def map_panel(crop: np.ndarray, prob: np.ndarray, text: str) -> np.ndarray:
    """Crop and defect map side by side under a caption bar."""
    return caption_bar(np.hstack([crop, heat_overlay(crop, prob)]), text)
