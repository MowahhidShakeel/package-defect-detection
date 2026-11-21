"""
Shared data handling for the preprocessing stage: file locations, the
dataset split and the print-block annotations.

Annotations are stored in ORIGINAL image coordinates (four corner points),
so they stay valid if the alignment settings are changed later.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from config import DATA_DIR, RAW_FILTERED_DIR, WEIGHTS_DIR, RESULTS_DIR
from preprocessing.align import (
    AlignConfig, Box, align_carton, apply_affine, box_corners, bounding_box, rotate_box_180,
)

MANIFEST_PATH = RAW_FILTERED_DIR / "tapal_session_split.json"       # written by split_dataset.py
ANNOTATIONS_PATH = RAW_FILTERED_DIR / "print_region_quads.json"     # written by annotate_aligned.py
CROP_REGION_PATH = WEIGHTS_DIR / "crop_region.json"                 # written by build_crops.py
CROPS_DIR = DATA_DIR / "crops_aligned"                              # written by build_crops.py
PREPROCESSING_RESULTS_DIR = RESULTS_DIR / "preprocessing"

SPLIT_KEYS = ["train_ok", "val_ok", "test_ok", "test_not_ok"]

HAS_PRINT, MISSING_PRINT, WRONG_VIEW = "has_print", "missing_print", "wrong_view"


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing {MANIFEST_PATH}. Run: python -m preprocessing.split_dataset")
    return json.loads(MANIFEST_PATH.read_text())


def load_annotations() -> dict:
    return json.loads(ANNOTATIONS_PATH.read_text()) if ANNOTATIONS_PATH.exists() else {}


def save_annotations(annotations: dict) -> None:
    ANNOTATIONS_PATH.write_text(json.dumps(annotations, indent=2))


@dataclass
class UprightView:
    """An image after carton alignment and, if needed, a 180-degree turn,
    with the point mappings between it and the original image."""
    image: np.ndarray
    M: np.ndarray               # original -> aligned
    upside_down: bool
    face_rect: Optional[Box]
    found: bool

    @classmethod
    def build(cls, img_bgr: np.ndarray, cfg: AlignConfig, upside_down: bool) -> "UprightView":
        al = align_carton(img_bgr, cfg)
        image, face = al.aligned, al.face_rect
        if upside_down:
            h, w = image.shape[:2]
            image = cv2.rotate(image, cv2.ROTATE_180)
            face = rotate_box_180(face, w, h) if face else None
        return cls(image, al.M, upside_down, face, al.found)

    def _turn(self, pts: np.ndarray) -> np.ndarray:
        h, w = self.image.shape[:2]
        return np.array([w, h], dtype=np.float64) - pts

    def to_upright(self, pts_original) -> np.ndarray:
        p = apply_affine(np.asarray(pts_original, dtype=np.float64), self.M)
        return self._turn(p) if self.upside_down else p

    def from_upright(self, pts_upright) -> np.ndarray:
        p = np.asarray(pts_upright, dtype=np.float64).reshape(-1, 2)
        p = self._turn(p) if self.upside_down else p
        return apply_affine(p, cv2.invertAffineTransform(self.M))

    def box_of(self, entry: dict) -> Optional[Box]:
        """Annotated print box in this view's coordinates."""
        if not entry.get("quad"):
            return None
        h, w = self.image.shape[:2]
        x1, y1, x2, y2 = bounding_box(self.to_upright(entry["quad"]))
        return max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)

    def quad_of(self, box: Box) -> list:
        """Box in this view -> four corner points in the original image."""
        return [[round(float(x), 2), round(float(y), 2)] for x, y in self.from_upright(box_corners(box))]
