"""
Shared data handling for the label locator: file locations, annotations and
the export of the YOLO training set. Used by annotate_aligned.py and
train_and_eval_locator.py.

Annotations are stored in ORIGINAL image coordinates (four corner points),
so they stay valid if the alignment settings are changed later.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import DATA_DIR, RAW_FILTERED_DIR, WEIGHTS_DIR, RESULTS_DIR, SEED
from label_locator import (
    LocatorConfig, CLASS_NAMES, CLS_UPRIGHT, CLS_UPSIDE_DOWN, Box,
    align_carton, apply_affine, box_corners, bounding_box, rotate_box_180, ink_mask,
)

MANIFEST_PATH = RAW_FILTERED_DIR / "tapal_session_split.json"       # written by annotate.py (split by frame)
ANNOTATIONS_PATH = RAW_FILTERED_DIR / "print_region_quads.json"     # written by annotate_aligned.py
YOLO_ALIGNED_DIR = DATA_DIR / "yolo_print_region_aligned"
CROPS_ALIGNED_DIR = DATA_DIR / "crops_aligned"
LOCATOR_WEIGHTS = WEIGHTS_DIR / "yolov8n_print_region_aligned.pt"
LOCATOR_STATS = WEIGHTS_DIR / "locator_stats.json"
LOCATOR_RESULTS_DIR = RESULTS_DIR / "locator"

SPLITS = [("train_ok", "train"), ("val_ok", "val"), ("test_ok", "test"), ("test_not_ok", "test")]

HAS_PRINT, MISSING_PRINT, WRONG_VIEW = "has_print", "missing_print", "wrong_view"


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing {MANIFEST_PATH}. Run filter_and_split_sessions() in annotate.py first.")
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
    def build(cls, img_bgr: np.ndarray, cfg: LocatorConfig, upside_down: bool) -> "UprightView":
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


def _yolo_line(cls_id: int, box: Box, w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    xc, yc = (x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    return f"{cls_id} {min(max(xc, 0), 1):.6f} {min(max(yc, 0), 1):.6f} {min(max(bw, 0.005), 1):.6f} {min(max(bh, 0.005), 1):.6f}\n"


def remove_print(view_img: np.ndarray, box: Box, face: Box, cfg: LocatorConfig) -> np.ndarray:
    """Paint the ink out of the print block to make a blank-flap training
    example from a good carton (real blank flaps stay in the test set)."""
    h, w = view_img.shape[:2]
    pad_x, pad_y = 0.06 * (box[2] - box[0]), 0.15 * (box[3] - box[1])
    region = (max(0, box[0] - pad_x), max(0, box[1] - pad_y), min(w, box[2] + pad_x), min(h, box[3] + pad_y))
    x1, y1 = int(round(region[0])), int(round(region[1]))
    local = cv2.dilate(ink_mask(view_img, region, face, cfg), np.ones((3, 3), np.uint8), iterations=2)
    full = np.zeros((h, w), dtype=np.uint8)
    full[y1:y1 + local.shape[0], x1:x1 + local.shape[1]] = local
    return cv2.inpaint(view_img, full, 4, cv2.INPAINT_TELEA)


def export_yolo_dataset(cfg: Optional[LocatorConfig] = None, blank_fraction: float = 0.3) -> dict:
    """Write the aligned YOLO dataset.

    train / val : each good carton upright (class 0), the same frame turned
                  180 degrees (class 1), and for a share of them a blank-flap
                  copy with no label.
    test        : every annotated image upright, for measuring box accuracy.
    """
    cfg = cfg or LocatorConfig()
    rng = random.Random(SEED)
    manifest, annotations = load_manifest(), load_annotations()
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = YOLO_ALIGNED_DIR / sub / split
            d.mkdir(parents=True, exist_ok=True)
            for old in d.glob("*.*"):
                old.unlink()

    counts = {"written": 0, "not_annotated": 0, "no_face": 0, "blank_negatives": 0}

    def write(split: str, stem: str, image: np.ndarray, line: str) -> None:
        cv2.imwrite(str(YOLO_ALIGNED_DIR / "images" / split / f"{stem}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (YOLO_ALIGNED_DIR / "labels" / split / f"{stem}.txt").write_text(line)
        counts["written"] += 1

    for split_key, split in SPLITS:
        for idx, item in enumerate(manifest[split_key]):
            entry = annotations.get(item["path"])
            if entry is None:
                counts["not_annotated"] += 1
                continue
            img = cv2.imread(item["path"])
            if img is None:
                continue
            view = UprightView.build(img, cfg, bool(entry.get("upside_down")))
            if not view.found:
                counts["no_face"] += 1
                if split == "test":   # kept as a background image; the locator must reject it
                    write(split, f"{split_key}_{idx:04d}", img, "")
                continue
            h, w = view.image.shape[:2]
            box = view.box_of(entry) if entry["status"] == HAS_PRINT else None
            stem = f"{split_key}_{idx:04d}"
            write(split, stem, view.image, _yolo_line(CLS_UPRIGHT, box, w, h) if box else "")

            if split in ("train", "val") and box:
                turned = cv2.rotate(view.image, cv2.ROTATE_180)
                write(split, stem + "_r180", turned, _yolo_line(CLS_UPSIDE_DOWN, rotate_box_180(box, w, h), w, h))
                if rng.random() < blank_fraction:
                    write(split, stem + "_blank", remove_print(view.image, box, view.face_rect, cfg), "")
                    counts["blank_negatives"] += 1

    (YOLO_ALIGNED_DIR / "data.yaml").write_text(
        f"path: {YOLO_ALIGNED_DIR.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n")
    print(f"[Locator dataset] {counts}")
    return counts
